import torch
import os
from torch._C import device
from torch.optim import AdamW
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from data import get_pretraining_datasets
from model import MAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Linear warmup scheduler
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # Cosine annealing
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1. + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_mae(model, train_dataset, batch_size=1000, base_lr=1.5e-4, 
                weight_decay=0.05, epochs=4000, warmup_epochs=400, 
                save_dir='./checkpoints', save_interval=500):
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
    )
    
    lr = base_lr * batch_size / 256
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)

    os.makedirs(save_dir, exist_ok=True)

    model.train()
    total_loss = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (inputs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            inputs = inputs.to(device)
            
            # Forward pass
            loss, _ = model(inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            batch_count += 1
        
        # Step learning rate scheduler
        scheduler.step()
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / batch_count
        total_loss.append(avg_epoch_loss)
        
        # Print status
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
            }, os.path.join(save_dir, f'mae_epoch_{epoch+1}.pth'))
            
            # Save loss plot
            plt.figure(figsize=(10, 5))
            plt.plot(total_loss)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(save_dir, 'loss.png'))
            plt.close()
    
    return model, total_loss


def main():
    set_seed(42)
    
    # Parameters
    batch_size = 1408 
    base_lr = 1.5e-4
    weight_decay = 0.05
    epochs = 4000
    warmup_epochs = 400


    # Check which device is being used
    print(f"Using device: {device}")
    
    # Create model
    model = MAE().to(device)
    
    # Print model stats
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = encoder_params + decoder_params
    
    print(f"Encoder parameters: {encoder_params/1e6:.2f}M")
    print(f"Decoder parameters: {decoder_params/1e6:.2f}M")
    print(f"Total parameters: {total_params/1e6:.2f}M")
    
    # Get dataset
    cifar10_train, _ = get_pretraining_datasets()
    train_dataset = cifar10_train
    model_name = "Mae-ViT-C10"
    
    print("Training on 'CIFAR-10")

    model, _ = train_mae(
        model, 
        train_dataset, 
        batch_size=batch_size,
        base_lr=base_lr,
        weight_decay=weight_decay,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        save_dir=f'./checkpoints/{model_name}'
    )

if __name__ == "__main__":
    main()