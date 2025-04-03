import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # expansion factor of 2 instead of 4
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2, bias=True),
            nn.ReLU(),  # Using ReLU instead of GELU 
            nn.Linear(embed_dim * 2, embed_dim, bias=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):

        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        
        return x

class MAEEncoder(nn.Module):
    def __init__(self, 
                 img_size=36,
                 patch_size=3, 
                 in_channels=3, 
                 embed_dim=192, 
                 num_layers=12, 
                 num_heads=3, 
                 dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2
        # For 36x36 images with 3x3 patches, we get 12x12=144 patches
        # +1 for the dummy patch used for classification
        self.total_patches = self.num_patches + 1
        
        # Patch embedding projection, no bias
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim, bias=False)
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_patches, embed_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x, mask_ratio=0.75):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # Extract patches: [B, num_patches, patch_size*patch_size*C]
        patches = self._extract_patches(x)
        
        # Add dummy patch (all zeros) for classification
        dummy_patch = torch.zeros(B, 1, patches.shape[-1], device=patches.device)
        patches = torch.cat([patches, dummy_patch], dim=1)
        
        # Create masking 
        num_patches = patches.shape[1] - 1  # Excluding the dummy patch
        len_keep = int(num_patches * (1 - mask_ratio))
        
        # Generate random indices to keep
        noise = torch.rand(B, num_patches, device=patches.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        dummy_idx = torch.ones(B, 1, device=patches.device, dtype=torch.long) * (num_patches)
        ids_keep = torch.cat([ids_keep, dummy_idx], dim=1)
        
        # Keep only unmasked patches
        patches_keep = torch.gather(patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, patches.shape[-1]))
        
        # Project patches to embedding dimension
        x = self.proj(patches_keep)
        
        # Create position embedding indices for the kept patches
        pos_ids_keep = torch.cat([ids_keep[:, :-1], dummy_idx], dim=1)
        
        # Add positional embeddings
        pos_embed = torch.gather(
            self.pos_embed.repeat(B, 1, 1), 
            dim=1, 
            index=pos_ids_keep.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        )
        x = x + pos_embed
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply final normalization and projection
        x = self.norm(x)
        x = self.out_proj(x)
        
        return x, ids_keep, ids_shuffle, mask_ratio
    
    def _extract_patches(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Reshape into patches
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        # x: [B, C, H//patch_size, W//patch_size, patch_size, patch_size]
        
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        # x: [B, H//patch_size, W//patch_size, C, patch_size, patch_size]
        
        x = x.view(B, -1, C * self.patch_size * self.patch_size)
        # x: [B, num_patches, C*patch_size*patch_size]
        
        return x

# MAE Decoder
class MAEDecoder(nn.Module):
    def __init__(self, 
                 img_size=36,
                 patch_size=3, 
                 in_channels=3, 
                 embed_dim=192, 
                 num_layers=4, 
                 num_heads=3, 
                 dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2
        # +1 for the dummy patch
        self.total_patches = self.num_patches + 1
        
        # Shared mask embedding (token)
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learnable positional embeddings for decoder
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_patches, embed_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Final norm layer
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection to patch pixels
        self.out_proj = nn.Linear(embed_dim, patch_size * patch_size * in_channels, bias=False)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize positional embeddings and mask token
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_embed, std=0.02)
        
    def forward(self, x, ids_keep, ids_shuffle, mask_ratio):
        """
        x: encoded patches [B, len_keep, embed_dim]
        ids_keep: indices of kept patches [B, len_keep]
        ids_shuffle: shuffled indices of patches [B, num_patches]
        mask_ratio: ratio of patches to mask
        """
        B = x.shape[0]
        
        # Create full representation with mask tokens
        x_full = self.mask_embed.repeat(B, self.total_patches, 1)
        
        # Put encoded tokens back at their positions using scatter
        batch_indices = torch.arange(B, device=x.device).view(-1, 1).repeat(1, ids_keep.size(1))
        x_full[batch_indices, ids_keep] = x
        
        # Add positional embeddings
        x_full = x_full + self.pos_embed
        
        # Apply transformer layers
        for layer in self.layers:
            x_full = layer(x_full)
        
        # Apply final normalization
        x_full = self.norm(x_full)
        
        # Project to patch pixels (excluding dummy patch)
        x_rec = self.out_proj(x_full[:, :-1, :])
        
        return x_rec
    
    def unpatchify(self, x_rec):
        """Convert patched representation back to image"""
        # x_rec: [B, num_patches, patch_size*patch_size*C]
        B = x_rec.shape[0]
        
        p = self.patch_size
        h = w = self.img_size // p
        C = self.in_channels
        
        x = x_rec.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        # x: [B, C, h, p, w, p]
        
        x = x.reshape(B, C, h * p, w * p)
        # x: [B, C, H, W]
        
        return x

# Complete MAE model
class MAE(nn.Module):
    def __init__(self, 
                 img_size=36,
                 patch_size=3, 
                 in_channels=3, 
                 embed_dim=192, 
                 encoder_layers=12,
                 decoder_layers=4,
                 num_heads=3, 
                 dropout=0.1,
                 mask_ratio=0.75,
                 decoder_embed_dim=192):
        super().__init__()
        
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.decoder = MAEDecoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=decoder_embed_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.mask_ratio = mask_ratio
    
    def forward(self, x):
        # Encode patches
        encoded, ids_keep, ids_shuffle, mask_ratio = self.encoder(x, self.mask_ratio)
        
        # Decode patches
        x_rec = self.decoder(encoded, ids_keep, ids_shuffle, mask_ratio)
        
        # Unpatchify to get image reconstruction
        recon = self.decoder.unpatchify(x_rec)
        
        # Original patches for loss calculation
        patches = self.encoder._extract_patches(x)
        
        # Calculate loss as per the paper: MSE_masked + α * MSE_unmasked
        # where α is the discounting factor (0.1)
        
        # Determine which patches were masked
        B = x.shape[0]
        num_patches = patches.shape[1]
        
        # Create a mask tensor (1 for masked, 0 for unmasked)
        mask = torch.ones(B, num_patches, device=x.device)
        
        # Set unmasked positions to 0
        unmasked_indices = ids_keep[:, :-1]  # Exclude dummy patch
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).repeat(1, unmasked_indices.shape[1])
        mask[batch_indices.flatten(), unmasked_indices.flatten()] = 0
        
        # Calculate MSE for masked and unmasked patches
        patched_recon = self.encoder._extract_patches(recon)
        
        # MSE for masked patches
        mse_masked = F.mse_loss(
            patched_recon[mask.bool()], 
            patches[mask.bool()]
        )
        
        # MSE for unmasked patches
        mse_unmasked = F.mse_loss(
            patched_recon[~mask.bool()], 
            patches[~mask.bool()]
        )
        
        # Total loss with discounting factor α = 0.1
        loss = mse_masked + 0.1 * mse_unmasked
        
        return loss, recon

# Fine-tuning ViT model for classification
class ViT_Classifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        
        # Freeze encoder if needed
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        # Only use the last embedding (dummy patch) for classification
        self.classifier = nn.Linear(encoder.embed_dim, num_classes)
    
    def forward(self, x):
        # Pass through encoder without masking (mask_ratio=0)
        # Use only the encoder part
        encoded, _, _, _ = self.encoder(x, mask_ratio=0)
        
        # Use the dummy patch embedding (last one) for classification
        cls_token = encoded[:, -1]
        
        # Classify
        logits = self.classifier(cls_token)
        
        return logits