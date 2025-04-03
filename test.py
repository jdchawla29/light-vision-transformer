from model import MAE
import torch
import torch.nn as nn

def test_mae():
    # Define the model
    model = MAE()

    # Create a random input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 36, 36)

    # Forward pass through the model
    _, output_tensor = model(input_tensor)

    # Check the output shape
    assert output_tensor.shape == (1, 3, 36, 36), f"Expected output shape (1, 3, 36, 36), got {output_tensor.shape}"


if __name__ == "__main__":
    test_mae()
    print("MAE model test passed.")