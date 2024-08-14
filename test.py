import torch
from torch import nn
import matplotlib.pyplot as plt

from config import *
from loss import loss_fn

def test_model(model, test_loader, device='cuda')->None:
    '''
    Tests the model and print the loss.
    
    Input:
    -----
        `model`: model to test \n
        `test_loader`: data loader for test data \n
        `device`: device to run the model on (CPU or GPU). default: 'cuda' \n
        
    Output:
    ------
        None
    '''
    # Move to device
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
        
    # Test
    test_losses = []
    with torch.inference_mode():
        for batch_idx, (data, targets) in enumerate(test_loader):
            # Get data to cuda if possible
            data = data.to('cuda')
            targets = targets.to('cuda')

            # Forward pass
            y_pred = model(data, PRED_LEN)
            test_loss = loss_fn(y_pred, targets)
            test_losses.append(test_loss.item())
    print(f'Test Loss: {sum(test_losses)/len(test_losses)}')