import torch
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from config import *
from plot_utils import plot_losses
from loss import loss_fn

def train_model(model:nn.Module,
          train_loader:torch.utils.data.DataLoader,
          val_loader:torch.utils.data.DataLoader,
          EPOCHS:int,
          LEARNING_RATE:float,
          device='cuda',
          logs=False,
          generate_plots=True,
          )->None:
    '''
    Train and save the model, also save the logs for `tensorboardX`.
    
    Input:
    -----
        `model`: model to train \n
        `train_loader`: data loader for training data \n
        `val_loader`: data loader for validation data \n
        `EPOCHS`: number of epochs to train the model \n
        `LEARNING_RATE`: learning rate for the optimizer \n
        `device`: device to run the model on (CPU or GPU) \n
        `logs`: whether to save the logs for `tensorboardX` or not \n
    Output:
    ------
        None
    '''
    
    # Move to device
    model.to(device)
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    # Define the summary writer
    if logs:
        writer = SummaryWriter(DIR_PATH + f'/logs/dataset_{FILE_NAME[:-4]}_seq_len_{SEQ_LEN}_pred_len_{PRED_LEN}_batch_size_{BATCH_SIZE}_lr_{LEARNING_RATE}')
    
    # Lists to store losses
    train_losses_over_epochs = []
    val_losses_over_epochs = []
    
    # Train the model
    for epoch in range(EPOCHS):
        losses = []
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')):
            # Get data to cuda if possible
            data = data.to('cuda')
            targets = targets.to('cuda')

            # forward
            y_pred = model(data, PRED_LEN)
            loss = loss_fn(y_pred, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())

            # gradient descent step
            optimizer.step()

        # Validate
        val_losses = []
        with torch.inference_mode():
            for batch_idx, (data, targets) in enumerate(val_loader):
                # Get data to cuda if possible
                data = data.to('cuda')
                targets = targets.to('cuda')

                # forward
                y_pred = model(data, PRED_LEN)
                val_loss = loss_fn(y_pred, targets)
                val_losses.append(val_loss.item())

        # Save losses
        train_losses_over_epochs.append(sum(losses)/len(losses))
        val_losses_over_epochs.append(sum(val_losses)/len(val_losses))
        
        # Print metrics
        print(f'Epoch {epoch+1}/{EPOCHS}: Train Loss: {sum(losses)/len(losses):.4f}, Val Loss: {sum(val_losses)/len(val_losses):4f}')
        print('-'*60)

        if logs:
            writer.add_scalar('Training loss', sum(losses)/len(losses), global_step=epoch)
            writer.add_scalar('Validation loss', sum(val_losses)/len(val_losses), global_step=epoch)
        
        # Adjust the learning rate
        scheduler.step(sum(val_losses)/len(val_losses))

    # Save the model
    torch.save(model.state_dict(), DIR_PATH + f'/models/{SYMBOL}/model_E{EPOCHS}_P{PRED_LEN}_S{torch.initial_seed()}.pth')
    
    if generate_plots: # Generate plots
        plot_losses(train_losses_over_epochs, val_losses_over_epochs)
    if logs: # Close the summary writer
        writer.close()