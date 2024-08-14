# Import necessary libraries
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import torch
import numpy as np

# Import custom modules
from config import *

def plot_losses(train_losses, val_losses, file_name=None) -> None:
    '''
    Plots losses generated during training
    
    Saves plot in the plots folder
    
    Input:
    -----
        `train_losses`: List of training losses over epochs \n
        `val_losses`: List of valdation losses over epochs \n
        `file_name`: Name of the file to save the plot \n
    Output:
    ------
        None
    '''

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label='Training loss', color='blue',
             alpha=0.7, linewidth=2, marker='o', markersize=5)
    plt.plot(val_losses, label='Validation loss', color='red',
             alpha=0.7, linewidth=2, marker='s', markersize=5)
    plt.legend()
    plt.title('Training/Validation Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Annotate the final values
    plt.annotate(f"Train Loss: {train_losses[-1]:.2f}", (len(train_losses)-1, train_losses[-1]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center',
                 alpha=0.7,
                 bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.2))
    plt.annotate(f"Val Loss: {val_losses[-1]:.2f}", (len(val_losses)-1, val_losses[-1]+max(train_losses)/10),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center',
                 alpha=0.7,
                 bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.2))

    # plt.show()

    if file_name:
        fig.savefig(DIR_PATH + '/plots/' + file_name, dpi=200)
    else:
        fig.savefig(DIR_PATH + f'/plots/{SYMBOL}/train-val-losses-' +
                    datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.png', dpi=200)

def plot_predictions(model, test_dataset, file_name=None) -> None:
    '''
    Plots predictions vs actual values in 6 subplots choosen randomly from the test dataset
    
    Saves plot in plots folder
    
    Input:
    -----
        `model`: model to inference with \n
        `test_loader`: data loader for test data  \n
        `file_name`: Name of the file to save the plot \n
    Output:
    ------
        None
    '''
    model.to(DEVICE) # Move the model to the device
    
    plt.style.use('ggplot')
    
    fig, ax = plt.subplots(3, 2, figsize=(10, 8))
    
    for i in range(3):
        for j in range(2):
            # Choose a random sample from the test dataset
            idx = np.random.randint(0, len(test_dataset))
            data, target = test_dataset[idx]

            # Make a prediction
            with torch.no_grad():
                y_pred = model(data.unsqueeze(0).to(DEVICE), PRED_LEN).cpu()
                y_pred = y_pred.squeeze(0) # remove the batch dimension

            # Concatenate the last close price from the data with the target
            actual_prices = torch.cat((data[:, 0], target[:, 0])).numpy()
            predictions = torch.cat((data[-1, 0].reshape(1), y_pred[:, 0])).numpy()
            prediction_x_range = np.arange(len(actual_prices)-PRED_LEN - 1, len(actual_prices))
            
            # Plot the actual values
            ax[i, j].plot(actual_prices, label='Actual', color='blue', alpha=0.7, linewidth=2)
            # Plot the predictions
            ax[i, j].plot(prediction_x_range, predictions, label='Predicted', color='red', alpha=0.7, linewidth=2)

    # Super y label
    fig.text(0.04, 0.5, 'Price ($)', va='center', rotation='vertical', size=12)
    # Super x label
    fig.text(0.5, 0.04, 'Days', ha='center', size=12)
    # Super legend
    fig.legend(['Actual', 'Predicted'], loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)
    
    # Set the title for the entire plot
    fig.suptitle('Predictions vs Actual Closing Prices', size=14)
    # Set the spacing between the title and the subplots
    fig.subplots_adjust(top=0.87)
    
    # plt.show()
    
    # Save the plot
    if file_name:
        fig.savefig(DIR_PATH + '/plots/' + file_name, dpi=200)
    else:
        fig.savefig(DIR_PATH + f'/plots/{SYMBOL}/predictions-' +
                    datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.png', dpi=200)

def plot_simulation(portfolio:dict, stock_data:pd.DataFrame, file_name=None)->None:
    '''
    Plots the simulation results.
    Saves plot in plots folder
    
    Input:
    -----
        `portfolio`: A dictionary \n
        `stock_data`: A pandas dataframe \n
        `file_name`: Name of the file to save the plot \n
    Output:
    ------
        None
    '''
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
        
    # Plot close prices
    ax1.plot(stock_data['close'].values, label='Close Price', color='blue', alpha=0.8, linewidth=1.5, zorder=2)
    
    # Plot buy/sell transactions using scatter plot
    buy_indices = [index for index, action in enumerate(portfolio['action_history']) if action=='buy']
    sell_indices = [index for index, action in enumerate(portfolio['action_history']) if action=='sell']
    # hold_indices = [index for index, action in enumerate(portfolio['action_history']) if action=='hold']
    
    marker_size = max(10, 80 - len(portfolio['action_history'])//10) # Increase the marker size for fewer transactions
    ax1.scatter(buy_indices, stock_data['close'].iloc[buy_indices], color='green', alpha=1, label='Buy', marker='^', s=marker_size, zorder=3)
    ax1.scatter(sell_indices, stock_data['close'].iloc[sell_indices], color='red', alpha=1, label='Sell', marker='v', s=marker_size, zorder=3)
    # ax1.scatter(hold_indices, stock_data['close'].iloc[hold_indices], color='gray', alpha=0.5, label='Hold')
    
    
    # Plot cummulative returns
    percentage_returns = [100*x for x in portfolio['Cummulative Return']]
    ax2.plot(percentage_returns, label='Cummulative Return', color='black', alpha=0.7, linewidth=1, linestyle='--', zorder=0)
    # Plot cummulative returns with area under the curve
    ax2.fill_between(range(len(percentage_returns)), 0, percentage_returns, color='gray', alpha=0.3)
    
    
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Price ($)')
    ax2.set_ylabel('Cummulative Return (%)')
    
    # Title based on net profit/loss
    final_cumulative_return = percentage_returns[-1]    
    if final_cumulative_return>0:
        ax1.set_title('Simulation Results: Final Returns: {:.2f}%'.format(final_cumulative_return), color='green')
    else:
        ax1.set_title('Simulation Results: Final Returns: {:.2f}%'.format(final_cumulative_return), color='red')
    
    # Remove horizontal gridlines for the secondary axis
    ax2.grid(False)

    # Legends for ax1 and ax2
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.85))

    # plt.show()
    
    # Save the plot
    if file_name: # If a file name is provided
        fig.savefig(DIR_PATH + '/plots/' + file_name, dpi=200, bbox_inches='tight')
    else: # Save the plot with a timestamp
        fig.savefig(DIR_PATH + f'/plots/{SYMBOL}/simulation-' +
                    datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.png', dpi=200, bbox_inches='tight')