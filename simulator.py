# I havent inlcuded short action in this simulation for a couple of reasons:
# 1. Shorting is not allowed in all markets
# 2. Shorting is not allowed in all accounts, you need a margin account to short
# 3. Return are amplified in short positions, both positive and negative
# 4. Risk of unlimited(theoritically infinte) losses as compared to finite losses in case of long positions
# 5. Buy-In-Risk, where the broker can force you to close your short position pre-maturely if the stock is hard to borrow
# 6. Short positon incure an interest rate need to be paid to the broker along with dividends paid to the lender. 
#    Simulating these would be a lot of work and not in the spirit of this assignment.

import pandas as pd
import torch
from torch import nn

from data_utils import read_data, pre_process_data
from plot_utils import plot_simulation
from config import *

def random_stratergy(portfolio, current_price)->(str, int):
    action = random.choice(['buy', 'sell', 'hold'])
    if action == 'buy':
        max_shares = portfolio['cash'] // (current_price * (1 + TRADE_COMMISION)) # Maximum number of shares that can be bought
        if max_shares == 0:
            return 'hold', 0
        else:
            return 'buy', max_shares
    elif action == 'sell':
        if portfolio['stock_holdings'] == 0:
            return 'hold', 0
        else:
            return 'sell', portfolio['stock_holdings']
    else:
        return 'hold', 0

def get_action(portfolio, current_price, predictions)->(str, int):
    # Look only at first 5 predictions
    predictions = predictions[:5]
    if predictions.mean() > current_price: # Buy if the price is predicted to go up
        max_shares = portfolio['cash'] // (current_price * (1 + TRADE_COMMISION)) # Maximum number of shares that can be bought
        if max_shares == 0:
            return 'hold', 0
        else:
            return 'buy', max_shares
    
    elif predictions.mean() < current_price: # Sell if the price is predicted to go down
        if portfolio['stock_holdings'] == 0:
            return 'hold', 0
        else:
            return 'sell', portfolio['stock_holdings']
    
    else: # Hold if the price is predicted to stay the same
        return 'hold', 0

def buy(portfolio, index, current_price, shares_to_buy):
    # Update portfolio
    cash_spent = shares_to_buy * current_price * (1 + TRADE_COMMISION)
    portfolio['cash'] -= cash_spent
    portfolio['stock_holdings'] += shares_to_buy
    portfolio['transaction_history'].append(f"Buy {shares_to_buy} shares on day {index}, price: {current_price:.2f}, Cash Spent: {cash_spent:.2f}")
    portfolio['action_history'].append('buy')
    portfolio['value'] = portfolio['cash'] + (portfolio['stock_holdings'] * current_price)
    portfolio['Cummulative Return'].append((portfolio['value'] - INITIAL_CASH) / INITIAL_CASH)
    
def sell(portfolio, index, current_price, shares_to_sell=None):    
    # Update portfolio
    cash_earned = shares_to_sell * current_price * (1 - TRADE_COMMISION) 
    portfolio['cash'] += cash_earned
    portfolio['stock_holdings'] -= shares_to_sell
    portfolio['transaction_history'].append(f"Sell {shares_to_sell} shares on day {index}, price: {current_price:.2f}, Cash Earned: {cash_earned:.2f}")
    portfolio['action_history'].append('sell')
    portfolio['value'] = portfolio['cash'] + (portfolio['stock_holdings'] * current_price)
    portfolio['Cummulative Return'].append((portfolio['value'] - INITIAL_CASH) / INITIAL_CASH)
    
def hold(portfolio, index, current_price):
    portfolio['transaction_history'].append(f"Hold on day {index}, price: {current_price:.2f}") # Append 'hold' to the transaction history
    portfolio['action_history'].append('hold') # Append 'hold' to the action history
    portfolio['value'] = portfolio['cash'] + (portfolio['stock_holdings'] * current_price) # Update the portfolio value, since the price has changed
    portfolio['Cummulative Return'].append((portfolio['value'] - INITIAL_CASH) / INITIAL_CASH) # Update the cummulative returns, since the price has changed

def simulate(model, test_dataset)->None:
    '''
    Simulates the trading of a stock using the predictions from the model.
    
    Input:
    -----
        `model`: model to train \n
        `test_dataset`: data loader for training data \n
    Output:
    ------
        None
    '''
    
    # Move to device and set to evaluation mode
    model.to(DEVICE)
    model.eval()
    
    print(f'Simulating {len(test_dataset)} days')
    
    portfolio = {
        "cash": INITIAL_CASH, # Amount of cash in hand
        "stock_holdings": 0, # Number of shares currently held
        "value": INITIAL_CASH, # Value of portfolio at the end of each day
        "transaction_history": [], # List of all transactions made
        "action_history": [], # List of all actions taken
        "Cummulative Return": [] # List of cummulative returns at the end of each day
    }

    optimal_actions_taken = 0
    
    for index, (data, target) in enumerate(test_dataset):
        # Get data to cuda if possible
        data = data.to(DEVICE).unsqueeze(0) # Add the batch dimension
        target = target.to(DEVICE)[:, 0].cpu().detach().numpy().reshape(-1) # Choose close price from the target and move to cpu

        # Forward pass with no gradient calculation
        with torch.inference_mode():
            # Model output will be of shape (batch_size, pred_len, output_size)
            y_pred = model(data, PRED_LEN).squeeze(0) # Remove the batch dimension
            predictions = y_pred[:, 0].cpu().detach().numpy().reshape(-1) # Choose close price from the predictions and move to cpu
        
        current_price = data[-1, 0, 0].cpu().item() # Get the close price for the current day and move to cpu
        
        action, trade_amount = get_action(portfolio, current_price, predictions) # Get the action and number of shares based on the predictions
        optimal_action, _ = get_action(portfolio, current_price, target) # Get the optimal action based on the actual prices
        
        if action == optimal_action: # If the action taken is optimal
            optimal_actions_taken += 1 # Increment the optimal actions taken counter
        
        # Simulate the action
        if action=='buy':
            buy(portfolio, index, current_price, trade_amount)
        elif action=='sell':
            sell(portfolio, index, current_price, trade_amount)
        else:
            hold(portfolio, index, current_price)
    
    # Print performance metrics
    print(f"Initial Cash: {INITIAL_CASH}")
    print("Final Portfolio Value:", portfolio['value'])
    print(f"Total Returns: {100*portfolio['Cummulative Return'][-1]:.2f}%")
    print(f"Optimal Actions Taken: {100*optimal_actions_taken/len(test_dataset):.2f}%")
    
    # Plot the results
    plot_simulation(portfolio, test_dataset.df[:len(portfolio['Cummulative Return'])])
    

if __name__ == '__main__':
    import random
    # Test simulation of random stratergy
    print("Testing random action stratergy")
    
    # Load stock price data into a DataFrame
    stock_data = read_data(FILE_PATH)
    stock_data = pre_process_data(stock_data)[-100:] # Last 5% of the data
    print(f"Simulating {stock_data.__len__()} days")

    # Initialize portfolio and other parameters
    portfolio = {
        "cash": INITIAL_CASH, # Amount of cash in hand
        "stock_holdings": 0, # Number of shares currently held
        "value": INITIAL_CASH, # Value of portfolio at the end of each day
        "transaction_history": [], # List of all transactions made
        "action_history": [], # List of all actions taken
        "Cummulative Return": [] # List of cummulative returns at the end of each day
    }

    # Simulation loop
    for index, row in stock_data.iterrows():
        
        action, trade_amount = random_stratergy(portfolio, row['close']) # Get the action and number of shares based on the predictions
        
        if action=='buy':
            buy(portfolio, index, current_price=row['close'], shares_to_buy=trade_amount)
        elif action=='sell':
            sell(portfolio, index, current_price=row['close'], shares_to_sell=trade_amount)
        else:
            hold(portfolio, index, current_price=row['close'])
    

    # Print performance metrics
    print("Final Portfolio Value:", portfolio['value'])
    print(f"Total Returns: {100*portfolio['Cummulative Return'][-1]:.2f}%")

    # Plot the results
    plot_simulation(portfolio, stock_data, file_name='random_action_stratergy.png')