import torch
from torch.utils.data import DataLoader

from config import *
from dataloader import StockData
from model import PricePredictor
from data_utils import read_data, pre_process_data, create_folder
from train import train_model
from test import test_model
from plot_utils import plot_predictions
from simulator import simulate

if __name__ == '__main__':
    # Set the seed for reproducibility
    torch.manual_seed(42)
    
    # Read the data
    df = read_data(FILE_PATH)
    df = pre_process_data(df)
    
    # Calculate the train, validation and test sizes
    train_size = int(TRAIN_RATIO * len(df))
    val_size = int(VAL_RATIO * len(df))
    test_size = len(df) - train_size - val_size

    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")
    
    # Create train, validation and test sets sequentially from the dataframe
    train_dataset = StockData(df[:train_size], SEQ_LEN, PRED_LEN, INPUT_COLS, OUTPUT_COLS)
    val_dataset = StockData(df[train_size:train_size+val_size], SEQ_LEN, PRED_LEN, INPUT_COLS, OUTPUT_COLS)
    test_dataset = StockData(df[train_size+val_size:], SEQ_LEN, PRED_LEN, INPUT_COLS, OUTPUT_COLS)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create a model
    model = PricePredictor(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
    
    # Create a folder to store the plots
    create_folder(DIR_PATH + f'/models/{SYMBOL}')
    create_folder(DIR_PATH + f'/plots/{SYMBOL}')
    create_folder(DIR_PATH + '/logs')
    
    # Call train function and save the model
    train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)
    
    # Call the test function to print loss and plot predictions
    test_model(model, test_loader)
    
    # Plot predictions
    plot_predictions(model, test_dataset)
    
    # Call the simulator function
    simulate(model, test_dataset)
    
    # Check the plots folder for the results
    # All results are stored in plots folder