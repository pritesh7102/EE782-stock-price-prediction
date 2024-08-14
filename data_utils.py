import pandas as pd
import os

def create_folder(folder_path: str) -> None:
    '''
    Creates a folder at the specified path if it doesn't exist.
    
    Input:
    -----
        `folder_path`: Path to the folder to be created \n
    Output:
    ------
        None
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def read_data(file_path) -> pd.DataFrame:
    '''
    Reads the data from the specified file path and returns a pandas DataFrame.
    
    Input:
    -----
        `file_path`: Path to the file containing the data \n
    Output:
    ------
        `df`: Pandas DataFrame containing the data \n
    '''
    # Read the data
    df = pd.read_csv(file_path, sep=',', index_col=False, header=None, names=['date', 'open', 'high', 'low', 'close', 'volume'])
    return df

def RSI(series: pd.Series, period: int) -> pd.Series: 
    '''
    Calculates the Relative Strength Index (RSI) of a pandas series.
    
    Input:
    -----
        `series`: Pandas series containing the data \n
        `period`: Period over which the RSI is to be calculated \n
    Output:
    ------
        `rsi`: Pandas series containing the RSI values \n
    '''
    # Calculate price changes
    delta = series.diff(1)
    
    # Separate positive and negative price changes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and average loss over the specified period
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def pre_process_data(df: pd.DataFrame)-> pd.DataFrame:
    '''
    Makes a copy and pre-processes the data and returns a pandas DataFrame.
        - Resample and creates day-wise OHLCV data
        - Drops NAN values
        - Rephare data into different columns: ['year', 'month', 'day', 'day_of_week']
        - Adds indicators based on closing price: ['ema_5', 'ema_10', 'ema_20', 'RSI']
             
    Input:
    -----
        `df`: Pandas DataFrame containing the OHLCV & datetime data \n
    Output:
    ------
        `df_resampled`: Pandas DataFrame containing the pre-processed data \n
    '''
    df_resampled = df.copy()
    
    # Convert the 'date' column to a datetime object
    df_resampled['date'] = pd.to_datetime(df_resampled['date'])
    
    # Resample the data to daily OHLCV using the 'date' column as the index
    df_resampled = df_resampled.resample('D', on='date').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'})

    # Reset the index to have the date as a separate column
    df_resampled.reset_index(inplace=True)

    # Remove rows with NaN values
    df_resampled.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

    # Add additional date-related columns if needed
    df_resampled['year'] = df_resampled['date'].dt.year
    df_resampled['month'] = df_resampled['date'].dt.month
    df_resampled['day'] = df_resampled['date'].dt.day
    df_resampled['day_of_week'] = df_resampled['date'].dt.dayofweek

    # Delete the date column
    df_resampled.drop(['date'], axis=1, inplace=True)


    # EMAs
    df_resampled['ema_5'] = df_resampled['close'].ewm(span=5, adjust=False).mean()
    df_resampled['ema_10'] = df_resampled['close'].ewm(span=10, adjust=False).mean()
    df_resampled['ema_20'] = df_resampled['close'].ewm(span=20, adjust=False).mean()

    # RSI
    df_resampled['RSI'] = RSI(df_resampled['close'], 14)
    df_resampled['RSI'].fillna(0, inplace=True)

    df_resampled.reset_index(inplace=True, drop=True)

    return df_resampled