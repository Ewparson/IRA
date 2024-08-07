import yfinance as yf
import ta

def collect_data(ticker="AAPL", period="max"):
    # Download historical data
    data = yf.download(ticker, period=period)
    
    # Calculate RSI
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    
    # Calculate VWAP
    data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(data['High'], data['Low'], data['Close'], data['Volume']).volume_weighted_average_price()
    
    # Calculate MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()
    
    # Calculate Moving Averages
    data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
    data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
    data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    data['EMA_200'] = ta.trend.EMAIndicator(data['Close'], window=200).ema_indicator()
    
    # Save data to CSV
    file_name = f"data/{ticker}_AAPL_historical_data"
    data.to_csv(file_name)
    print(f"Data collection complete. File saved as: {file_name}")
    return file_name
