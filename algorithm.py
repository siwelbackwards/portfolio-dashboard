import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime


#DATA COLLECTION
yf.pdr_override()
# list of tickers
NYSE100tickers = [
    "MMM", "ABT", "ACN", "AIG", "MO", "T", "BAC", "BK", "BAX", "BRK-B", "BA", "BMY", 
    "CAT", "C", "KO", "CL", "GLW", "DHR", "DE", "DVN", "D", "DUK", "DD", "LLY", 
    "EMR", "EOG", "EXC", "XOM", "FDX", "F", "BEN", "FCX", "GD", "GE", "GIS", "GM", 
    "GS", "HAL", "HD", "HON", "HPQ", "IBM", "ITW", "JNJ", "JPM", "KMB", "LVS", "LMT", 
    "LOW", "MA", "MCD", "MRK", "MET", "MS", "NKE", "NOV", "OXY", "PEP", "PFE", "PM", 
    "PNC", "PG", "PRU", "RTX", "SLB", "SPG", "SO", "SCCO", "TGT", "TRV", "USB", "UNP", 
    "UPS", "UNH", "VZ", "V", "WBA", "WMT", "DIS", "WFC", "YUM"
]

NIFTY50tickers = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", 
    "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL", "BRITANNIA", "CIPLA", 
    "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", 
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", 
    "INDUSINDBK", "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "LTIM", 
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE", 
    "SBILIFE", "SBIN", "SUNPHARMA", "TATAMOTORS", "TATASTEEL", "TCS", 
    "TATACONSUM", "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO"
]

EURONEXT100tickers = [
    "ABI.BR", "AC.PA", "ADYEN.AS", "AGS.BR", "AGN.AS", "ADP.PA", "AD.AS", "AIR.PA", 
    "AI.PA", "AKER.OL", "AKZA.AS", "ALO.PA", "MT.AS", "ARGX.BR", "AKE.PA", "ASM.AS", 
    "ASML.AS", "CS.PA", "ATO.PA", "BB.PA", "BIM.PA", "BNP.PA", "EN.PA", "BVI.PA", 
    "CAP.PA", "CA.PA", "ACA.PA", "CRG.IR", "BN.PA", "DSY.PA", "DNB.OL", "EDEN.PA", 
    "EDP.LS", "FGR.PA", "ENGI.PA", "EL.PA", "ENX.PA", "FDJ.PA", "FLTR.IR", "GALP.LS", 
    "GFC.PA", "SK.PA", "HEIA.AS", "INGA.AS", "NK.PA", "JMT.LS", "DEC.PA", "KER.PA", 
    "KPN.AS", "LI.PA", "LR.PA", "MC.PA", "ML.PA", "ORA.PA", "RI.PA", "PHIA.AS", 
    "PRX.AS", "PUB.PA", "RAND.AS", "RCO.PA", "RNO.PA", "SHELL.AS", "RYAAY", "SAF.PA", 
    "SGO.PA", "SAN.PA", "SU.PA", "GLE.PA", "SW.PA", "SOLB.BR", "STLA", 
    "TEL.OL", "TEP.PA", "HO.PA", "TTE.PA", "UBI.PA", "UCB.BR", "UMI.BR", 
    "UNA.AS", "UMG.AS", "FR.PA", "VIE.PA", "DG.PA", "VIV.PA", "WKL.AS", "WLN.PA"
]

TASItickers = [
    "1010.SR", "1020.SR", "1030.SR", "1050.SR", "1060.SR", "1080.SR", "1111.SR", 
    "1120.SR", "1140.SR", "1150.SR", "1180.SR", "1182.SR", "1183.SR", "1201.SR", 
    "1202.SR", "1210.SR", "1211.SR", "1212.SR", "1213.SR", "1214.SR", "1301.SR", 
    "1302.SR", "1303.SR", "1304.SR", "1320.SR", "1321.SR", "1322.SR", "1810.SR", 
    "1820.SR", "1831.SR", "1832.SR", "1833.SR", "2001.SR", "2010.SR", "2020.SR", 
    "2030.SR", "2040.SR", "2050.SR", "2060.SR", "2070.SR", "2080.SR", "2081.SR", 
    "2082.SR", "2083.SR", "2090.SR", "2100.SR", "2110.SR", "2120.SR", "2130.SR", 
    "2140.SR", "2150.SR", "2160.SR", "2170.SR", "2180.SR", "2190.SR", "2200.SR", 
    "2210.SR", "2220.SR", "2222.SR", "2223.SR", "2230.SR", "2240.SR", "2250.SR", 
    "2270.SR", "2280.SR", "2281.SR", "2282.SR", "2283.SR", "2290.SR", "2300.SR", 
    "2310.SR", "2320.SR", "2330.SR", "2340.SR", "2350.SR", "2360.SR", "2370.SR", 
    "2380.SR", "2381.SR", "2382.SR", "3001.SR", "3002.SR", "3003.SR", "3004.SR", 
    "3005.SR", "3007.SR", "3008.SR", "3010.SR", "3020.SR", "3030.SR", "3040.SR", 
    "3050.SR", "3060.SR", "3080.SR", "3090.SR", "3091.SR", "3092.SR", "4001.SR", 
    "4002.SR", "4003.SR", "4004.SR", "4005.SR", "4006.SR", "4007.SR", "4008.SR", 
    "4009.SR", "4011.SR", "4012.SR"
]

SSE50tickers = [
    "600010.SS", "600028.SS", "600030.SS", "600031.SS", "600036.SS",
    "600048.SS", "600050.SS", "600089.SS", "600104.SS", "600111.SS",
    "600196.SS", "600276.SS", "600309.SS", "600406.SS", "600436.SS",
    "600438.SS", "600519.SS", "600690.SS", "600745.SS", "600809.SS",
    "600887.SS", "600893.SS", "600900.SS", "600905.SS", "601012.SS",
    "601066.SS", "601088.SS", "601166.SS", "601225.SS", "601288.SS",
    "601318.SS", "601390.SS", "601398.SS", "601628.SS", "601633.SS",
    "601668.SS", "601669.SS", "601728.SS", "601857.SS", "601888.SS",
    "601899.SS", "601919.SS", "603259.SS", "603260.SS", "603288.SS",
    "603501.SS", "603799.SS", "603986.SS", "688111.SS", "688599.SS"
]

closing_prices = {} #define closing prices

def fetch_closing_prices(tickers):
    start = datetime(1990, 1, 1)  # Set start date
    end = datetime.now()  # Set end date to current date
    for ticker in tickers:
        data = pdr.get_data_yahoo(ticker, start=start, end=end)
        closing_prices[ticker] = data['Close']


# Fetch and store closing prices for each list of tickers
fetch_closing_prices(NYSE100tickers)
fetch_closing_prices(EURONEXT100tickers)
fetch_closing_prices(TASItickers)
fetch_closing_prices(SSE50tickers)
fetch_closing_prices(NIFTY50tickers)

start = datetime(1990, 1, 1) #started 1990 as latest that any of these FRED 
end = datetime.now()
yf.pdr_override()

usa_gdp = pdr.get_data_fred("FYGDP", start, end) # Usa Gross Domestic Product in billions
usa_cpi = pdr.get_data_fred("CPIAUCSL", start, end) #Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
usa_unemployment = pdr.get_data_fred('UNRATE', start, end) #Usa Unemployment rate
usa_pop = pdr.get_data_fred('CNP16OV', start, end) #Usa Population Level in 1000s
usa_interest = pdr.get_data_fred('REAINTRATREARAT1YE', start, end) #1-Year Real Interest Rate (REAINTRATREARAT1YE)
usa_oil = pdr.get_data_fred("DCOILWTICO", start, end) #West Texas Intermediate (WTI) - Cushing, Oklahoma 
asia_lng = pdr.get_data_fred("PNGASJPUSDM", start, end) #Global price of LNG, Asia
china_gdp = pdr.get_data_fred("MKTGDPCNA646NWDB", start, end) #Gross Domestic Product for China 
china_cpi = pdr.get_data_fred("CPALTT01CNM659N", start, end) #Consumer Price Index: All Items: Total for China
china_unemployment = pdr.get_data_fred("LMUNRRTTCNQ156S", start, end) #Other Labor Market Measures: Registered Unemployment: Rate: Total for China
china_pop = pdr.get_data_fred("POPTOTCNA647NWDB", start, end) #Population, Total for China
china_interest = pdr.get_data_fred("IR3TIB01CNM156N", start, end) #Interest Rates: 3-Month or 90-Day Rates and Yields: Interbank Rates: Total for China 
japan_gdp = pdr.get_data_fred("JPNNGDP", start, end) #Gross Domestic Product for Japan
japan_cpi = pdr.get_data_fred("JPNCPIALLMINMEI", start, end) #Consumer Price Index: All Items: Total for Japan
japan_unemployment = pdr.get_data_fred("LRUN64TTJPM156S", start, end) #Unemployment Rate: Aged 15-64: All Persons for Japan 
japan_pop = pdr.get_data_fred("POPTOTJPA647NWDB", start, end) #Population, Total for Japan
japan_interest = pdr.get_data_fred("IRLTLT01JPM156N", start, end) # Interest Rates: Long-Term Government Bond Yields: 10-Year: Main (Including Benchmark) for Japan
india_gdp = pdr.get_data_fred("MKTGDPINA646NWDB", start, end) #Gross Domestic Product for India
india_cpi = pdr.get_data_fred("INDCPIALLMINMEI", start, end) #Consumer Price Index: All Items: Total for India
india_pop = pdr.get_data_fred("POPTOTINA647NWDB", start, end) #Population, Total for India
india_interest = pdr.get_data_fred("INDIRLTLT01STM", start, end) #Interest Rates: Long-Term Government Bond Yields: 10-Year: Main (Including Benchmark) for India
uk_oil = pdr.get_data_fred("OLPRUKA", start, end) #Oil Prices in the United Kingdom
uk_gdp = pdr.get_data_fred("UKNGDP", start, end) #Gross Domestic Product for United Kingdom
uk_cpi = pdr.get_data_fred("GBRCPIALLMINMEI", start, end) #Consumer Price Index: All Items: Total for United Kingdom
uk_unemployment = pdr.get_data_fred("UNRTUKA", start, end) #Unemployment Rate in the United Kingdom
uk_pop = pdr.get_data_fred("POPUKA", start, end) #Population in the United Kingdom
uk_interest = pdr.get_data_fred("IRLTLT01GBM156N", start, end) #Interest Rates: Long-Term Government Bond Yields: 10-Year: Main (Including Benchmark) for United Kingdom (IRLTLT01GBM156N)	
saudi_gdp = pdr.get_data_fred("MKTGDPSAA646NWDB", start, end) #Gross Domestic Product for Saudi Arabia
saudi_cpi = pdr.get_data_fred("SAUCPALTT01IXOBM", start, end) #Consumer Price Index: All Items: Total for Saudi Arabia
saudi_pop = pdr.get_data_fred("POPTOTSAA647NWDB", start, end) #Population, Total for Saudi Arabia
eu_oil = pdr.get_data_fred("DCOILBRENTEU", start, end) #Crude Oil Prices: Brent - Europe
eu_cpi = pdr.get_data_fred("CP0000EZ19M086NEST", start, end) #Harmonized Index of Consumer Prices: All Items for Euro area (19 countries)
eu_interest = pdr.get_data_fred("BAMLHE00EHYIEY", start, end) #ICE BofA Euro High Yield Index Effective Yield 
eu_unemployment = pdr.get_data_fred("LRHUTTTTEZM156S", start, end) #Harmonised Unemployment - Monthly Rates: Total: All Persons for the Euro Area (19 Countries)
eu_population = pdr.get_data_fred("SPPOPTOTLEUU", start, end) #Population, Total for the European Union
eu_gdp = pdr.get_data_fred("CPMNACSCAB1GQEU272020", start, end) #Gross Domestic Product for European Union (27 Countries from 2020
saudi_oil = pd.read_csv(r"C:\Users\Lewis\Downloads\wti-crude-oil-prices-10-year-daily-chart.csv")
india_unemployment = pd.read_csv(r"C:\Users\Lewis\Downloads\india-unemployment-rate.csv")
saudi_unemployment = pd.read_csv(r"C:\Users\Lewis\Downloads\statistic_id262524_unemployment-rate-in-saudi-arabia-in-2022.csv")
saudi_interest = pd.read_csv(r"C:\Users\Lewis\Downloads\estimated_interest_rates.csv")

#ALGORITHM
def evaluate_economic_indicators(usa_gdp, usa_cpi, usa_unemployment, usa_interest, usa_oil):
    """
    Evaluates economic indicators to categorize the economic environment.
    
    Parameters:
    - usa_gdp: DataFrame of USA Gross Domestic Product.
    - usa_cpi: DataFrame of USA Consumer Price Index.
    - usa_unemployment: DataFrame of USA Unemployment Rate.
    - usa_interest: DataFrame of USA 1-Year Real Interest Rate.
    - usa_oil: DataFrame of USA West Texas Intermediate Oil Prices.
    
    Returns:
    - A dictionary with analysis and categorization of the economic environment.
    """
    
    # Example analysis (You would replace this with actual analysis code)
    # Calculate the latest yearly change for each indicator
    latest_gdp_change = usa_gdp.pct_change(periods=12).iloc[-1]
    latest_cpi_change = usa_cpi.pct_change(periods=12).iloc[-1]
    latest_unemployment_change = usa_unemployment.pct_change(periods=12).iloc[-1]
    latest_interest_change = usa_interest.pct_change(periods=12).iloc[-1]
    latest_oil_change = usa_oil.pct_change(periods=12).iloc[-1]
    
    # Categorize economic environment based on indicator trends
    # This is a simplified example. You would need to use more sophisticated analysis
    # based on economic theories and models.
    if latest_gdp_change > 0 and latest_cpi_change > 0.02 and latest_unemployment_change < 0:
        economic_environment = "expanding"
    elif latest_gdp_change < 0 and latest_unemployment_change > 0:
        economic_environment = "contracting"
    else:
        economic_environment = "stable"
    
    # Compile results
    results = {
        "gdp_change": latest_gdp_change,
        "cpi_change": latest_cpi_change,
        "unemployment_change": latest_unemployment_change,
        "interest_change": latest_interest_change,
        "oil_change": latest_oil_change,
        "economic_environment": economic_environment
    }
    
    return results

def moving_average(data, period):
    return data.rolling(window=period).mean()

def analyze_stock_market(tickers_data):
    """
    Analyzes stock market data for given tickers.
    
    Parameters:
    - tickers_data: Dictionary where key is ticker symbol and value is DataFrame of historical stock data.
    
    Returns:
    - Analysis results as a dictionary with key being ticker symbol and value being analysis summary.
    """
    analysis_results = {}
    
    for ticker, data in tickers_data.items():
        # Ensure data is sorted by date
        data.sort_index(inplace=True)
        
        # Calculate indicators such as moving averages
        data['MA50'] = moving_average(data['Close'], 50)
        data['MA200'] = moving_average(data['Close'], 200)
        
        # Identify potential buy/sell signals, for example, MA50 crossing above MA200 (Golden Cross)
        golden_crosses = (data['MA50'] > data['MA200']) & (data['MA50'].shift(1) <= data['MA200'].shift(1))
        death_crosses = (data['MA50'] < data['MA200']) & (data['MA50'].shift(1) >= data['MA200'].shift(1))
        
        # Summarize findings
        summary = {
            'golden_cross_dates': data[golden_crosses].index.tolist(),
            'death_cross_dates': data[death_crosses].index.tolist(),
            'current_price': data['Close'].iloc[-1],
            'MA50': data['MA50'].iloc[-1],
            'MA200': data['MA200'].iloc[-1]
        }
        
        analysis_results[ticker] = summary
    
    return analysis_results

def generate_signals(economic_environment, stock_analysis_results):
    """
    Generates buy/sell signals based on economic indicators and stock market analysis.

    Parameters:
    - economic_environment: Dictionary with analysis and categorization of the economic environment.
    - stock_analysis_results: Dictionary with stock analysis results, including potential buy/sell signals based on technical indicators.

    Returns:
    - Dictionary with buy/sell signals for stocks.
    """
    signals = {}
    
    for ticker, analysis in stock_analysis_results.items():
        # Initialize signal as 'hold' by default
        signal = 'hold'
        
        # Example of simple decision logic
        if economic_environment['economic_environment'] == 'expanding':
            if 'golden_cross_dates' in analysis and analysis['golden_cross_dates']:
                # Buy signal if there's a recent golden cross in an expanding economy
                signal = 'buy'
        elif economic_environment['economic_environment'] == 'contracting':
            if 'death_cross_dates' in analysis and analysis['death_cross_dates']:
                # Sell signal if there's a recent death cross in a contracting economy
                signal = 'sell'
        
        # Update signals dictionary with decision
        signals[ticker] = signal
    
    return signals


def allocate_portfolio(signals, total_capital=10000000):
    """
    Allocate a portfolio based on buy, sell, and hold signals.

    Parameters:
    - signals: Dictionary with buy/sell/hold signals for each stock.
    - total_capital: Total investment capital available for allocation.

    Returns:
    - A dictionary with allocated capital for each stock.
    """
    # Initialize the allocation dictionary
    allocation = {}

    # Count the number of buy signals to divide the capital equally among them
    buy_signals_count = sum(1 for signal in signals.values() if signal == 'buy')
    
    # Assuming equal allocation for each buy signal
    capital_per_buy_signal = total_capital / buy_signals_count if buy_signals_count else 0
    
    for ticker, signal in signals.items():
        if signal == 'buy':
            # Allocate capital equally among stocks with buy signals
            allocation[ticker] = capital_per_buy_signal
        elif signal == 'sell':
            # For sell signals, you might want to liquidate or not allocate capital
            allocation[ticker] = 0
        elif signal == 'hold':
            # For hold signals, you could choose to maintain existing positions or allocate a minimal amount
            # For simplicity, we're not allocating to 'hold' signals here
            allocation[ticker] = 0

    return allocation


def execute_trades(portfolio, allocations):
    """
    Simulates executing trades based on portfolio allocation in a hypothetical scenario.

    Parameters:
    - portfolio: A list of tuples representing the current portfolio. Each tuple contains (symbol, quantity).
    - allocations: A dictionary with allocated capital for each stock symbol. Here, we'll assume 'quantity' for simplicity.

    Returns:
    - Updated portfolio after executing trades.
    """
    # Convert the current portfolio into a dictionary for easier manipulation
    portfolio_dict = {symbol: quantity for symbol, quantity in portfolio}
    
    for symbol, additional_quantity in allocations.items():
        if symbol in portfolio_dict:
            # If the stock is already in the portfolio, increase the quantity
            portfolio_dict[symbol] += additional_quantity
        else:
            # If the stock is not in the portfolio, add it with the allocated quantity
            portfolio_dict[symbol] = additional_quantity

    # Convert the updated portfolio back into a list of tuples
    updated_portfolio = [(symbol, quantity) for symbol, quantity in portfolio_dict.items()]

    return updated_portfolio

# Example usage
current_portfolio = [("AAPL", 100), ("MSFT", 50)]  # Example starting portfolio
allocations = {"AAPL": 20, "GOOG": 10}  # Simulating buying 20 more AAPL shares and adding 10 GOOG shares

updated_portfolio = execute_trades(current_portfolio, allocations)
print("Updated Portfolio:", updated_portfolio)


def main():
    # First, fetch all the closing prices for the stocks
    fetch_closing_prices(NYSE100tickers)
    fetch_closing_prices(EURONEXT100tickers)
    fetch_closing_prices(TASItickers)
    fetch_closing_prices(SSE50tickers)
    fetch_closing_prices(NIFTY50tickers)
    
    # Evaluate economic indicators
    usa_economic_environment = evaluate_economic_indicators(usa_gdp, usa_cpi, usa_unemployment, usa_interest, usa_oil)
    
    # Analyze the stock market for NYSE
    nyse_analysis_results = analyze_stock_market(closing_prices)
    
    # Generate signals based on economic environment and stock market analysis
    nyse_signals = generate_signals(usa_economic_environment, nyse_analysis_results)
    
    # Allocate portfolio based on signals
    nyse_allocations = allocate_portfolio(nyse_signals)
    
    # Execute trades based on the current portfolio and new allocations
    current_portfolio = [("AAPL", 100), ("MSFT", 50)]  # Example starting portfolio
    updated_portfolio = execute_trades(current_portfolio, nyse_allocations)
    print("Updated Portfolio:", updated_portfolio)

if __name__ == "__main__":
    main()
