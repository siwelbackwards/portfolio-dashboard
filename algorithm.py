import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import numpy as np
import time
from pytrends.request import TrendReq
#DATA COLLECTION AND MERGING
#################################
yf.pdr_override()

#Defining period of time
start = '2014-01-01'
end = '2024-01-01'

#List of tickers
NYSE100tickers = [
    "MMM", "ABT", "ACN", "AIG", "MO", "T", "BAC", "BK", "BAX", "BRK-B", "BA", "BMY", 
    "CAT", "C", "KO", "CL", "GLW", "DHR", "DE", "DVN", "D", "DUK", "DD", "LLY", 
    "EMR", "EOG", "EXC", "XOM", "FDX", "F", "BEN", "FCX", "GD", "GE", "GIS", "GM", 
    "GS", "HAL", "HD", "HON", "HPQ", "IBM", "ITW", "JNJ", "JPM", "KMB", "LVS", "LMT", 
    "LOW", "MA", "MCD", "MRK", "MET", "MS", "NKE", "NOV", "OXY", "PEP", "PFE", "PM", 
    "PNC", "PG", "PRU", "RTX", "SLB", "SPG", "SO", "SCCO", "TGT", "TRV", "USB", "UNP", 
    "UPS", "UNH", "VZ", "V", "WBA", "WMT", "DIS", "WFC", "YUM"
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

#Macro economic factors being recived
usa_gdp = pdr.get_data_fred("GDP", start, end) # Usa Gross Domestic Product in billions
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

#Create a new date range that includes all days in the period
daily_index = pd.date_range(start=start, end=end, freq='D')

#Interpolating economic data so in daily format like stocks (currently using linear interpolation but may look to change that)
usa_gdp = usa_gdp.reindex(daily_index).interpolate(method='linear')
usa_cpi = usa_cpi.reindex(daily_index).interpolate(method='linear')
usa_unemployment = usa_unemployment.reindex(daily_index).interpolate(method='linear')
usa_pop = usa_pop.reindex(daily_index).interpolate(method='linear')
usa_interest = usa_interest.reindex(daily_index).interpolate(method='linear')
usa_oil = usa_oil.reindex(daily_index).interpolate(method='linear')
asia_lng = asia_lng.reindex(daily_index).interpolate(method='linear')
china_gdp = china_gdp.reindex(daily_index).interpolate(method='linear')
china_cpi = china_cpi.reindex(daily_index).interpolate(method='linear')
china_unemployment = china_unemployment.reindex(daily_index).interpolate(method='linear')
china_pop = china_pop.reindex(daily_index).interpolate(method='linear')
china_interest = china_interest.reindex(daily_index).interpolate(method='linear')
japan_gdp = japan_gdp.reindex(daily_index).interpolate(method='linear')
japan_cpi = japan_cpi.reindex(daily_index).interpolate(method='linear')
japan_unemployment = japan_unemployment.reindex(daily_index).interpolate(method='linear')
japan_pop = japan_pop.reindex(daily_index).interpolate(method='linear')
japan_interest = japan_interest.reindex(daily_index).interpolate(method='linear')
india_gdp = india_gdp.reindex(daily_index).interpolate(method='linear')
india_cpi = india_cpi.reindex(daily_index).interpolate(method='linear')
india_pop = india_pop.reindex(daily_index).interpolate(method='linear')
india_interest = india_interest.reindex(daily_index).interpolate(method='linear')
uk_oil = uk_oil.reindex(daily_index).interpolate(method='linear')
uk_gdp = uk_gdp.reindex(daily_index).interpolate(method='linear')
uk_cpi = uk_cpi.reindex(daily_index).interpolate(method='linear')
uk_unemployment = uk_unemployment.reindex(daily_index).interpolate(method='linear')
uk_pop = uk_pop.reindex(daily_index).interpolate(method='linear')
uk_interest = uk_interest.reindex(daily_index).interpolate(method='linear')
saudi_gdp = saudi_gdp.reindex(daily_index).interpolate(method='linear')
saudi_cpi = saudi_cpi.reindex(daily_index).interpolate(method='linear')
saudi_pop = saudi_pop.reindex(daily_index).interpolate(method='linear')
eu_oil = eu_oil.reindex(daily_index).interpolate(method='linear')
eu_cpi = eu_cpi.reindex(daily_index).interpolate(method='linear')
eu_interest = eu_interest.reindex(daily_index).interpolate(method='linear')
eu_unemployment = eu_unemployment.reindex(daily_index).interpolate(method='linear')
eu_population = eu_population.reindex(daily_index).interpolate(method='linear')
eu_gdp = eu_gdp.reindex(daily_index).interpolate(method='linear')

#country and corresponding stock exchange (need to add the other stock exchange that the model will be trained on)
def country_exchange(country):
    if country == "USA":
        return NYSE100tickers
    elif country == "China":
        return SSE50tickers
    elif country == "Japan":
        return None 
    elif country == "UK":
        return None 
    elif country == "Saudi":
        return TASItickers 
    elif country == "EU":
        return EURONEXT100tickers
    elif country == "India":
        return None
    else:
        return None 

#country economic factors
country_macros = {
    "USA": {"GDP": usa_gdp, "CPI": usa_cpi, "Unemployment": usa_unemployment, "Population": usa_pop, "Interest": usa_interest, "Oil": usa_oil},
    "China": {"GDP": china_gdp, "CPI": china_cpi, "Unemployment": china_unemployment, "Population": china_pop, "Interest": china_interest, "Oil": asia_lng},
    "Japan": {"GDP": japan_gdp, "CPI": japan_cpi, "Unemployment": japan_unemployment, "Population": japan_pop, "Interest": japan_interest, "Oil": asia_lng},
    "UK": {"GDP": uk_gdp, "CPI": uk_cpi, "Unemployment": uk_unemployment, "Population": uk_pop, "Interest": uk_interest, "Oil": uk_oil},
    "Saudi": {"GDP": saudi_gdp, "CPI": saudi_cpi, "Unemployment": saudi_unemployment, "Population": saudi_pop, "Interest": saudi_interest, "Oil": saudi_oil},
    "EU": {"GDP": eu_gdp, "CPI": eu_cpi, "Unemployment": eu_unemployment, "Population": eu_population, "Interest": eu_interest, "Oil": eu_oil},
    "India": {"GDP": india_gdp, "CPI": india_cpi, "Unemployment": india_unemployment, "Population": india_pop, "Interest": india_interest, "Oil": asia_lng}
}

def macro_trends(country_data): #macro_trends(country_macros["USA"]) - example how to call
    gdp = country_data["GDP"]
    cpi = country_data["CPI"]
    unemployment = country_data["Unemployment"]
    pop = country_data["Population"]
    interest = country_data["Interest"]
    energy = country_data["Oil"]

#Setting Google trends data for US and time period ect...
pytrends = TrendReq(hl='en-US', tz=360)
timeframe = '2014-01-01 2024-01-01'
geo = 'US'

# Dictionary to store Google Trends data for each ticker
trends_data = {}

# For loop to gather all Google Trends data
for ticker in NYSE100tickers:
    pytrends.build_payload([ticker], timeframe=timeframe, geo=geo)
    # Fetch the interest over time and remove 'isPartial' column
    df = pytrends.interest_over_time().drop(columns=['isPartial'], errors='ignore')
    # Store the DataFrame in the dictionary
    trends_data[ticker] = df
    # Sleep to avoid hitting rate limits when fetching data from google
    time.sleep(1)

# Function to retrieve individual stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']  # Gathers closing

# Initialize the dictionary again for clarity
stock_data_frames = {}

# Adding sector data
ticker_to_sector = {}  # Dictionary to hold mapping of tickers to sectors

for ticker in NYSE100tickers:
    stock_info = yf.Ticker(ticker)
    sector = stock_info.info['sector']
    ticker_to_sector[ticker] = sector



# Algorithm to fetch stock data and merge with all economic indicators for all tickers
for tick in NYSE100tickers:
    # Fetch stock data
    stock_data = get_stock_data(tick, start, end)
    # Merge with Economic factors
    merged_data = pd.merge(stock_data.to_frame(name='Close'), usa_gdp, left_index=True, right_index=True, how='left')
    merged_data = pd.merge(merged_data, usa_cpi, left_index=True, right_index=True, how='left', suffixes=('', '_cpi'))
    merged_data = pd.merge(merged_data, usa_unemployment, left_index=True, right_index=True, how='left', suffixes=('', '_unemployment'))
    merged_data = pd.merge(merged_data, usa_pop, left_index=True, right_index=True, how='left', suffixes=('', '_pop'))
    merged_data = pd.merge(merged_data, usa_interest, left_index=True, right_index=True, how='left', suffixes=('', '_interest'))
    merged_data = pd.merge(merged_data, usa_oil, left_index=True, right_index=True, how='left', suffixes=('', '_oil'))
    #merged_data = pd.merge(merged_data, sector, left_index=True, right_index=True, how='left', suffixes=('', '_sector'))
    
    # Merge with Google Trends data
    if tick in trends_data:
        google_trends_data = trends_data[tick]
        # Ensure the Google Trends data index is a datetime index
        google_trends_data.index = pd.to_datetime(google_trends_data.index)
        
        # Merge Google Trends data (google data currently not working could be issue from hitting rate limits when fetching data from google)
        merged_data = pd.merge(merged_data, google_trends_data, left_index=True, right_index=True, how='left', suffixes=('', f'_{tick}_trend'))
    
    # Store the merged DataFrame
    stock_data_frames[tick] = merged_data

    
# Merging with stock data
for tick in NYSE100tickers:
    if tick in stock_data_frames:
        stock_data_frames[tick]['Sector'] = ticker_to_sector[tick]


# Define the list of industries
industries = ['Industrials', 'Healthcare', 'Technology', 'Financial Services',
              'Consumer Defensive', 'Communication Services', 'Energy', 'Utilities',
              'Basic Materials', 'Consumer Cyclical', 'Real Estate']

# Create a NumPy array from the covariance matrix
V = np.array([[ 6.132e+03,  3.080e+02,  2.600e+01, -1.250e+02,  1.270e+02,
        -1.150e+02,  4.130e+02,  4.000e+01, -1.460e+02, -4.200e+01,
        -5.180e+02],
       [ 3.080e+02,  5.845e+03,  3.000e+00,  2.560e+02, -5.500e+01,
         3.330e+02, -2.770e+02, -2.440e+02,  2.850e+02, -3.340e+02,
        -7.080e+02],
       [ 2.600e+01,  3.000e+00,  5.758e+03,  3.900e+01, -1.830e+02,
        -2.300e+02, -2.540e+02,  8.000e+01,  1.840e+02,  4.070e+02,
         1.890e+02],
       [-1.250e+02,  2.560e+02,  3.900e+01,  6.278e+03, -4.490e+02,
         5.200e+01, -4.980e+02, -4.790e+02, -2.660e+02, -2.430e+02,
         4.100e+01],
       [ 1.270e+02, -5.500e+01, -1.830e+02, -4.490e+02,  5.877e+03,
        -2.560e+02,  3.960e+02, -1.100e+01,  3.100e+01, -3.180e+02,
         8.500e+01],
       [-1.150e+02,  3.330e+02, -2.300e+02,  5.200e+01, -2.560e+02,
         5.920e+03, -2.330e+02, -1.540e+02,  6.900e+01,  5.600e+01,
        -5.090e+02],
       [ 4.130e+02, -2.770e+02, -2.540e+02, -4.980e+02,  3.960e+02,
        -2.330e+02,  6.316e+03,  7.450e+02, -1.060e+02, -2.980e+02,
        -2.110e+02],
       [ 4.000e+01, -2.440e+02,  8.000e+01, -4.790e+02, -1.100e+01,
        -1.540e+02,  7.450e+02,  5.853e+03, -1.190e+02,  7.000e+00,
        -5.300e+01],
       [-1.460e+02,  2.850e+02,  1.840e+02, -2.660e+02,  3.100e+01,
         6.900e+01, -1.060e+02, -1.190e+02,  5.717e+03, -4.400e+01,
        -1.680e+02],
       [-4.200e+01, -3.340e+02,  4.070e+02, -2.430e+02, -3.180e+02,
         5.600e+01, -2.980e+02,  7.000e+00, -4.400e+01,  5.981e+03,
         3.070e+02],
       [-5.180e+02, -7.080e+02,  1.890e+02,  4.100e+01,  8.500e+01,
        -5.090e+02, -2.110e+02, -5.300e+01, -1.680e+02,  3.070e+02,
         6.035e+03]])
inv_V=np.linalg.inv(V)

#edit r CHANGE LATER DATE
random_numbers = np.random.rand(11, 1) * 0.1
r = random_numbers + 1

# Assuming inv_V is defined elsewhere and is an 11x11 matrix
alpha = np.dot(np.ones((1,11)), np.dot(inv_V, np.ones((11, 1))))
beta = np.dot(np.ones((1, 11)), np.dot(inv_V, r))
gamma = np.dot(r.T, np.dot(inv_V, r))
delta = alpha*beta - gamma**2
mu=1.05 #expected return of portfoilio

# Calculate lambda values
lambda_1 = (gamma - beta * mu) / delta
lambda_2 = (alpha * mu - beta) / delta

# Calculate the efficient portfolio
efficient_portfolio = lambda_1 * np.dot(inv_V, np.ones((11, 1))) + lambda_2 * np.dot(inv_V, r)

# Normalize the weights to ensure they sum up to 1
efficient_portfolio_weights = efficient_portfolio / np.sum(efficient_portfolio)

#TIMESERIES ALGORITHM MONTE CARLO
####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_simulations = 1000
num_days = 252

# Placeholder for simulation results
simulation_results = {}

# Iterate over each stock in your DataFrame
for ticker in stock_data_frames.keys():
    # Extract the 'Close' prices for the current ticker
    close_prices = stock_data_frames[ticker]['Close']
    
    # Calculate necessary statistics
    last_price = close_prices[-1]
    returns = close_prices.pct_change()
    mean_daily_return = returns.mean()
    variance_daily_return = returns.var()
    drift = mean_daily_return - (0.5 * variance_daily_return)
    volatility = returns.std()
    
    # Prepare a list to collect all simulations
    all_simulations = []
    
    # Run simulations
    for x in range(num_simulations):
        price_series = [last_price]
        
        for y in range(num_days):
            price = price_series[-1] * np.exp(drift + volatility * np.random.normal())
            price_series.append(price)
        
        # Add the completed simulation to the list
        all_simulations.append(price_series)

    # Convert the list of simulations into a DataFrame all at once
    simulation_df = pd.DataFrame(all_simulations).transpose()

    # Store the simulation results
    simulation_results[ticker] = simulation_df

# Placeholder for expected returns
expected_returns = {}

# Analyze simulation results and calculate expected returns
for ticker, simulation_df in simulation_results.items():
    # Calculate the mean of the final day's prices across all simulations
    expected_future_price = simulation_df.iloc[-1].mean()
    last_price = stock_data_frames[ticker]['Close'].iloc[-1]
    expected_return = (expected_future_price - last_price) / last_price
    expected_returns[ticker] = expected_return

# Select the best stock per industry based on expected returns
best_stocks_per_industry = {}
for industry in industries:
    industry_stocks = {ticker: expected_returns[ticker] for ticker, sector in ticker_to_sector.items() if sector == industry}
    if industry_stocks:  # Check if there are stocks in this industry
        best_stock = max(industry_stocks, key=industry_stocks.get)
        best_stocks_per_industry[industry] = best_stock

print(best_stocks_per_industry)
