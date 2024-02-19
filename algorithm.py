import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import numpy as np
#DATA COLLECTION
#################################
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

#macro economic factors
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

country_macros = {
    "USA": {"GDP": usa_gdp, "CPI": usa_cpi, "Unemployment": usa_unemployment, "Population": usa_pop, "Interest": usa_interest, "Oil": usa_oil},
    "China": {"GDP": china_gdp, "CPI": china_cpi, "Unemployment": china_unemployment, "Population": china_pop, "Interest": china_interest, "Oil": asia_lng},
    "Japan": {"GDP": japan_gdp, "CPI": japan_cpi, "Unemployment": japan_unemployment, "Population": japan_pop, "Interest": japan_interest, "Oil": None},
    "UK": {"GDP": uk_gdp, "CPI": uk_cpi, "Unemployment": uk_unemployment, "Population": uk_pop, "Interest": uk_interest, "Oil": uk_oil},
    "Saudi": {"GDP": saudi_gdp, "CPI": saudi_cpi, "Unemployment": None, "Population": saudi_pop, "Interest": saudi_interest, "Oil": saudi_oil},
    "EU": {"GDP": eu_gdp, "CPI": eu_cpi, "Unemployment": eu_unemployment, "Population": eu_population, "Interest": eu_interest, "Oil": eu_oil},
    "India": {"GDP": india_gdp, "CPI": india_cpi, "Unemployment": None, "Population": india_pop, "Interest": india_interest, "Oil": asia_lng}
}

def macro_trends(country_data): #macro_trends(country_macros["USA"]) - example how to call
    gdp = country_data["GDP"]
    cpi = country_data["CPI"]
    unemployment = country_data["Unemployment"]
    pop = country_data["Population"]
    interest = country_data["Interest"]
    energy = country_data["Oil"]


    
#ALGORITHM - intialises portfoilio
###################################################################
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize yfinance
yf.pdr_override()

# Function to fetch historical data for a list of tickers
def fetch_data(tickers, start_date, end_date):
    stock_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            stock_data[ticker] = hist['Close']
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return pd.DataFrame(stock_data)

# Function to fetch current prices for a list of tickers
def fetch_current_prices(tickers):
    current_prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            current_prices[ticker] = hist['Close'].iloc[0]
        except Exception as e:
            print(f"Error fetching current price for {ticker}: {e}")
    return current_prices

# Function to calculate shares to buy given a budget and current prices
def calculate_shares(budget, prices):
    shares = {ticker: budget / price for ticker, price in prices.items()}
    return shares

# Consolidate all tickers into one list for the analysis
all_tickers = NYSE100tickers + EURONEXT100tickers + TASItickers + SSE50tickers

# Define the investment period for historical analysis
start_date = '2019-01-01'
end_date = '2024-01-01'
historical_data = fetch_data(all_tickers, start_date, end_date)

# Calculate returns and volatility for each stock
returns = historical_data.pct_change().mean() * 252
volatility = historical_data.pct_change().std() * np.sqrt(252)

# Combine returns and volatility into a DataFrame
performance = pd.DataFrame({'Returns': returns, 'Volatility': volatility})

# Select stocks based on a strategy, e.g., highest returns to volatility ratio
selected_stocks = (performance['Returns'] / performance['Volatility']).nlargest(10).index.tolist()

# Define initial budget and fetch current prices for the selected stocks
initial_budget = 10_000_000  # £10 million
current_prices = fetch_current_prices(selected_stocks)
allocated_budget_per_stock = initial_budget / len(selected_stocks)

# Calculate how many shares to buy for each selected stock
shares_to_buy = calculate_shares(allocated_budget_per_stock, current_prices)

# Display the calculated shares to buy
for stock, num_shares in shares_to_buy.items():
    print(f"Buy {num_shares:.2f} shares of {stock}")

#ALGORITHM - that updates portfolio
##################
def evaluate_macro_trend(indicators):
    trends = []
    for key, indicator_data in indicators.items():
        if isinstance(indicator_data, pd.DataFrame) or isinstance(indicator_data, pd.Series):
            if len(indicator_data) >= 2:
                last_value = indicator_data.iloc[-1]
                second_last_value = indicator_data.iloc[-2]
                if isinstance(last_value, pd.Series):
                    last_value = last_value.values[0]
                if isinstance(second_last_value, pd.Series):
                    second_last_value = second_last_value.values[0]
                trend = last_value > second_last_value
                trends.append(trend)
    return trends.count(True) > len(trends) / 2 if trends else False

def decide_action(stock_data, macro_trend):
    stock_trend_up = stock_data.iloc[-1] > stock_data.iloc[-2]  #Compares to last closing price to the price 2 days ago
    if stock_trend_up and macro_trend:
        return "BUY"
    elif not stock_trend_up and not macro_trend:
        return "SELL"
    else:
        return "HOLD"

def analyze_stocks_for_country(country):
    tickers = country_exchange(country)
    if not tickers:
        print(f"No tickers found for {country}.")
        return
    macro_indicators = country_macros.get(country, {})
    macro_trend = evaluate_macro_trend(macro_indicators)

    actions = {}
    for ticker in tickers:
        try:
            stock_data = pdr.get_data_yahoo(ticker, start=datetime.now() - timedelta(days=365), end=datetime.now())['Close']
            action = decide_action(stock_data, macro_trend)
            actions[ticker] = action
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    return actions

def buy_sell_option(country):
    actions = analyze_stocks_for_country(country)
    for ticker, action in actions.items():
        print(f"{ticker}: {action}")
