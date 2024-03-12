from dash import Dash, html, dcc, Input, Output, State, dash_table
import numpy.linalg as linalg
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

#Function to model last 6 months porfolio performance
#######################################################
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data for a list of tickers."""
    stock_data = {}
    for ticker in tickers:
        stock = yf.download(ticker, start=start_date, end=end_date)
        if not stock.empty:
            # Forward fill to handle weekends and holidays, then interpolate remaining missing values
            stock = stock['Adj Close'].fillna(method='ffill').interpolate(method='linear')
            stock_data[ticker] = stock
    combined_data = pd.DataFrame(stock_data)
    # Ensure that the DataFrame is filled for weekends/holidays
    combined_data = combined_data.asfreq('B').fillna(method='ffill').interpolate(method='linear')
    return combined_data

def calculate_portfolio_value(stock_data, allocations):
    """Calculate daily portfolio value based on stock data and allocations,
    taking into account short positions."""
    # Ensure alignment of indices and columns for multiplication
    aligned_data, aligned_allocations = stock_data.align(allocations, join='right', axis=1)
    aligned_data_filled = aligned_data.fillna(method='ffill').fillna(0)  # Forward fill to handle missing values
    daily_returns = aligned_data_filled.pct_change().fillna(0)

    # Adjust returns for short positions
    # For short positions (negative weights), a positive return means a decrease in portfolio value and vice versa
    adjusted_returns = daily_returns.multiply(aligned_allocations, axis='columns')

    portfolio_returns = adjusted_returns.sum(axis=1)
    portfolio_value = (1 + portfolio_returns).cumprod() * 100  # assuming initial value of 100
    return portfolio_value

# Define the 6-month period
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

# DATA COLLECTION AND MERGING
#################################
yf.pdr_override()

# Defining period of time
start = '2004-01-01'
end = '2024-01-01'

FTSE350tickers = [
    "AAL.L", "ABDN.L", "ABF.L", "ADM.L", "AGR.L", "AGT.L", "AHT.L", "AJB.L", "AML.L", "ANTO.L",
    "AO.L", "APAX.L", "APEO.L", "ASCL.L", "ASHM.L", "ASL.L", "ATG.L", "AUTO.L",
    "AV.L", "AZN.L", "BA.L", "BAB.L", "BAG.L", "BAKK.L", "BARC.L", "BATS.L", "BBGI.L", "BBH.L",
    "BBOX.L", "BBY.L", "BCG.L", "BCPT.L", "BDEV.L", "BEZ.L", "BGEO.L", "BGFD.L", "BHMG.L", "BKG.L",
    "BLND.L", "BME.L", "BNKR.L", "BOWL.L", "BP.L", "BRBY.L", "BREE.L", "BRK.L", "BSE.L",
    "BVIC.L", "BWNG.L", "BWY.L", "BYG.L", "CBG.L", "CCH.L", "CEY.L",
    "CLDN.L", "CLI.L", "CLIG.L", "CMCX.L", "CNA.L", "CNE.L", "COA.L", "COST.L", "CPG.L", "CPP.L",
    "CRDA.L", "CRH.L", "CRST.L", "CSN.L", "CVSG.L", "CWK.L", "DCC.L", "DGE.L", "DIG.L", "DLG.L",
    "DNLM.L", "DSCV.L", "EDIN.L", "EMG.L", "ENOG.L", "ENQ.L", "EPIC.L", "ESNT.L", "EVR.L",
    "EXPN.L", "EZJ.L", "FCIT.L", "FDM.L", "FERG.L", "FRES.L", "FRP.L", "FUTR.L", "FXPO.L", "GAW.L",
    "GCP.L", "GFRD.L", "GKN.L", "GLEN.L", "GPE.L", "GRG.L", "GRI.L", "GSK.L",
    "HAS.L", "HBR.L", "HICL.L", "HIK.L", "HL.L", "HLCL.L", "HLMA.L", "HMSO.L", "HNE.L", "HOME.L",
    "HSBA.L", "HSL.L", "HSX.L", "HTWS.L", "HWDN.L", "IAG.L", "ICP.L", "ICGT.L", "IGG.L",
    "IHG.L", "III.L", "IMB.L", "INF.L", "INPP.L", "INVP.L", "IPO.L", "JII.L",
    "JMAT.L", "JMG.L", "JUP.L", "KGF.L", "KIE.L", "KNOS.L", "KWS.L", "LAND.L", "LGEN.L",
    "LLOY.L", "LMP.L", "LONR.L", "LRE.L", "LSEG.L", "LTG.L", "LWDB.L", "LXI.L", "MAB.L",
    "MARS.L", "MCB.L", "MCLS.L", "MGNS.L", "MKS.L", "MNG.L", "MNKS.L", "MONY.L",
    "MRO.L", "MTO.L", "MUT.L", "NCC.L", "NG.L", "OCDO.L", "OML.L", "PAY.L", "PCT.L", "PCTN.L", "PDG.L", "PETS.L",
    "PEY.L",
    "PHNX.L", "PIN.L", "PRU.L", "PSN.L", "PSON.L", "QQ.L", "RAT.L", "RCP.L", "RDW.L", "REL.L",
    "RIO.L", "RMV.L", "RNK.L", "ROR.L", "RR.L", "RRS.L", "RTO.L", "SBRY.L", "SCT.L", "SDP.L",
    "SDR.L", "SEPL.L", "SGE.L", "SGRO.L", "SHI.L", "SHP.L", "SIV.L", "SKG.L", "SKY.L",
    "SMDS.L", "SMIN.L", "SMT.L", "SMWH.L", "SN.L", "SNR.L", "SOLG.L", "SPX.L", "SRP.L",
    "SSE.L", "SST.L", "STAN.L", "STJ.L", "SVS.L", "SVT.L", "SYNT.L", "TATE.L", "TCAP.L", "TEP.L",
    "THG.L", "THRL.L", "TIFS.L", "TLW.L", "TPK.L", "TRIG.L", "TRN.L", "TSCO.L", "TW.L", "UBM.L",
    "UKCM.L", "UKW.L", "ULVR.L", "UTG.L", "UU.L", "VCT.L", "VED.L", "VOD.L", "VP.L", "WEIR.L",
    "WG.L", "WIZZ.L", "WPP.L", "WTB.L", "WWH.L"
]

NYSE100tickers = [
    "MMM", "ABT", "ACN", "AIG", "MO", "T", "BAC", "BK", "BAX", "BRK-B", "BA", "BMY",
    "CAT", "C", "KO", "CL", "GLW", "DHR", "DE", "DVN", "D", "DUK", "DD", "LLY",
    "EMR", "EOG", "EXC", "XOM", "FDX", "F", "BEN", "FCX", "GD", "GE", "GIS", "GM",
    "GS", "HAL", "HD", "HON", "HPQ", "IBM", "ITW", "JNJ", "JPM", "KMB", "LVS", "LMT",
    "LOW", "MA", "MCD", "MRK", "MET", "MS", "NKE", "NOV", "OXY", "PEP", "PFE", "PM",
    "PNC", "PG", "PRU", "RTX", "SLB", "SPG", "SO", "SCCO", "TGT", "TRV", "USB", "UNP",
    "UPS", "UNH", "VZ", "V", "WBA", "WMT", "DIS", "WFC", "YUM"
]

#Industry allocation
#############################
#loading preprocessed data of server setup on creating covariance matrix
new_data = pd.read_excel("https://github.com/siwelbackwards/portfolio-dashboard/raw/main/sector_averages_.xlsx")
new_covariance_matrix = new_data.cov()
for sector in new_data.columns[1:]:
    new_covariance_matrix.loc[sector, sector] = new_data[sector].var()
V = new_covariance_matrix

industries = ['Basic Materials', 'Communication Services', 'Consumer Cyclical', 'Consumer Defensive', 'Energy', 'Financial Services', 'Health Care', 'Industrials', 'Real Estate', 'Technology', 'Utilities']
V_inv = linalg.inv(V)
one_vector = np.ones(11)
mu=1.03
r_vector = np.array(new_data.mean(axis='rows'))
alpha = one_vector @ V_inv @ one_vector.T

beta = one_vector @ V_inv @ r_vector.T

gamma = r_vector @ V_inv @ r_vector.T

delta = alpha * gamma - beta ** 2

lambda_1 = (gamma - beta * mu) / delta
lambda_2 = (alpha * mu - beta) / delta

efficient_portfolio_weights = lambda_1 * one_vector @ V_inv + lambda_2 * r_vector @ V_inv
allocations = dict(zip(industries, efficient_portfolio_weights.flatten()))

allocations_df = pd.DataFrame(list(allocations.items()), columns=['Industry', 'Weight'])
allocations_df['Weight'] = allocations_df['Weight'] / allocations_df['Weight'].sum()
allocations_df['Allocation'] = allocations_df['Weight'].apply(lambda x: f"{x*100:+.2f}%")

#Returns past 6 Months
#######################################
def get_stock_list(dataframe, title, industry_stock_mapping):
    # Filter the DataFrame based on the title
    if title == 'Long Allocations':
        filtered_df = dataframe[dataframe['Weight'] > 0]
    else:  # For short allocations
        filtered_df = dataframe[dataframe['Weight'] < 0].copy()
        filtered_df['Weight'] = filtered_df['Weight'].abs()

    # Map industries to stocks
    filtered_df = filtered_df.copy()
    filtered_df['Stock'] = filtered_df['Industry'].map(industry_stock_mapping)

    return filtered_df['Stock'].tolist()

def map_industries_to_stocks(allocations_df, industry_stock_mapping):
    # Create an empty DataFrame to store the mapping results
    stock_allocations_df = pd.DataFrame(columns=['Stock', 'Weight'])

    # Loop through each industry in the allocations DataFrame
    for index, row in allocations_df.iterrows():
        industry = row['Industry']
        weight = row['Weight']

        # Get the stock list for the current industry
        if industry in industry_stock_mapping:
            stocks = industry_stock_mapping[industry]

            # If the industry maps to multiple stocks, distribute the weight evenly across the stocks
            if isinstance(stocks, list):
                number_of_stocks = len(stocks)
                weights = [weight / number_of_stocks] * number_of_stocks
                temp_df = pd.DataFrame({
                    'Stock': stocks,
                    'Weight': weights
                })
            else:
                temp_df = pd.DataFrame({
                    'Stock': [stocks],
                    'Weight': [weight]
                })

            # Append the results to the stock_allocations_df DataFrame
            stock_allocations_df = pd.concat([stock_allocations_df, temp_df], ignore_index=True)

    return stock_allocations_df

def calculate_six_month_returns(stock_list):
    # Definung the start and end dates for the 6-month period
    end_date = datetime.now()
    start_date = end_date - timedelta(6 * 30)  # Approximate 6 months back

    returns = {}

    # Downloading stock data and calculate returns
    for stock in stock_list:
        data = yf.download(stock, start=start_date, end=end_date)
        if not data.empty:
            # Calculate the return: (end_price - start_price) / start_price
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            total_return = (end_price - start_price) / start_price
            returns[stock] = total_return * 100  # Convert to percentage
        else:
            returns[stock] = None  # No data for this stock

    return returns
#Pie chart allocations FTSE and NYSE
#Taking data that uses the alorgithm of the economic state to laod and process and then generate bests tocks per industry
#############################
industry_to_stock = pd.read_excel('https://github.com/siwelbackwards/portfolio-dashboard/raw/main/ftse%20stocks%20industies.xlsx')
industry_stock_mapping = industry_to_stock.iloc[0].to_dict()

industry_to_stockNYSE = pd.read_excel('https://github.com/siwelbackwards/portfolio-dashboard/raw/main/nyse%20stocks%20industies.xlsx')
industry_stock_mappingNYSE = industry_to_stockNYSE.iloc[0].to_dict()

def get_stock_name(ticker):
    try:
        # Fetching the stock info
        stock_info = yf.Ticker(ticker).info
        # Return the long name of the stock
        return stock_info.get('longName', ticker)
    except Exception as e:
        # In case of an error, return the ticker
        return ticker

def generate_pie_chart(dataframe, title, industry_stock_mapping):
    # Separating positive and negative allocations if necessarys
    if title == 'Long Allocations':
        filtered_df = dataframe[dataframe['Weight'] > 0]
        colors = ['#4CAF50', '#FFEB3B', '#03A9F4', '#E91E63', '#FF9800']
    else:  # For short allocations
        filtered_df = dataframe[dataframe['Weight'] < 0].copy()
        filtered_df['Weight'] = filtered_df['Weight'].abs()
        colors = ['#051C2A', '#163E93', '#30A3DA', '#dbb880', '#483d8b', '#ffff9f', '#ff77ff', '#4b5320', '#8f5c14']

    # Map industries to stocks
    filtered_df['Stock'] = filtered_df['Industry'].map(industry_stock_mapping)

    # Replaces tickers with full stock names
    filtered_df['Full Name'] = filtered_df['Stock'].apply(get_stock_name)

    # Create the pie chart with full stock names
    fig = go.Figure(data=[go.Pie(
        labels=filtered_df['Full Name'],
        values=filtered_df['Weight'],
        hole=.3,
        domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]},
        marker=dict(
            colors=colors,  # Apply the selected color palette
            line=dict(color='#FFFFFF', width=2)
        ),
        hoverinfo='label+percent',
        textinfo='none'
    )])

    # Adjusting figure layout for theme
    fig.update_layout({
        'plot_bgcolor': '#ffffff',  # Transparent background
        'paper_bgcolor': '#ffffff',  # Transparent around the chart
        'showlegend': True,
        'legend': {
            'font': {'color': '#000000', 'family': 'Arial, sans-serif', 'size': 16},
            'orientation': 'h',  # Horizontal orientation for legend
            'yanchor': 'bottom',
            'y': -0.2,
            'xanchor': 'center',
            'x': 0.5
        },
        'title': {
            'text': title,
            'font': {'size': 40, 'color': '#000000', 'family': 'Arial, sans-serif'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        'font': {
            'color': '#000000',  # Black for all other text in the chart
            'family': 'Arial, sans-serif',
            'size': 16
        },
        'margin': dict(t=80, b=80, l=0, r=0),
        'height': 700
    })

    return fig

#FTSE AND NYSE ALLOCATION
#Excel sheet recalled of the allocation - updated using an external python server that runs and does calculations every reshuffle date, and uses intergation to update github excel file
countryallocation = pd.read_excel('https://github.com/siwelbackwards/portfolio-dashboard/raw/main/Country%20allocation.xlsx')
allocation_nyse = countryallocation["NYSE"].iloc[0]
allocation_ftse = countryallocation["FTSE"].iloc[0]

#ECONOMIC STATE CALCULATION
#Excel sheet recalled of the economic state - updated using an external python server that runs and does calculations every reshuffle date, and uses intergation to update github excel file
state = pd.read_excel('https://github.com/siwelbackwards/portfolio-dashboard/raw/main/Economic_state.xlsx')
economic_phase = state["Economic state"].iloc[0]

#EXPECTED RETURNS CALCUALTION
###############################
# Creating a Dash app
def update_output(n_clicks, name, email, message):
    if n_clicks == 0:
        return ""
    else:
        # Logic for handling the submission could go here
        return True, "Thank you for your submission. We will get back to you shortly."

# Calculate the next target date
def calculate_next_target_date():
    now = datetime.now()
    current_year = now.year
    target_month = 3 if now.month < 9 else 9  # March or September
    target_date = datetime(year=current_year, month=target_month, day=1)
    if target_date < now:
        if target_month == 9:
            target_date = datetime(year=current_year + 1, month=3, day=1)
        else:
            target_date = datetime(year=current_year, month=9, day=1)
    return target_date

# Get the list of NYSE stocks for both long and short allocations
long_nyse_stocks = get_stock_list(allocations_df, 'Long Allocations', industry_stock_mappingNYSE)
short_nyse_stocks = get_stock_list(allocations_df, 'Short Allocations', industry_stock_mappingNYSE)

# Calculate the 6-month returns for both long and short allocations
long_six_month_returns_nyse = calculate_six_month_returns(long_nyse_stocks)
short_six_month_returns_nyse = {k: -v for k, v in calculate_six_month_returns(short_nyse_stocks).items()}

# Combine the returns and create a DataFrame to store them
combined_returns_nyse = {**long_six_month_returns_nyse, **short_six_month_returns_nyse}
returns_df_nyse = pd.DataFrame(list(combined_returns_nyse.items()), columns=['Ticker', '6 Month Return (%)'])
stock_allocations_df_nyse = map_industries_to_stocks(allocations_df, industry_stock_mappingNYSE)
# Merge with allocations to get the weights
returns_df_nyse = returns_df_nyse.merge(stock_allocations_df_nyse, left_on='Ticker', right_on='Stock')

# Calculate the weighted returns and sum them to get the total portfolio return for NYSE
returns_df_nyse['Weighted Return'] = returns_df_nyse['6 Month Return (%)'] * returns_df_nyse['Weight']
total_portfolio_return_nyse = returns_df_nyse['Weighted Return'].sum()

long_ftse_stocks = get_stock_list(allocations_df, 'Long Allocations', industry_stock_mapping)
short_ftse_stocks = get_stock_list(allocations_df, 'Short Allocations', industry_stock_mapping)

# Calculate the 6-month returns for both long and short allocations
long_six_month_returns = calculate_six_month_returns(long_ftse_stocks)
short_six_month_returns = {k: -v for k, v in calculate_six_month_returns(short_ftse_stocks).items()}

# Combine the returns and create a DataFrame to store them
combined_returns = {**long_six_month_returns, **short_six_month_returns}
returns_df = pd.DataFrame(list(combined_returns.items()), columns=['Ticker', '6 Month Return (%)'])
stock_allocations_df = map_industries_to_stocks(allocations_df, industry_stock_mapping)
# Merge with allocations to get the weights
returns_df = returns_df.merge(stock_allocations_df, left_on='Ticker', right_on='Stock')

# Calculate the weighted returns and sum them to get the total portfolio return
returns_df['Weighted Return'] = returns_df['6 Month Return (%)'] * returns_df['Weight']
total_portfolio_return = returns_df['Weighted Return'].sum()

#Past 6 months grpah home page calculations
####################################################################
nyse_tickers = stock_allocations_df_nyse['Stock'].tolist()
ftse_tickers = stock_allocations_df['Stock'].tolist()
nyse_stock_data = fetch_stock_data(nyse_tickers, start_date, end_date)
ftse_stock_data = fetch_stock_data(ftse_tickers, start_date, end_date)

# Calculate portfolio values
nyse_portfolio_value = calculate_portfolio_value(nyse_stock_data, stock_allocations_df_nyse.set_index('Stock')['Weight'])
ftse_portfolio_value = calculate_portfolio_value(ftse_stock_data, stock_allocations_df.set_index('Stock')['Weight'])

# Combine NYSE and FTSE portfolio values for a complete portfolio
combined_portfolio_value = nyse_portfolio_value*allocation_nyse + ftse_portfolio_value*allocation_ftse

def generate_future_dates(start_date, num_days):
    # Generate a date range from start_date for num_days, excluding weekends
    future_dates = pd.bdate_range(start=start_date, periods=num_days).tolist()
    return future_dates
def simulate_stock_prices(ticker, last_price, num_simulations=2500, num_days=126):
    data = yf.download(ticker, period='60mo')
    close_prices = data['Adj Close']
    returns = close_prices.pct_change()
    mean_daily_return = returns.mean()
    variance_daily_return = returns.var()
    drift = mean_daily_return - (0.5 * variance_daily_return)
    volatility = returns.std()

    simulations = []
    for _ in range(num_simulations):
        prices = [last_price]
        for _ in range(num_days):
            shock = drift + volatility * np.random.normal()
            price = prices[-1] * np.exp(shock)
            prices.append(price)
        simulations.append(prices)
    return np.mean(simulations, axis=0)

def simulate_portfolio_returns(stock_allocations_df, num_simulations=100, num_days=126):
    portfolio_simulations = []
    for index, row in stock_allocations_df.iterrows():
        ticker = row['Stock']
        weight = row['Weight']
        last_price = yf.download(ticker, period='1d')['Adj Close'][-1]
        simulated_prices = simulate_stock_prices(ticker, last_price, num_simulations, num_days)

        # Calculate daily returns from simulated prices
        daily_returns = np.diff(simulated_prices) / simulated_prices[:-1] * 100
        weighted_daily_returns = daily_returns * weight

        portfolio_simulations.append(weighted_daily_returns)

    # Aggregate daily returns across the portfolio
    total_portfolio_simulation = np.sum(portfolio_simulations, axis=0)
    return total_portfolio_simulation

# Simulate NYSE Portfolio
nyse_simulation = simulate_portfolio_returns(stock_allocations_df_nyse)

# Simulate FTSE Portfolio
ftse_simulation = simulate_portfolio_returns(stock_allocations_df)

def plot_portfolio_projections(simulation):
    plt.figure(figsize=(10, 6))
    plt.plot(simulation)
    plt.title("Portfolio Performance Projection")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.show()

# Plot NYSE Portfolio Projection
plot_portfolio_projections(nyse_simulation)

# Plot FTSE Portfolio Projection
plot_portfolio_projections(ftse_simulation)
def plot_portfolio_performance():
    # Create the figure
    fig = go.Figure()

    # Add trace for the combined portfolio with a specified color
    fig.add_trace(go.Scatter(
        x=combined_portfolio_value.index,
        y=combined_portfolio_value,
        mode='lines',
        name='Combined Portfolio Value',
        line=dict(color='#6279ec'),  # Specified hex color code
    ))

    # Add trace for the NYSE portfolio with a specified color
    fig.add_trace(go.Scatter(
        x=nyse_portfolio_value.index,
        y=nyse_portfolio_value,
        mode='lines',
        name='NYSE Portfolio Value',
        line=dict(color='#00a8f3'),  # Specified hex color code
    ))

    # Add trace for the FTSE portfolio with a specified color
    fig.add_trace(go.Scatter(
        x=ftse_portfolio_value.index,
        y=ftse_portfolio_value,
        mode='lines',
        name='FTSE Portfolio Value',
        line=dict(color='#051c2c'),  # Specified hex color code
    ))

    # Update layout to match the provided image style
    fig.update_layout(
        title='Past 6 Month Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (%)',
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        xaxis=dict(
            showline=True,
            showgrid=False,
            linecolor='black',  # Line color for x-axis
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            gridcolor='lightgrey',  # Grid color for y-axis
            linecolor='black',  # Line color for y-axis
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=0, b=0)  # Remove margins
    )

    return fig

def update_countdown(n):
    target_date = calculate_next_target_date()
    now = datetime.now()
    time_left = target_date - now

    days, remainder = divmod(time_left.total_seconds(), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    if time_left.total_seconds() > 0:
        return f"Time until portfolio reshuffle: {int(days)} days, {int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    else:
        return "Time for portfolio reshuffle!"

# Precompute NYSE and FTSE cumulative returns at startup
nyse_daily_returns = simulate_portfolio_returns(stock_allocations_df_nyse, 2500, 126)
ftse_daily_returns = simulate_portfolio_returns(stock_allocations_df, 2500, 126)

nyse_cumulative_returns = np.cumsum(nyse_daily_returns)
ftse_cumulative_returns = np.cumsum(ftse_daily_returns)
# Combine simulations according to allocations
combined_cumulative_returns = allocation_nyse * nyse_cumulative_returns + allocation_ftse * ftse_cumulative_returns

# Calculate daily returns from cumulative returns
combined_daily_returns = np.diff(combined_cumulative_returns) / combined_cumulative_returns[:-1]

# Calculate the variance of the combined daily returns
variance_combined = np.var(combined_daily_returns, ddof=1)  # ddof=1 provides an unbiased estimator by using N-1 in the denominator

# Convert variance to a percentage for easier interpretation
variance_combined_percent = variance_combined * 100

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=['https://github.com/siwelbackwards/portfolio-dashboard/raw/main/style.css'],
           external_scripts=['https://cdn.jsdelivr.net/gh/siwelbackwards/portfolio-dashboard/my_clientside.js'])

# Defining the layout of the app
app.layout = html.Div([
    # Sidebar
    html.Div([
        # Wrapper div for logo and title with flex display
        html.Div([
            html.H2('Portfolio Dashboard', style={'color': '#ffffff', 'margin': '0px', 'padding': '0px'}),
            html.Img(src='https://raw.githubusercontent.com/siwelbackwards/portfolio-dashboard/main/stock_logo.png?raw=true', style={'height': '60px', 'align-self': 'center'}),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between',
                  'padding-right': '10px', 'padding-left': '10px'}),

        html.H4('Navigation', style={'color': '#ffffff'}),
        dcc.Dropdown(
            id='page-selector',
            options=[
                {'label': 'Home', 'value': 'Home'},
                {'label': 'FTSE Allocation', 'value': 'FTSE Allocation'},
                {'label': 'NYSE Allocation', 'value': 'NYSE Allocation'},
                {'label': 'Forecasted Returns', 'value': 'Forecasted Returns'},
                {'label': 'Help Page', 'value': 'Help Page'},
            ],
            value='Home',  # Default value
            style={'color': '#000000'},  # Ensuring text color inside the dropdown is black for visibility
        ),
        html.Br(),
        dcc.Markdown("""
Select the level of risk for your investment portfolio. 
The level of risk you choose will determine the level of investments and how they are balanced to meet your risk tolerance and expected return.

- **High Risk**: Primarily consists of equities with high volatility and potential for high returns. Suitable for investors with a higher tolerance for risk.
- **Medium Risk**: A balanced mix of equities and fixed income. Suitable for investors seeking a balance between growth and income.
- **Low Risk**: Focuses more on lower volatility, with a smaller portion allocated to high risk and return holds. Designed for conservative investors.

Your selection will dynamically adjust the portfolio allocation across different assets and sectors to match the chosen risk level.
""", style={'color': '#ffffff'}),
        dcc.Dropdown(
            id='risk-selector',
            options=[
                {'label': 'High', 'value': 'High'},
                {'label': 'Medium', 'value': 'Medium'},
                {'label': 'Low', 'value': 'Low'}
            ],
            value='Low',  # Default value
            style={'color': '#000000'},  # Adjusted for consistency in dropdown text color
            className='Selector'  # Using the class for dropdown styling
        )
    ], className='sidebar', style={'width': '20%', 'background-color': '#051c2c', 'color': '#ffffff', 'height': '100vh', 'padding': '10px'}),

    # Main content area
    html.Div(id='page-content', style={'width': '80%', 'padding': '20px'}),

    # The interval component for countdown updates
    dcc.Interval(
        id='interval-component',
        interval=1000,  # in milliseconds
        n_intervals=0  # start at 0
    ),
], style={'display': 'flex', 'flex-direction': 'row'})

@app.callback(
    Output('page-content', 'children'),
    [Input('page-selector', 'value')],
)
def render_page_content(page_selection):
    if page_selection == 'Home':
        portfolio_performance_fig = plot_portfolio_performance()
        target_date = calculate_next_target_date()
        now = datetime.now()
        remaining = target_date - now
        days = remaining.days
        hours, remainder = divmod(remaining.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        #industry allocation
        transposed_allocations = allocations_df[['Industry', 'Allocation']].set_index('Industry').T

        # Reset the index to turn the 'Industry' names into a regular column
        transposed_allocations.reset_index(drop=True, inplace=True)

        # Prepare the data for the DataTable
        transposed_data = [
            {transposed_allocations.columns[i]: transposed_allocations.iloc[0][i] for i in
             range(len(transposed_allocations.columns))}
        ]

        # Define the columns for the DataTable
        transposed_columns = [{'name': 'Industry', 'id': 'Metric'}] + [{'name': industry, 'id': industry} for industry in
                                                                     allocations_df['Industry']]
        # Use this transposed data to populate the DataTable
        industry_allocation_table = dash_table.DataTable(
            id='table',
            columns=transposed_columns,
            data=transposed_data,
            style_table={'height': 'auto', 'overflowY': 'auto'},
            style_as_list_view=True,  # Use list view for a more modern look
            style_header={
                'backgroundColor': '#053259',  # Header background color
                'color': 'white',  # Header font color
                'fontWeight': 'bold',  # Make header font bold
                'fontFamily': 'BowerBold',  # Apply the custom font
            },
            style_cell={
                'backgroundColor': 'white',
                'color': '#333333',
                'fontFamily': 'BowerBold',  # Ensuring the custom font is used here
                'border': '1px solid #e5e5e5',  # Light grey border for each cell
                'textAlign': 'left',  # Align text to left
                'padding': '10px',  # Add some padding for text in cells
            },
        )

        # country Allocation table
        allocation_table = dash_table.DataTable(
            columns=[{"name": "Market", "id": "market"}, {"name": "Allocation", "id": "allocation"}],
            data=[
                {"market": "NYSE", "allocation": f"{allocation_nyse * 100:.2f}%"},
                {"market": "FTSE", "allocation": f"{allocation_ftse * 100:.2f}%"}
            ],
            style_table={'margin-top': '20px'},  # Add some spacing between the timer and the table
            style_as_list_view=True,  # Use list view for a more modern look
            style_header={
                'backgroundColor': '#053259',  # Header background color
                'color': 'white',  # Header font color
                'fontWeight': 'bold',  # Make header font bold
                'fontFamily': 'BowerBold',  # Apply the custom font
            },
            style_cell={
                'backgroundColor': 'white',
                'color': '#333333',
                'fontFamily': 'BowerBold',  # Ensuring the custom font is used here
                'border': '1px solid #e5e5e5',  # Light grey border for each cell
                'textAlign': 'left',  # Align text to left
                'padding': '10px',  # Add some padding for text in cells
            },
        )
        #Expected 6 months return table
        return_graph = allocation_nyse*total_portfolio_return_nyse + allocation_ftse*total_portfolio_return
        return_graph = f"{return_graph:.2f}%"
        expected_returns_table = dash_table.DataTable(
            columns=[
                {"name": "Economic Phase", "id": "economic_phase"},
                {"name": "Past 6 Months Return", "id": "past6_return"}
            ],
            data=[
                {"economic_phase": economic_phase, "past6_return": return_graph}
            ],
            # Optional: Style the table to your liking
            style_table={'margin-top': '20px', 'width': '100%'},
            style_as_list_view=True,  # Use list view for a more modern look
            style_header={
                'backgroundColor': '#053259',  # Header background color
                'color': 'white',  # Header font color
                'fontWeight': 'bold',  # Make header font bold
                'fontFamily': 'BowerBold',  # Apply the custom font
            },
            style_cell={
                'backgroundColor': 'white',
                'color': '#333333',
                'fontFamily': 'BowerBold',  # Ensuring the custom font is used here
                'border': '1px solid #e5e5e5',  # Light grey border for each cell
                'textAlign': 'left',  # Align text to left
                'padding': '10px',  # Add some padding for text in cells
            },
        )

        top_row_layout = html.Div([
            # Past 6 Months Return
            html.Div([
                expected_returns_table
            ], style={'flex': '1', 'margin': '10px'}),  # Adjusted for flexibility and spacing

            # Market Allocation
            html.Div([
                allocation_table
            ], style={'flex': '1', 'margin': '10px'}),  # Adjusted for flexibility and spacing

            # Countdown Timer
            html.Div([
                html.Div(id='countdown-timer-display')
                # Ensure this is connected to the callback updating the countdown
            ], style={'flex': '1', 'margin': '10px', 'text-align': 'center'})
            # Adjusted for flexibility, spacing, and centering text
        ], style={'display': 'flex', 'justify-content': 'space-around', 'align-items': 'center'})

        home_page_layout = html.Div([
            top_row_layout,  # This is the new row layout

            # Industry Allocation Table
            html.Div([
                industry_allocation_table
            ], style={'margin-top': '20px'}),

            # Portfolio Performance Graph
            dcc.Graph(figure=portfolio_performance_fig, style={'margin-top': '20px'},config={
        'staticPlot': True,  # This disables zoom, pan, and other interactive behaviors
        'displayModeBar': False  # This hides the mode bar
    }),
        ])

        return home_page_layout
    elif page_selection == 'FTSE Allocation':
        # Calculate expected return for the next 6 months for FTSE
        expected_return_ftse_6_months = ftse_cumulative_returns[-1]
        long_allocations_fig = generate_pie_chart(allocations_df, 'Long Allocations', industry_stock_mapping)
        short_allocations_fig = generate_pie_chart(allocations_df, 'Short Allocations', industry_stock_mapping)

        return html.Div([
            html.Div([
                html.Div([
                    html.H4(f"Past 6 Month FTSE Portfolio Return: {total_portfolio_return:.2f}%")
                ], style={
                'border': '4px solid #053259',
                'padding': '10px',
                'border-radius': '5px',
                'background-color': '#ffffff',
                'text-align': 'center',
                'width': '50%',
                'margin': '5px'
                }),
                html.Div([
                    html.H4(f"Expected Return in the Next 6 Months: {expected_return_ftse_6_months:.2f}%"),
                ], style={
                'border': '4px solid #053259',
                'padding': '10px',
                'border-radius': '5px',
                'background-color': '#ffffff',
                'text-align': 'center',
                'width': '50%',
                'margin': '5px'
                })
            ], style={'display': 'flex', 'margin-bottom': '20px', 'justify-content': 'space-between'}),

            html.Div([
                dcc.Graph(figure=long_allocations_fig, style={'display': 'inline-block', 'width': '50%'}),
                dcc.Graph(figure=short_allocations_fig, style={'display': 'inline-block', 'width': '50%'}),
            ], style={
                'display': 'flex',
                'justify-content': 'center'
            })
        ])
    elif page_selection == 'NYSE Allocation':
        # Calculate expected return for the next 6 months for NYSE
        expected_return_nyse_6_months = nyse_cumulative_returns[-1]

        long_allocations_fig_nyse = generate_pie_chart(allocations_df, 'Long Allocations', industry_stock_mappingNYSE)
        short_allocations_fig_nyse = generate_pie_chart(allocations_df, 'Short Allocations', industry_stock_mappingNYSE)

        return html.Div([
            html.Div([
                html.Div([
                    html.H4(f"Past 6 Month NYSE Portfolio Return: {total_portfolio_return_nyse:.2f}%")
                ], style={
                'border': '4px solid #053259',
                'padding': '10px',
                'border-radius': '5px',
                'background-color': '#ffffff',
                'text-align': 'center',
                'width': '50%',
                'margin': '5px'
                }),
                html.Div([
                    html.H4(f"Expected Return in the Next 6 Months: {expected_return_nyse_6_months:.2f}%"),
                ], style={
                'border': '4px solid #053259',
                'padding': '10px',
                'border-radius': '5px',
                'background-color': '#ffffff',
                'text-align': 'center',
                'width': '50%',
                'margin': '5px'
                })
            ], style={'display': 'flex', 'margin-bottom': '20px', 'justify-content': 'space-between'}),

            html.Div([
                dcc.Graph(figure=long_allocations_fig_nyse, style={'display': 'inline-block', 'width': '50%'}),
                dcc.Graph(figure=short_allocations_fig_nyse, style={'display': 'inline-block', 'width': '50%'})
            ], style={
                'display': 'flex',
                'justify-content': 'center'
            })
        ])

    elif page_selection == 'Forecasted Returns':
        # Expected return in the next 6 months from the end of the combined portfolio number
        expected_return_6_months = combined_cumulative_returns[-1]  # Getting the last value

        # Generate future dates for the x-axis
        future_dates = generate_future_dates(datetime.now(), len(combined_cumulative_returns))

        # Plotting the combined cumulative returns with dates
        fig = go.Figure()

        # Add NYSE and FTSE forecasts as more transparent lines
        fig.add_trace(
            go.Scatter(x=future_dates, y=nyse_cumulative_returns, mode='lines', name='NYSE Cumulative Returns',
                       line=dict(color='#00a8f3', dash='dash'), opacity=0.5))
        fig.add_trace(
            go.Scatter(x=future_dates, y=ftse_cumulative_returns, mode='lines', name='FTSE Cumulative Returns',
                       line=dict(color='#051c2c', dash='dash'), opacity=0.5))

        # Add combined cumulative returns as a more prominent line
        fig.add_trace(
            go.Scatter(x=future_dates, y=combined_cumulative_returns, mode='lines', name='Combined Cumulative Returns',
                       line=dict(color='#6279ec')))

        # Update layout to match the home page graph style
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Cumulative Returns (%)',
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            xaxis=dict(tickformat='%B %Y', showline=True, showgrid=True, linecolor='black'),
            yaxis=dict(showline=True, showgrid=True, gridcolor='lightgrey', linecolor='black'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family='Arial, sans-serif', size=16, color='#333'),
            margin=dict(l=0, r=0, t=50, b=50)
        )

        # Create a layout that includes both the expected return text and the forecast graph
        forecast_layout = html.Div([
            # Expected Return and Variance in a row (Centered within a container)
            html.Div([  # Add a wrapper div for centering
                html.Div([
                    html.H4(f"Expected Return in the Next 6 Months: {expected_return_6_months:.2f}%")
                ], style={
                    'border': '4px solid #053259',
                    'padding': '10px',
                    'border-radius': '5px',
                    'background-color': '#ffffff',
                    'text-align': 'center',
                    'width': '50%',
                    'display': 'inline-block',
                    'margin': '5px'
                }),
                html.Div([
                    html.H4(f"Variance of Expected Return: {variance_combined_percent:.2f}%")
                ], style={
                    'border': '4px solid #053259',
                    'padding': '10px',
                    'border-radius': '5px',
                    'background-color': '#ffffff',
                    'text-align': 'center',
                    'width': '50%',
                    'display': 'inline-block',
                    'margin': '5px'
                }),
            ], style={'display': 'flex', 'justify-content': 'center'}),

            # Center the Graph
            html.Div(dcc.Graph(
                figure=fig,
                config={
                    'staticPlot': True,
                    'displayModeBar': False
                }
            ), style={'text-align': 'center'})  # Wrap in div and center it
        ], style={'text-align': 'center'}  # Center the main layout as well
        )
        return forecast_layout


    elif page_selection == 'Help Page':

        contact_form = html.Form([

            dcc.Input(type='hidden', name='_captcha', value='false'),
            dcc.Input(type='text', id='name', name='name', placeholder='Your name', required=True),
            dcc.Input(type='email', id='email', name='email', placeholder='Your email', required=True),
            dcc.Textarea(id='message', name='message', placeholder='Your message here'),
            # Contact Button
            html.Button('Submit', id='submit-btn', n_clicks=0),

        ], action='https://formsubmit.co/pmyld12@nottingham.ac.uk', method='POST')

        submission_message_div = html.Div(id='submission-message')
        return html.Div([
            html.H2("Help Page"),
            html.H4("Navigation"),
            html.P(
                "To navigate through the dashboard, use the dropdown menu on the sidebar labeled 'Navigation'. You can switch between different views, including:"),
            html.Ul([
                html.Li("Home: Overview of your portfolio's performance over the last 6 months."),
                html.Li("FTSE Allocation: Detailed insights into your allocations in the FTSE market."),
                html.Li("NYSE Allocation: Detailed insights into your allocations in the NYSE market."),
                html.Li(
                    "Forecasted Returns: Our forecast for your portfolio's performance based on the current economic phase."),
                html.Li("Help Page: You are here now!"),
            ]),
            html.H4("Risk Selector"),
            html.P(
                "You can tailor the dashboard view to match your investment risk level. Choose between 'Low', 'Medium', or 'High' risk to filter the information presented across the dashboard. This will adjust the investment strategies and expected returns to match your risk tolerance."),
            html.H4("Portfolio Performance"),
            html.P(
                "On the Home page, you'll find a graph that illustrates your portfolio's performance over time. This includes a breakdown of NYSE and FTSE allocations, as well as the combined portfolio value."),
            html.H4("Allocations"),
            html.P(
                "The FTSE and NYSE Allocation pages provide pie charts that show how your investments are distributed across different industries. You'll also see the total 6-month return for each market."),
            html.H4("Economic Phases and Forecasted Returns"),
            html.P(
                "Based on the current economic phase, our model predicts potential returns. This can help you understand potential future returns on investments"),
            # Contact form after help text:
            html.H4("Contact Us"),
            contact_form,
            submission_message_div
        ])
    else:
        # Return different content for other pages or empty div if not "Home"
        return html.Div([html.H3(f'{page_selection} Page Content')])

app.clientside_callback(
    """
    function(n_intervals) {
        return window.dash_clientside.clientside.update_timer(n_intervals);
    }
    """,
    Output('countdown-timer-display', 'children'),
    [Input('interval-component', 'n_intervals')]
)

if __name__ == '__main__':
    app.run_server(debug=True)