import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Portfoilo Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="auto")

alt.themes.enable("dark")

#######################
# Sidebar
with st.sidebar:
    st.title('Portfolio Dashboard')
    risk_list = ['High','Medium','Low']
    selected_risk = st.selectbox('Select your level of risk', risk_list)


# Define a function to format numbers
def format_number(number):
    # Formats the number to two decimal places and adds commas
    return f"{number:,.2f}"

# Load the Excel file
df = pd.read_excel('Stock test sheet.xlsx')

# Randomly select N stocks
N = random.randint(4, 15)  # Random stock numbers chosen
selected_stocks = df.sample(n=N)

# Calculate biggest winner and loser
biggest_winner = selected_stocks.loc[selected_stocks['Change%'].idxmax()]
biggest_loser = selected_stocks.loc[selected_stocks['Change%'].idxmin()]

# Create columns for layout
cols = st.columns((1.5, 1, 1), gap='medium')

# Display biggest win and loss
with cols[0]:
    st.markdown('#### Biggest Portfolio Winner')
    winner_name = biggest_winner['Ticker symbol']
    winner_price = format_number(biggest_winner['Price'])
    winner_change = format_number(biggest_winner['Change%'])
    st.metric(label=winner_name, value=f"${winner_price}", delta=f"{winner_change}%")
    
    st.markdown('#### Biggest Portfolio Loss')
    loser_name = biggest_loser['Ticker symbol']
    loser_price = format_number(biggest_loser['Price'])
    loser_change = format_number(biggest_loser['Change%'])
    st.metric(label=loser_name, value=f"${loser_price}", delta=f"{loser_change}%")

# Display pie chart
with cols[1]:
    # Prepare data for the pie chart
    industry_counts = selected_stocks['Industry'].value_counts()

    # Define a color palette (shades of blue)
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5']

    # Create a pie chart with Plotly
    fig = go.Figure(data=[go.Pie(labels=industry_counts.index, values=industry_counts.values, pull=[0.1]*len(industry_counts), hole=.3)])
    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                      marker=dict(colors=colors, line=dict(color='#000000', width=0.2)))

    # Update layout for dark mode compatibility and set the height
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'showlegend': True,
        'font': {'color': '#FFFFFF'},
        'height': 500,  # Specify the height of the pie chart
    })

    # Display the pie chart in Streamlit
    st.plotly_chart(fig)


# Initialize the portfolio
initial_value = 10_000_000  # 10 million pounds
portfolio_value = initial_value
years = 10
months = years * 12

# Dates range for the next 10 years, monthly
dates = pd.date_range(start=pd.Timestamp.now(), periods=months, freq='M')

# Determine growth multipliers based on risk selection
if selected_risk == 'Low':
    multipliers = np.random.uniform(0.98**(1/12), 1.05**(1/12), size=months)
elif selected_risk == 'Medium':
    multipliers = np.random.uniform(0.95**(1/12), 1.1**(1/12), size=months)
else:  # Assuming 'High' risk
    multipliers = np.random.uniform(0.8**(1/12), 1.3**(1/12), size=months)

# Simulate monthly growth
values = [initial_value]
for multiplier in multipliers:
    portfolio_value *= multiplier
    values.append(portfolio_value)

# Convert portfolio values to millions for y-axis labeling
values_in_millions = [value / 1_000_000 for value in values]

# The dates array should start from the current date, so we need to include the current month
# as the first entry in the dates array
dates = pd.date_range(start=pd.Timestamp.now(), periods=months + 1, freq='M')

# Make sure the lengths match by using the same number of periods for dates and values
assert len(dates) == len(values_in_millions), "The dates and values must be of the same length."

# Create a DataFrame for plotting
df2 = pd.DataFrame({'Date': dates, 'Portfolio Value': values_in_millions})

# Convert portfolio values to millions for y-axis labeling
values_in_millions = [value / 1_000_000 for value in values[1:]]
# Plotting the line chart for 10-Year Portfolio Growth using Plotly
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df2['Date'], y=df2['Portfolio Value'], mode='lines')
)

fig.update_layout(
    title='10-Year Portfolio Growth Simulation',
    xaxis_title='Year',
    yaxis_title='Portfolio Value (in Millions)',
    xaxis=dict(
        tickmode='auto', 
        nticks=10,
        tickformat="%Y"  # Format x-axis ticks to show only the year
    ),
    yaxis=dict(
        tickformat=".0f"  # Format y-axis ticks to two decimal places
    ),
    template='plotly_dark',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)'
)

# Display the line chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.write('This line chart simulates the growth of a portfolio starting with 10 million pounds, with a random monthly growth multiplier between 0.95 and 1.1 over the next 10 years.')

#contact form
st.header(":mailbox: Get In Touch for Help!")
contact_form = """
<form action="https://formsubmit.co/pmyld12@nottingham.ac.uk" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send</button>
</form>
"""
st.markdown(contact_form, unsafe_allow_html=True)

# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
