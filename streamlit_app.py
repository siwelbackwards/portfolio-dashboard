import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Excel file
df = pd.read_excel('Stock test sheet.xlsx')

# Randomly select N stocks
N = 10  # Number of stocks to select
selected_stocks = df.sample(n=10)

# Prepare data for the pie chart
industry_counts = selected_stocks['Industry'].value_counts()

# Create a pie chart
fig, ax = plt.subplots()
ax.pie(industry_counts, labels=industry_counts.index, autopct='%1.1f%%')

# Display the pie chart in Streamlit
st.pyplot(fig)





# Initialize the portfolio
initial_value = 10_000_000  # 10 million pounds
portfolio_value = initial_value
years = 10
months = years * 12

# Dates range for the next 10 years, monthly
dates = pd.date_range(start=pd.Timestamp.now(), periods=months, freq='M')

# Simulate monthly growth
multipliers = np.random.uniform(0.95, 1.15, size=months)
values = [initial_value]

for multiplier in multipliers:
    portfolio_value *= multiplier
    values.append(portfolio_value)

# Create a DataFrame for plotting
df = pd.DataFrame({'Date': dates, 'Portfolio Value': values[1:]})

# Plotting
st.title('10-Year Portfolio Growth Simulation')
st.line_chart(df.set_index('Date'))

st.write('This line chart simulates the growth of a portfolio starting with 10 million pounds, with a random monthly growth multiplier between 0.95 and 1.15 over the next 10 years.')
