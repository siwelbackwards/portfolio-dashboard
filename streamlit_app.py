import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the Excel file
df = pd.read_excel('Stock test sheet.xlsx')

# Randomly select N stocks
N = random.randint(4, 15) #random stock numbers chosen
selected_stocks = df.sample(n=N)

# Prepare data for the pie chart
industry_counts = selected_stocks['Industry'].value_counts()

# Define a color palette (shades of blue)
colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5']

# Create a pie chart with McKinsey styling
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(industry_counts, labels=industry_counts.index, autopct='%1.1f%%', colors=colors, startangle=140, textprops={'color':"w"})

# Improve the display of the labels and percentages
for text in texts:
    text.set_color('gray')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_size('x-large')
    autotext.set_weight('bold')

# Add a shadow for a 3D effect
plt.gca().set(aspect="equal", title='Industry Distribution')

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
