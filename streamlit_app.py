import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
df = pd.read_excel(r'C:\Users\Lewis\Desktop\Stock test sheet.xlsx')

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
