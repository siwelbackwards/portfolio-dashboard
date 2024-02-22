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


selected_stocks = best_stocks_per_industry

# Calculate biggest winner and loser
biggest_winner = selected_stocks.loc[selected_stocks['Change%'].idxmax()]
biggest_loser = selected_stocks.loc[selected_stocks['Change%'].idxmin()]
