import streamlit as st
import pandas as pd
import os
import time
import datetime
import pymongo

# Main entrance of the WebUI for Crypto Trading AI Prediction
# Include: Database management, Data visualization, Backtesting, Training, Prediction
# User can replace backbone such as RNN, LSTM, Transformer model to predict the price of BTC
# And able to download data from OKX exchange
# Able to specify the FACTORs to predict the price
# Hyperparameters can be adjusted