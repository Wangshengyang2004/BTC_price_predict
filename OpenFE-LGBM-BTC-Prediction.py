from openfe import OpenFE, transform
import lightgbm as lgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file
btc_data = pd.read_csv("data-BTC-USDT-from-2022-09-01-to-2023-10-05.csv")

# Convert 'date' to datetime
# btc_data['date'] = pd.to_datetime(btc_data['date'])

# Set 'date' as the index
btc_data.set_index('Timestamp', inplace=True)

# Sort the data by date
btc_data.sort_index(inplace=True)

# Data Cleaning: Drop rows with missing 'Close' values
btc_data = btc_data.dropna(subset=['Close'])

# Exploratory Data Analysis: Descriptive Statistics
print(btc_data.describe())

# Exploratory Data Analysis: Visualization (e.g., Heatmap)
# correlation_matrix = btc_data.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
# plt.title("Correlation Matrix")
# plt.show()

# Feature Engineering: Calculate Technical Indicators
btc_data['ema_12'] = btc_data['Close'].ewm(span=12, adjust=False).mean()
btc_data['ema_26'] = btc_data['Close'].ewm(span=26, adjust=False).mean()
btc_data['macd'] = btc_data['ema_12'] - btc_data['ema_26']
btc_data['signal_line'] = btc_data['macd'].ewm(span=9, adjust=False).mean()
btc_data['macd_histogram'] = btc_data['macd'] - btc_data['signal_line']

# More Technical Indicators Here...

# Label Creation: Example - Price Direction
btc_data['price_direction'] = (btc_data['Close'].shift(-1) >= btc_data['Close']).astype(int)

# Handle Missing Values for Technical Indicators
technical_indicator_columns = ['ema_12', 'ema_26', 'macd', 'signal_line', 'macd_histogram']
btc_data[technical_indicator_columns] = btc_data[technical_indicator_columns].fillna(method='bfill')

# Drop rows with missing labels
btc_data = btc_data.dropna(subset=['price_direction'])

# Feature Selection
feature_columns = ['Close','macd', 'signal_line', 'macd_histogram', 'ema_12', 'ema_26']

scaler = MinMaxScaler(feature_range=(-1, 1))
for col in feature_columns:
    btc_data[col] = scaler.fit_transform(btc_data[col].values.reshape(-1,1))

X = btc_data[feature_columns]
y = btc_data['price_direction']

# Data Splitting (80% training, 20% testing)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False)

ofe = OpenFE()

# n_jobs
n_jobs = 8

# generate new features
features = ofe.fit(data=train_x, label=train_y, n_jobs=n_jobs)
# transform the train and test data according to generated features. 
train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs) 

# Model Training: LightGBM for Price Direction Prediction
model_direction = lgb.LGBMClassifier()
model_direction.fit(train_x, train_y)

# Model Evaluation
y_pred_direction = model_direction.predict(test_x)
accuracy_direction = accuracy_score(test_y, y_pred_direction)
classification_report_direction = classification_report(test_y, y_pred_direction)

print("Accuracy:", accuracy_direction)
print("Classification Report:\n", classification_report_direction)



