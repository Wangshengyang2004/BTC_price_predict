import numpy as np 
import pandas as pd 
filepath = 'BTC-USDT-from-2023-09-01-to-2023-10-19.csv'
data = pd.read_csv(filepath)
data = data.sort_values('Timestamp')
data['Date'] = data['Timestamp']
data.head()
data.shape
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_style("darkgrid")
# plt.figure(figsize = (15,9))
# plt.plot(data[['Close']])
# plt.xticks(range(0,data.shape[0],20), data['Date'].loc[::20], rotation=45)
# plt.title("****** Stock Price",fontsize=18, fontweight='bold')
# plt.xlabel('Date',fontsize=18)
# plt.ylabel('Close Price (USD)',fontsize=18)
# plt.show()

price = data[['Close']]
price.info()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

def split_data(stock, lookback):
    data_raw = stock.to_numpy() 
    data = []
    
    # you can free play（seq_length）
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

lookback = 20
x_train, y_train, x_test, y_test = split_data(price, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)


import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)


input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
    

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


import time

hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))

predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))

x_test[-1]

import math, time
from sklearn.metrics import mean_squared_error

# make predictions
y_test_pred = model(x_test)
print(y_test_pred)
# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
lstm.append(trainScore)
lstm.append(testScore)
lstm.append(training_time)

y_test_pred_df = pd.DataFrame(y_test_pred)

x_test.shape

# shift train predictions for plotting
trainPredictPlot = np.empty_like(price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

# shift test predictions for plotting
testPredictPlot = np.empty_like(price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

original = scaler.inverse_transform(price['Close'].values.reshape(-1,1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)


#----

forecast_days = 3000
def predict_future(model, input_seq, forecast_days):
    future_predictions = []

    for _ in range(forecast_days):
        with torch.no_grad():
            pred = model(input_seq)
            future_predictions.append(pred[0].item())
            
            # Create new input sequence by appending the predicted value and removing the first value
            new_seq = input_seq[0].numpy().flatten()
            new_seq = np.append(new_seq[1:], pred[0].item())
            input_seq = torch.FloatTensor(new_seq).view(1, -1, 1)

    return future_predictions

# Start forecasting using the last sequence from the test set
last_sequence = x_test[-1].view(1, -1, 1)
future_preds = predict_future(model, last_sequence, forecast_days)

# Inverse scale the predictions to get the actual values
future_preds_actual = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

plt.figure(figsize=(15, 6))
plt.plot(data['Close'].values, label='Historical Data')
plt.plot(np.arange(len(data['Close'].values), len(data['Close'].values) + forecast_days), future_preds_actual, label='Future Predictions', linestyle='solid', color='red')
plt.title("BTC-USDT Price Forecasting")
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
