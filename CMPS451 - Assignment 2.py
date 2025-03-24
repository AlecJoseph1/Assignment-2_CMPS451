import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
file_path = "pss10.csv"  # Change to your actual file path
df = pd.read_csv(file_path, parse_dates=["month"], index_col="month")

# Ensure data is sorted by date
df = df.sort_index()

# Select feature (assuming 'temperature' column exists)
data = df[['TEMP']].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Function to create sequences for LSTM
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

# Define time steps (e.g., using last 30 days to predict next day)
time_steps = 30
X, y = create_sequences(data_scaled, time_steps)

# Split data into train and test sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
def build_lstm_model():
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

# Build BiLSTM model
def build_bilstm_model():
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True), input_shape=(time_steps, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

# Train models
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

lstm_model = build_lstm_model()
lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

bilstm_model = build_bilstm_model()
bilstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate models
lstm_loss = lstm_model.evaluate(X_test, y_test)
bilstm_loss = bilstm_model.evaluate(X_test, y_test)

print(f"LSTM Model Loss: {lstm_loss}")
print(f"BiLSTM Model Loss: {bilstm_loss}")

# Predict next 6 months (assuming daily data)
future_days = 180
predictions = []

# Start with last known data
input_seq = X_test[-1]  

for _ in range(future_days):
    pred = lstm_model.predict(input_seq.reshape(1, time_steps, 1))[0]
    predictions.append(pred)
    
    # Update sequence (shift left and add new prediction)
    input_seq = np.roll(input_seq, -1)
    input_seq[-1] = pred

# Convert predictions back to original scale
predicted_temps = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(pd.date_range(start=df.index[-1], periods=future_days, freq="D"), predicted_temps, label="Predicted Soil Temp")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Predicted Soil Temperature for Next 6 Months")
plt.legend()
plt.show()