import numpy as np
import matplotlib.pyplot as plt
from preprocessing import download_btc_data, scale_data, train_test_split, create_dataset
from model import build_lstm_model

# 1) Veriyi indir
print("Downloading BTC data...")
btc_data = download_btc_data()

# 2) Ölçekle
scaled_data, scaler = scale_data(btc_data)

# 3) Train/Test ayır
train_data, test_data = train_test_split(scaled_data)

# 4) Dataset oluştur
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 5) LSTM için reshape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 6) Model oluştur
model = build_lstm_model((X_train.shape[1], 1))

# 7) Eğit
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 8) Test tahminleri
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 9) 2025 tahminleri
future_data = scaled_data[-time_step:].reshape(1, time_step, 1)
future_predictions = []

for i in range(12):
    pred = model.predict(future_data)
    future_predictions.append(pred[0][0])
    pred_reshaped = np.array(pred).reshape(1, 1, 1)
    future_data = np.append(future_data[:, 1:, :], pred_reshaped, axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 10) Sonuçları yazdır
months = ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06',
          '2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12']

print("\n📈 2025 Bitcoin Price Predictions:")
print("-" * 35)
for month, price in zip(months, future_predictions):
    print(f"{month}: ${price[0]:,.2f}")

print("\n✅ Training completed!")
