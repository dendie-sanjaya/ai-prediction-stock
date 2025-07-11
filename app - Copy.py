import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. Konfigurasi dan Pengaturan ---
# Ganti dengan jalur file CSV data saham Anda
DATA_FILE = 'data_saham_multivariable.csv'
# Kolom yang akan digunakan sebagai fitur (input X)
FEATURES = ['Open', 'High', 'Low', 'Volume', 'Close'] # Contoh: Harga Pembukaan, Tertinggi, Terendah, Volume, Penutupan
# Kolom yang akan diprediksi (target Y)
TARGET = 'Close'
# Ukuran jendela waktu (timesteps) untuk LSTM
N_TIMESTEPS = 60 # Menggunakan 60 hari/periode data sebelumnya untuk memprediksi 1 hari/periode berikutnya
# Rasio pembagian data training dan testing
TEST_SIZE = 0.2
# Epoh pelatihan model
EPOCHS = 100
# Ukuran batch
BATCH_SIZE = 32
# Parameter Early Stopping
ES_PATIENCE = 10 # Berhenti jika tidak ada peningkatan validasi loss selama 10 epoh
# Rasio Dropout
DROPOUT_RATE = 0.2
# Jumlah hari yang akan diprediksi ke depan
PREDICT_DAYS_AHEAD = 5

# --- 2. Pengumpulan dan Pra-pemrosesan Data ---
def load_and_preprocess_data(file_path, features_cols, target_col, timesteps):
    df = pd.read_csv(file_path)
    # Pastikan kolom tanggal ada dan diatur sebagai indeks jika perlu untuk time series
    # df['Date'] = pd.to_datetime(df['Date'])
    # df.set_index('Date', inplace=True)

    # Pilih fitur dan target
    data = df[features_cols + [target_col]].values

    # Normalisasi data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Siapkan data untuk LSTM (X: fitur, Y: target)
    X, y = [], []
    for i in range(timesteps, len(scaled_data)):
        X.append(scaled_data[i-timesteps:i, :len(features_cols)]) # Ambil semua fitur
        y.append(scaled_data[i, features_cols.index(target_col)]) # Target adalah kolom target pada timesteps saat ini
    
    X, y = np.array(X), np.array(y)

    return X, y, scaler, df # Mengembalikan scaler dan dataframe asli untuk konteks prediksi

# --- 3. Pembagian Data Training dan Testing ---
def split_data(X, y, test_size):
    # Untuk prediksi rekursif, kita perlu memastikan data test mencakup data terbaru
    # sehingga kita bisa mengambil 'X_last_sequence' dari data test.
    # Oleh karena itu, kita tidak menggunakan train_test_split dari sklearn dengan shuffle=False,
    # tetapi membagi secara manual.
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test

# --- 4. Pembangunan Model LSTM ---
def build_lstm_model(input_shape, dropout_rate):
    model = Sequential()
    # Layer LSTM pertama
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Layer LSTM kedua
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # Layer output Dense
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- 5. Pelatihan Model ---
def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size, patience):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

# --- 6. Fungsi Pembantu untuk Inverse Transform Skala Multivariabel ---
def inverse_transform_prediction(scaled_prediction, scaler, features_cols, target_col):
    # Buat array dummy dengan jumlah kolom sesuai data asli (features + target)
    dummy_array = np.zeros((len(scaled_prediction), len(features_cols) + 1))
    # Masukkan prediksi skala ke posisi kolom target
    target_idx = features_cols.index(target_col)
    dummy_array[:, target_idx] = scaled_prediction.flatten()
    
    # Lakukan inverse_transform pada seluruh array dummy
    inverted_data = scaler.inverse_transform(dummy_array)
    # Ambil hanya kolom target yang sudah di-inverse_transform
    return inverted_data[:, target_idx]

# --- 7. Evaluasi Model pada Data Uji (Prediksi 1-langkah) ---
def evaluate_model_on_test(model, X_test, y_test, scaler, features_cols, target_col):
    predictions_scaled = model.predict(X_test)
    
    # Inverse transform prediksi dan nilai aktual
    predictions = inverse_transform_prediction(predictions_scaled, scaler, features_cols, target_col)
    actual_values = inverse_transform_prediction(y_test.reshape(-1, 1), scaler, features_cols, target_col) # y_test perlu direshape agar sesuai

    # Hitung Metrik
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    mae = mean_absolute_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)

    print(f"\n--- Metrik Evaluasi pada Data Uji (1-Langkah Prediksi) ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Visualisasi Prediksi vs Aktual
    plt.figure(figsize=(14, 7))
    plt.plot(actual_values, label='Harga Aktual')
    plt.plot(predictions, label='Harga Prediksi (1-Langkah)')
    plt.title('Prediksi Harga Saham vs Harga Aktual (Data Uji)')
    plt.xlabel('Waktu (Hari dari Awal Data Uji)')
    plt.ylabel('Harga Saham')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 8. Fungsi Prediksi Multi-Hari ke Depan ---
def predict_future_prices(model, initial_sequence, scaler, features_cols, target_col, timesteps, days_ahead):
    """
    Memprediksi harga saham untuk N hari ke depan secara rekursif.
    initial_sequence: Urutan data terakhir yang diketahui (sudah diskalakan).
                      Harus memiliki shape (1, timesteps, num_features).
    """
    future_predictions_scaled = []
    current_sequence = initial_sequence.copy() # Salinan agar tidak mengubah urutan asli

    for _ in range(days_ahead):
        # Prediksi satu langkah ke depan
        next_prediction_scaled = model.predict(current_sequence)[0, 0]
        future_predictions_scaled.append(next_prediction_scaled)
        
        # Perbarui urutan: hapus data terlama, tambahkan prediksi baru
        # Ini adalah bagian yang tricky: kita hanya memprediksi harga TARGET.
        # Untuk memprediksi fitur lain (Open, High, Low, Volume) secara akurat,
        # kita butuh model terpisah atau asumsi.
        # Untuk kesederhanaan, kita akan mengasumsikan fitur lain tetap atau
        # menggunakan nilai dari hari terakhir yang diketahui (ini simplifikasi besar!)
        
        # Buat array untuk "hari baru" yang diskalakan
        new_day_features_scaled = np.zeros(len(features_cols))
        # Masukkan prediksi harga TARGET ke posisi yang benar
        new_day_features_scaled[features_cols.index(target_col)] = next_prediction_scaled
        
        # Untuk fitur lain selain TARGET (misal Open, High, Low, Volume),
        # kita bisa menggunakan nilai dari hari terakhir di `current_sequence`
        # atau menerapkan heuristik lain. Ini adalah batasan prediksi rekursif
        # multivariabel jika Anda hanya memprediksi satu output (TARGET).
        # Untuk demo ini, kita akan pakai nilai dari hari terakhir yang diketahui untuk fitur non-target.
        last_known_day_features = current_sequence[0, -1, :] # fitur dari hari terakhir di urutan
        
        for i, col in enumerate(features_cols):
            if col != target_col:
                new_day_features_scaled[i] = last_known_day_features[i] # Asumsi: fitur non-target sama seperti hari sebelumnya

        # Tambahkan hari baru ke urutan dan hapus hari terlama
        current_sequence = np.append(current_sequence[:, 1:, :], new_day_features_scaled.reshape(1, 1, -1), axis=1)

    # Inverse transform semua prediksi
    predictions_actual_scale = inverse_transform_prediction(
        np.array(future_predictions_scaled).reshape(-1, 1), 
        scaler, 
        features_cols, 
        target_col
    )
    
    return predictions_actual_scale

# --- Eksekusi Utama Skrip ---
if __name__ == "__main__":
    print("Memuat dan memproses data...")
    X, y, scaler, df_original = load_and_preprocess_data(DATA_FILE, FEATURES, TARGET, N_TIMESTEPS)
    
    print(f"Shape X: {X.shape}, Shape y: {y.shape}")

    print("Membagi data training dan testing...")
    # Pembagian manual untuk memastikan X_test yang terbaru tersedia untuk prediksi masa depan
    split_index = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"Shape X_train: {X_train.shape}, Shape y_train: {y_train.shape}")
    print(f"Shape X_test: {X_test.shape}, Shape y_test: {y_test.shape}")

    print("Membangun model LSTM...")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), dropout_rate=DROPOUT_RATE)
    model.summary()

    print("Melatih model LSTM...")
    history = train_model(model, X_train, y_train, X_test, y_test, EPOCHS, BATCH_SIZE, ES_PATIENCE)

    print("Mengevaluasi model pada data uji (1-langkah prediksi)...")
    evaluate_model_on_test(model, X_test, y_test, scaler, FEATURES, TARGET)

    # --- Prediksi Harga untuk N Hari ke Depan ---
    print(f"\n--- Melakukan prediksi harga untuk {PREDICT_DAYS_AHEAD} hari ke depan ---")
    
    # Ambil urutan terakhir dari data historis yang tersedia (dari X_test paling akhir)
    # Ini adalah titik awal untuk prediksi masa depan
    last_sequence_known_data = X[-1:].copy() # Pastikan shape (1, timesteps, num_features)

    future_predictions = predict_future_prices(
        model, 
        last_sequence_known_data, 
        scaler, 
        FEATURES, 
        TARGET, 
        N_TIMESTEPS, 
        PREDICT_DAYS_AHEAD
    )

    print(f"\nHarga saham prediksi untuk {PREDICT_DAYS_AHEAD} hari ke depan:")
    for i, price in enumerate(future_predictions):
        print(f"Hari {i+1}: {price:.2f}")
    
    # Visualisasi prediksi masa depan
    # Gabungkan data aktual (hingga akhir data) dengan prediksi masa depan
    
    # Ambil harga penutupan aktual dari seluruh dataset untuk plot
    actual_closing_prices = df_original[TARGET].values

    # Buat indeks untuk plot
    indices = np.arange(len(actual_closing_prices))
    future_indices = np.arange(len(actual_closing_prices), len(actual_closing_prices) + PREDICT_DAYS_AHEAD)

    plt.figure(figsize=(16, 8))
    plt.plot(indices, actual_closing_prices, label='Harga Aktual Historis')
    plt.plot(future_indices, future_predictions, marker='o', linestyle='--', color='red', label=f'Prediksi {PREDICT_DAYS_AHEAD} Hari ke Depan')
    plt.title(f'Harga Saham Aktual dan Prediksi {PREDICT_DAYS_AHEAD} Hari Mendatang')
    plt.xlabel('Waktu (Hari)')
    plt.ylabel('Harga Saham')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot loss training dan validation
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()