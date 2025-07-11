import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt # Diperlukan untuk visualisasi jika ingin ditampilkan

# --- 1. Konfigurasi dan Pengaturan (harus konsisten dengan app.py saat pelatihan) ---
DATA_FILE = 'data_saham_multivariable.csv'
FEATURES = ['Open', 'High', 'Low', 'Volume', 'Close']
TARGET = 'Close'
N_TIMESTEPS = 60 # HARUS SAMA DENGAN SAAT PELATIHAN MODEL!
PREDICT_DAYS_AHEAD = 5 # Jumlah hari yang ingin diprediksi ke depan

# --- 2. Fungsi Pembantu untuk Inverse Transform Skala Multivariabel ---
# Fungsi ini sama persis dengan yang ada di app.py
def inverse_transform_prediction(scaled_prediction, scaler, features_cols, target_col):
    dummy_array = np.zeros((len(scaled_prediction), len(features_cols) + 1))
    target_idx = features_cols.index(target_col)
    dummy_array[:, target_idx] = scaled_prediction.flatten()
    inverted_data = scaler.inverse_transform(dummy_array)
    return inverted_data[:, target_idx]

# --- 3. Fungsi Prediksi Multi-Hari ke Depan ---
# Fungsi ini sama persis dengan yang ada di app.py
def predict_future_prices(model, initial_sequence, scaler, features_cols, target_col, timesteps, days_ahead):
    future_predictions_scaled = []
    current_sequence = initial_sequence.copy() 

    for _ in range(days_ahead):
        next_prediction_scaled = model.predict(current_sequence)[0, 0]
        future_predictions_scaled.append(next_prediction_scaled)
        
        new_day_features_scaled = np.zeros(len(features_cols))
        target_idx = features_cols.index(target_col)
        new_day_features_scaled[target_idx] = next_prediction_scaled
        
        last_known_day_features = current_sequence[0, -1, :]
        
        for i, col in enumerate(features_cols):
            if col != target_col:
                new_day_features_scaled[i] = last_known_day_features[i]

        current_sequence = np.append(current_sequence[:, 1:, :], new_day_features_scaled.reshape(1, 1, -1), axis=1)

    predictions_actual_scale = inverse_transform_prediction(
        np.array(future_predictions_scaled).reshape(-1, 1), 
        scaler, 
        features_cols, 
        target_col
    )
    
    return predictions_actual_scale

# --- 4. Eksekusi Utama Skrip Prediksi ---
if __name__ == "__main__":
    print("Memuat model dan scaler...")
    try:
        loaded_model = load_model('model_saham_lstm.h5')
        loaded_scaler = joblib.load('scaler_saham.pkl')
        print("Model dan scaler berhasil dimuat.")
    except Exception as e:
        print(f"Error saat memuat model/scaler: {e}")
        print("Pastikan Anda sudah menjalankan 'app.py' terlebih dahulu untuk melatih dan menyimpan model/scaler.")
        exit() # Keluar jika model/scaler tidak dapat dimuat

    print(f"Memuat data historis terbaru dari '{DATA_FILE}'...")
    df = pd.read_csv(DATA_FILE)
    
    # Pilih fitur dan target, kemudian normalisasi
    data_for_scaling = df[FEATURES + [TARGET]].values
    scaled_data_full = loaded_scaler.transform(data_for_scaling) # Gunakan scaler yang dimuat

    # Ambil urutan terakhir dari data historis yang tersedia
    # Ini adalah "history" yang akan dimasukkan ke model untuk memulai prediksi
    if len(scaled_data_full) < N_TIMESTEPS:
        print(f"Error: Data historis ({len(scaled_data_full)} hari) tidak cukup untuk timesteps ({N_TIMESTEPS}).")
        exit()
        
    # Pastikan mengambil N_TIMESTEPS dari kolom fitur saja
    # Ambil N_TIMESTEPS baris terakhir, dan semua kolom fitur (tidak termasuk target)
    initial_sequence = scaled_data_full[-N_TIMESTEPS:, :len(FEATURES)] 
    initial_sequence = initial_sequence.reshape(1, N_TIMESTEPS, len(FEATURES)) # Reshape sesuai input model LSTM (samples, timesteps, features)

    print(f"\nMelakukan prediksi harga untuk {PREDICT_DAYS_AHEAD} hari ke depan...")
    future_predictions = predict_future_prices(
        loaded_model, 
        initial_sequence, 
        loaded_scaler, 
        FEATURES, 
        TARGET, 
        N_TIMESTEPS, 
        PREDICT_DAYS_AHEAD
    )

    print(f"\nHarga saham prediksi untuk {PREDICT_DAYS_AHEAD} hari ke depan:")
    # Untuk mendapatkan tanggal prediksi jika ada kolom tanggal di df_original
    last_date = pd.to_datetime(df['Date'].iloc[-1]) if 'Date' in df.columns else None # Asumsi ada kolom 'Date'
    
    for i, price in enumerate(future_predictions):
        predicted_date = last_date + pd.Timedelta(days=i+1) if last_date else f"Hari {i+1} dari Prediksi"
        print(f"{predicted_date}: {price:.2f}")

    # --- Visualisasi Hasil (Opsional) ---
    print("\nMenghasilkan visualisasi prediksi...")
    actual_closing_prices = df[TARGET].values

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
    plt.savefig('prediksi_masa_depan_terbaru.png') # Simpan plot prediksi baru
    plt.show() # Tampilkan plot (hanya jika Anda menjalankan di lingkungan interaktif)
    plt.clf()
    print("Visualisasi disimpan sebagai 'prediksi_masa_depan_terbaru.png'")