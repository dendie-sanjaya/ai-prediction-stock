import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib # Import library joblib untuk memuat scaler

# --- 1. Konfigurasi dan Pengaturan (Harus Sesuai dengan Pelatihan) ---
# Ganti dengan jalur file CSV data saham historis baru Anda
HISTORY_DATA_FILE = 'history-data_saham_multivariable.csv'
# Nama file model dan scaler yang sudah disimpan
MODEL_FILE = 'model_saham_lstm.h5'
SCALER_FILE = 'scaler_saham.pkl'

# Kolom yang digunakan sebagai fitur (harus sama persis seperti saat pelatihan)
FEATURES = ['Open', 'High', 'Low', 'Volume', 'Close']
# Kolom yang menjadi target (harus sama persis seperti saat pelatihan)
TARGET = 'Close'
# Ukuran jendela waktu (timesteps) untuk LSTM (harus sama persis seperti saat pelatihan)
N_TIMESTEPS = 10
# Jumlah hari yang akan diprediksi ke depan
PREDICT_DAYS_AHEAD = 5

# --- 2. Fungsi Pembantu untuk Pra-pemrosesan Data (Mirip dengan Pelatihan) ---
def prepare_data_for_prediction(df, scaler, features_cols, target_col, timesteps):
    """
    Mempersiapkan data historis untuk prediksi, tanpa melatih scaler baru.
    """
    # Pilih fitur dan target
    # Pastikan urutan kolom sesuai dengan FEATURES + [TARGET] saat scaler dilatih
    data = df[features_cols + [target_col]].values

    # Skalakan data menggunakan scaler yang sudah dimuat
    scaled_data = scaler.transform(data)

    # Siapkan data untuk LSTM (hanya X terakhir yang dibutuhkan untuk prediksi rekursif)
    # Ambil urutan terakhir dari data yang diskalakan
    if len(scaled_data) < timesteps:
        raise ValueError(f"Data historis kurang dari N_TIMESTEPS ({timesteps}). Tidak bisa membuat urutan awal.")

    # Ambil urutan terakhir dari semua fitur yang diskalakan
    X_last_sequence = scaled_data[-timesteps:, :len(features_cols)]
    
    # Reshape untuk input model LSTM (sampel, timesteps, fitur)
    X_last_sequence = X_last_sequence.reshape(1, timesteps, len(features_cols))

    return X_last_sequence, data # Mengembalikan juga data asli yang diskalakan untuk inverse transform

# --- 3. Fungsi Pembantu untuk Inverse Transform Skala Multivariabel ---
def inverse_transform_prediction(scaled_prediction, scaler, features_cols, target_col):
    """
    Mengembalikan prediksi skala ke nilai asli menggunakan scaler multivariabel.
    """
    dummy_array = np.zeros((len(scaled_prediction), len(features_cols) + 1))
    target_idx = features_cols.index(target_col)
    dummy_array[:, target_idx] = scaled_prediction.flatten()
    
    inverted_data = scaler.inverse_transform(dummy_array)
    return inverted_data[:, target_idx]

# --- 4. Fungsi Prediksi Multi-Hari ke Depan (Sama seperti sebelumnya) ---
def predict_future_prices(model, initial_sequence, scaler, features_cols, target_col, timesteps, days_ahead):
    """
    Memprediksi harga saham untuk N hari ke depan secara rekursif.
    initial_sequence: Urutan data terakhir yang diketahui (sudah diskalakan).
                      Harus memiliki shape (1, timesteps, num_features).
    """
    future_predictions_scaled = []
    current_sequence = initial_sequence.copy()

    for _ in range(days_ahead):
        next_prediction_scaled = model.predict(current_sequence)[0, 0]
        future_predictions_scaled.append(next_prediction_scaled)
        
        # Buat array untuk "hari baru" yang diskalakan
        new_day_features_scaled = np.zeros(len(features_cols))
        target_idx = features_cols.index(target_col)
        new_day_features_scaled[target_idx] = next_prediction_scaled
        
        last_known_day_features = current_sequence[0, -1, :]
        
        for i, col in enumerate(features_cols):
            if col != target_col:
                new_day_features_scaled[i] = last_known_day_features[i] # Asumsi: fitur non-target sama seperti hari sebelumnya

        current_sequence = np.append(current_sequence[:, 1:, :], new_day_features_scaled.reshape(1, 1, -1), axis=1)

    predictions_actual_scale = inverse_transform_prediction(
        np.array(future_predictions_scaled).reshape(-1, 1), 
        scaler, 
        features_cols, 
        target_col
    )
    
    return predictions_actual_scale

# --- Eksekusi Utama Skrip Prediksi ---
if __name__ == "__main__":
    print("--- Program Prediksi Harga Saham ---")
    print(f"Memuat model dari: {MODEL_FILE}")
    print(f"Memuat scaler dari: {SCALER_FILE}")
    print(f"Memuat data historis dari: {HISTORY_DATA_FILE}")

    try:
        # Muat model dan scaler yang sudah disimpan
        model = load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        print("Model dan Scaler berhasil dimuat.")
    except Exception as e:
        print(f"Error saat memuat model atau scaler: {e}")
        print("Pastikan 'model_saham_lstm.h5' dan 'scaler_saham.pkl' ada di direktori yang sama.")
        exit()

    try:
        # Muat data historis baru
        df_history = pd.read_csv(HISTORY_DATA_FILE)
        print(f"Data historis dimuat. Jumlah baris: {len(df_history)}")
    except FileNotFoundError:
        print(f"Error: File '{HISTORY_DATA_FILE}' tidak ditemukan. Pastikan nama file dan jalurnya benar.")
        exit()
    except Exception as e:
        print(f"Error saat membaca data historis: {e}")
        exit()

    # Siapkan data untuk prediksi
    try:
        initial_sequence_for_prediction, original_history_data = prepare_data_for_prediction(
            df_history, scaler, FEATURES, TARGET, N_TIMESTEPS
        )
        print(f"Urutan awal untuk prediksi disiapkan. Shape: {initial_sequence_for_prediction.shape}")
    except ValueError as e:
        print(f"Error persiapan data: {e}")
        print(f"Pastikan '{HISTORY_DATA_FILE}' memiliki setidaknya {N_TIMESTEPS} baris data.")
        exit()
    except Exception as e:
        print(f"Error tak terduga saat persiapan data: {e}")
        exit()

    # Lakukan prediksi multi-hari
    print(f"\nMelakukan prediksi harga untuk {PREDICT_DAYS_AHEAD} hari ke depan...")
    future_predictions = predict_future_prices(
        model, 
        initial_sequence_for_prediction, 
        scaler, 
        FEATURES, 
        TARGET, 
        N_TIMESTEPS, 
        PREDICT_DAYS_AHEAD
    )

    print(f"\nPrediksi harga saham untuk {PREDICT_DAYS_AHEAD} hari ke depan:")
    for i, price in enumerate(future_predictions):
        print(f"Hari {i+1}: {price:.2f}")

    # --- Visualisasi Prediksi ---
    # Ambil harga penutupan aktual dari seluruh dataset historis
    actual_closing_prices_history = df_history[TARGET].values

    # Buat indeks untuk plot
    indices_history = np.arange(len(actual_closing_prices_history))
    future_indices = np.arange(len(actual_closing_prices_history), 
                               len(actual_closing_prices_history) + PREDICT_DAYS_AHEAD)

    plt.figure(figsize=(16, 8))
    plt.plot(indices_history, actual_closing_prices_history, label='Harga Aktual Historis')
    plt.plot(future_indices, future_predictions, marker='o', linestyle='--', color='red', 
             label=f'Prediksi {PREDICT_DAYS_AHEAD} Hari ke Depan')
    
    # Tambahkan tanda untuk titik awal prediksi (hari terakhir data historis)
    plt.axvline(x=len(actual_closing_prices_history) -1, color='green', linestyle=':', 
                label='Titik Awal Prediksi', alpha=0.7)

    plt.title(f'Harga Saham Aktual Historis dan Prediksi {PREDICT_DAYS_AHEAD} Hari Mendatang')
    plt.xlabel('Waktu (Hari)')
    plt.ylabel('Harga Saham')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediksi_harga_masa_depan_dari_history.png')
    plt.show() # Tampilkan plot
    plt.clf()