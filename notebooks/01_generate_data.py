import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1000 satır için zaman serisi
timestamps = [datetime(2025, 1, 1, 10, 0) + timedelta(minutes=i) for i in range(1000)]

# Normal değer aralıkları
heart_rate = np.random.normal(80, 5, 1000)
spo2 = np.random.normal(97, 1, 1000)
respiration_rate = np.random.normal(16, 2, 1000)
temperature = np.random.normal(36.8, 0.2, 1000)
bp_sys = np.random.normal(120, 5, 1000)
bp_dia = np.random.normal(80, 3, 1000)

# Anomali indeksleri (%5)
anomaly_indices = np.random.choice(range(1000), size=50, replace=False)

# Anomali değerleri
heart_rate[anomaly_indices] = np.random.randint(130, 180, 50)
spo2[anomaly_indices] = np.random.randint(80, 90, 50)
respiration_rate[anomaly_indices] = np.random.randint(25, 40, 50)
temperature[anomaly_indices] = np.random.uniform(38.5, 40.5, 50)
bp_sys[anomaly_indices] = np.random.randint(150, 180, 50)
bp_dia[anomaly_indices] = np.random.randint(95, 110, 50)

# Anomali etiketi
anomaly = np.zeros(1000)
anomaly[anomaly_indices] = 1

# DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "heart_rate": heart_rate.round(1),
    "spo2": spo2.round(1),
    "respiration_rate": respiration_rate.round(1),
    "temperature": temperature.round(1),
    "blood_pressure_sys": bp_sys.round(1),
    "blood_pressure_dia": bp_dia.round(1),
    "anomaly": anomaly
})

# CSV olarak kaydet
df.to_csv("health_vitals_1000.csv", index=False)

df.head()