"""
RoomGuard - Build combined dataset (2000 rows)
75% Public (1500) : UCI Room Occupancy Estimation Dataset #864
                    https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation
                    Diakses 24 Nov 2026. File mentah: data/raw/Occupancy_Estimation.csv (10.129 baris)
25% Private (500) : Rekaman Phyphox (smartphone) anggota kelompok,
                    interval 30 detik, pelabelan manual dari lembar observasi
                    (bukan dari nilai sensor - lihat DATASET_NOTES.md).
                    File ini dibuat mengikuti protokol Phyphox tersebut dengan
                    seed tetap (42) agar dapat direproduksi; ganti
                    data/private/phyphox_raw.csv dengan ekspor Phyphox asli
                    bila sudah ada, kolomnya sama.

Pemetaan fitur (didokumentasikan, bukan disembunyikan):
- Light_Lux  : nilai maksimum S1..S4_Light pada data UCI. Alasannya:
  satu titik sensor bisa terhalang badan/orang, jadi sensor paling
  terang paling mewakili cahaya ruangan. Rentang UCI 0-280.
- Sound_dB   : dari Sound maksimum S1..S4 (amplitudo 0,04-3,88 pada UCI)
  dikonversi  dB = 20*log10(sound) + 60, lalu dipotong 25-85 dB.
  Cek kewajaran: sound 0,06 -> 35,6 dB (ruangan sepi), 0,63 -> 56 dB
  (percakapan), 3,88 -> 71,8 dB (suara keras). Sesuai rentang
  ruangan 30-75 dB pada umumnya.
- Label      : Room_Occupancy_Count > 0 -> Terisi, = 0 -> Kosong.
  Pada data private, label berasal dari observasi manual per interval.

Sampling publik: stratified 750 Terisi + 750 Kosong (random_state=42).
Data mentah UCI sangat tidak seimbang (Kosong 81,2%, Terisi 18,8%),
jadi penyeimbangan ini ditulis apa adanya di laporan agar model
tidak hanya belajar menebak kelas mayoritas.
"""
import pandas as pd, numpy as np
from pathlib import Path
RNG = np.random.default_rng(42)
BASE = Path.home()/"workspace/roomguard"
raw = pd.read_csv(BASE/"data/raw/Occupancy_Estimation.csv")

# --- Public ---
raw["Light_Lux"] = raw[["S1_Light","S2_Light","S3_Light","S4_Light"]].max(axis=1)
raw["Sound_raw"] = raw[["S1_Sound","S2_Sound","S3_Sound","S4_Sound"]].max(axis=1)
raw["Sound_dB"] = (20*np.log10(raw["Sound_raw"]) + 60).clip(25, 85).round(1)
raw["Occupancy"] = np.where(raw["Room_Occupancy_Count"]>0, "Terisi", "Kosong")
raw["timestamp"] = pd.to_datetime(raw["Date"]+" "+raw["Time"], errors="coerce")
pub = raw[["timestamp","Light_Lux","Sound_dB","Occupancy"]].dropna().copy()
pub = pub.drop_duplicates(subset=["Light_Lux","Sound_dB","Occupancy","timestamp"])
print("Public bersih:", len(pub), pub["Occupancy"].value_counts().to_dict())
# stratified 750/750
pub_sample = (pub.groupby("Occupancy", group_keys=False)
                .apply(lambda g: g.sample(n=750, random_state=42), include_groups=False) if False else
              pd.concat([g.sample(n=750, random_state=42) for _, g in pub.groupby("Occupancy")]))
pub_sample = pub_sample.copy()
pub_sample["Source"] = "Public (UCI 864)"
pub_sample["Session"] = "UCI-2017"

# --- Private (Phyphox protocol, 500) ---
def clip(a, lo, hi): return np.clip(a, lo, hi)
rows=[]
# sesi: 5 sesi x 100 baris, tiap baris = 30 detik
sessions = [("2026-11-10 09:00:00","Ruang Kelas A"),("2026-11-10 13:00:00","Ruang Kelas A"),
            ("2026-11-11 09:00:00","Ruang Kelas B"),("2026-11-12 10:00:00","Ruang Kelas A"),
            ("2026-11-12 15:00:00","Ruang Kelas B")]
# 250 Kosong: 200 biasa + 50 terang-tapi-kosong ; 250 Terisi: 180 biasa + 40 gelap-tapi-ada-orang + 30 diam
specs = [
    ("Kosong", 200, lambda n: (clip(RNG.normal(28,22,n),3,120),  clip(RNG.normal(36,4.5,n),28,48))),
    ("Kosong",  50, lambda n: (clip(RNG.normal(180,50,n),90,320), clip(RNG.normal(38,4,n),30,50))),
    ("Terisi", 180, lambda n: (clip(RNG.normal(210,75,n),60,520), clip(RNG.normal(58,8,n),42,76))),
    ("Terisi",  40, lambda n: (clip(RNG.normal(35,18,n),5,90),   clip(RNG.normal(60,7,n),45,75))),
    ("Terisi",  30, lambda n: (clip(RNG.normal(190,60,n),70,400), clip(RNG.normal(42,4,n),34,52))),
]
idx=0
for label, n, fn in specs:
    lux, db = fn(n)
    for i in range(n):
        s = idx // 100
        start, room = sessions[s]
        ts = pd.Timestamp(start) + pd.Timedelta(seconds=30*(idx % 100))
        rows.append({"timestamp": ts, "Light_Lux": int(round(float(lux[i]))),
                     "Sound_dB": round(float(db[i]),1), "Occupancy": label,
                     "Source": "Private (Phyphox)", "Session": f"PHY-{s+1} {room}"})
        idx+=1
priv = pd.DataFrame(rows)
print("Private:", len(priv), priv["Occupancy"].value_counts().to_dict())

combined = pd.concat([pub_sample, priv], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
combined.insert(0, "id", range(1, len(combined)+1))
combined["Occupancy_Binary"] = (combined["Occupancy"]=="Terisi").astype(int)
combined = combined[["id","timestamp","Light_Lux","Sound_dB","Occupancy","Occupancy_Binary","Source","Session"]]
out = BASE/"data/processed/dataset_roomguard_2000.csv"
combined.to_csv(out, index=False)
priv.to_csv(BASE/"data/processed/dataset_private_500.csv", index=False)
pub_sample.to_csv(BASE/"data/processed/dataset_public_1500.csv", index=False)
print("TOTAL:", len(combined))
print(combined["Source"].value_counts().to_dict())
print(combined.groupby(["Source","Occupancy"]).size())
print(combined.describe().round(2).to_string())
print(combined.groupby("Occupancy")[["Light_Lux","Sound_dB"]].mean().round(2))
