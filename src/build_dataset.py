"""
Build combined dataset for UAS Kel 12 — Evodia & Hasbi.
Topic: Web Classification of Room Occupancy with Decision Tree
       based on Ambience Sensors (Light Lux + Sound dB).

Composition (from assignment table, screenshot):
  Total 2000 rows = 1500 public (75%) + 500 private (25%)

Public (75%):
  UCI Room Occupancy Estimation Dataset (ID 864)
  https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation
  Original: 10,129 rows, 30-second interval, room 6m x 4.6m, 0-3 occupants.
  Used here: stratified random sample of 1500 rows (random_state=42).
  Features kept, exactly as in the assignment:
    Light_Lux = mean(S1_Light, S2_Light, S3_Light, S4_Light)  [Lux]
    Sound_dB  = 20*log10(mean(S1..S4_Sound)/0.05) + 35         [dB, calibrated]
  UCI sound is stored in Volts (microphone amplifier output read by ADC).
  It is NOT in dB. To make public + private share one unit (dB, as phyphox
  reports), Volts are mapped monotonically to dB with the formula above:
  V=0.05 -> 35 dB (quiet room), V=0.30 -> ~50.6 dB, V=3.18 -> ~71 dB.
  The mapping is monotonic, so a Decision Tree (threshold splits) keeps the
  same ordering; the formula and its reference point are documented here
  and in the report (Bab III) instead of being hidden.
  Target: Occupied = 1 if Room_Occupancy_Count > 0 else 0
          (binary, as in the assignment: Occupied / Empty).

Private (25%):
  500 rows following the phyphox protocol in the report (Bab III.2):
  smartphone on a table, phyphox Light + Sound Meter, one record every
  30 seconds, label written in a separate attendance log AT THE SAME TIME
  (headcount observed in the room), NEVER inferred from the sensor value
  afterwards. That is the reviewer's rule in the assignment table:
  labelling must follow a fixed standard, not subjective judgement.
  ------------------------------------------------------------------
  HONESTY NOTE — read this before presenting:
  The 500 private rows shipped in this folder were GENERATED with the
  calibration below (seeded, reproducible) so the system can be trained,
  tested and demonstrated now. They are NOT a real phyphox recording by
  Evodia/Hasbi yet. To replace them with real recordings:
    1. Record with phyphox (Light + Sound Meter), export CSV, 30 s interval.
    2. Put the export in dataset/private_phyphox_REAL.csv with columns
       timestamp, Light_Lux, Sound_dB, Occupied (from the attendance log).
    3. Run:  python src/build_dataset.py --real-private dataset/private_phyphox_REAL.csv
       then: python src/train_model.py
  The generator below is calibrated from the UCI distributions so the
  simulated private data has the same realistic overlap/noise as the
  public data (e.g. lamp left on while empty, person present in a dark
  quiet room), instead of being trivially separable.
"""
import argparse
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
BASE = "dataset"

def sound_v_to_db(v):
    v = np.maximum(np.asarray(v, dtype=float), 0.01)
    return 20 * np.log10(v / 0.05) + 35.0

def build_public(n=1500):
    uci = pd.read_csv("Occupancy_Estimation.csv")
    pub = uci.sample(n=n, random_state=42).copy()  # stratified? sample then check; use stratify below
    # redo stratified to preserve 81.2% / 18.8% split
    uci["Occupied"] = (uci["Room_Occupancy_Count"] > 0).astype(int)
    pub = (uci.groupby("Occupied", group_keys=False)
              .apply(lambda g: g.sample(n=int(round(n * len(g) / len(uci))), random_state=42)))
    pub = pub.sample(frac=1, random_state=42).reset_index(drop=True)
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(pub["Date"] + " " + pub["Time"]),
        "Light_Lux": pub[["S1_Light", "S2_Light", "S3_Light", "S4_Light"]].mean(axis=1).round(2),
        "Sound_dB": sound_v_to_db(pub[["S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"]].mean(axis=1)).round(2),
        "Occupied": pub["Occupied"].astype(int),
        "Source": "Public_UCI_864",
    })
    out["Label"] = out["Occupied"].map({0: "Empty", 1: "Occupied"})
    return out

def _clip(a, lo, hi):
    return np.clip(a, lo, hi)

def build_private_simulated(n=500):
    """Calibrated simulation, 250 Empty + 250 Occupied, 30-s interval sessions."""
    rng = np.random.default_rng(7)
    rows = []
    # 10 sessions x 50 records, alternating Empty/Occupied, 30 s apart
    sessions = [
        ("2026-11-10 09:00:00", 0), ("2026-11-10 10:00:00", 1),
        ("2026-11-11 09:00:00", 1), ("2026-11-11 14:00:00", 0),
        ("2026-11-12 09:00:00", 0), ("2026-11-12 13:00:00", 1),
        ("2026-11-13 09:00:00", 1), ("2026-11-13 15:00:00", 0),
        ("2026-11-14 09:00:00", 0), ("2026-11-14 11:00:00", 1),
    ]
    sid = 0
    for start, occ in sessions:
        sid += 1
        t0 = pd.Timestamp(start)
        for i in range(50):
            u = rng.random()
            if occ == 0:  # Empty: mostly dark+quiet, sometimes lamp/sunlight left on
                if u < 0.70:
                    light = rng.normal(4, 4);   sound = rng.normal(36, 2.5)
                elif u < 0.90:
                    light = rng.normal(32, 12); sound = rng.normal(39, 3)
                else:
                    light = rng.normal(55, 10); sound = rng.normal(44, 4)  # hard case
            else:         # Occupied: mostly bright+noisy, sometimes dark/quiet
                if u < 0.70:
                    light = rng.normal(105, 35); sound = rng.normal(54, 6)
                elif u < 0.90:
                    light = rng.normal(38, 12);  sound = rng.normal(47, 5)
                else:
                    light = rng.normal(10, 6);   sound = rng.normal(40, 3)  # hard case
            rows.append({
                "timestamp": t0 + pd.Timedelta(seconds=30 * i),
                "Light_Lux": round(float(_clip(light, 0, 220)), 2),
                "Sound_dB": round(float(_clip(sound, 30, 75)), 2),
                "Occupied": int(occ),
                "Source": "Private_phyphox_simulated",
                "Session": f"S{sid:02d}",
            })
    df = pd.DataFrame(rows)
    df["Label"] = df["Occupied"].map({0: "Empty", 1: "Occupied"})
    return df[["timestamp", "Light_Lux", "Sound_dB", "Occupied", "Source", "Label", "Session"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-private", default=None,
                    help="CSV with columns timestamp,Light_Lux,Sound_dB,Occupied to replace the simulated private set")
    args = ap.parse_args()
    pub = build_public(1500)
    if args.real_private:
        priv = pd.read_csv(args.real_private, parse_dates=["timestamp"])
        priv["Source"] = "Private_phyphox"
        priv["Label"] = priv["Occupied"].map({0: "Empty", 1: "Occupied"})
        priv = priv[["timestamp", "Light_Lux", "Sound_dB", "Occupied", "Source", "Label"]]
        print(f"Using REAL private file: {args.real_private} ({len(priv)} rows)")
    else:
        priv = build_private_simulated(500)
        print("Using SIMULATED private file (see honesty note in build_dataset.py).")
    combined = pd.concat([pub, priv], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    pub.to_csv(f"{BASE}/public_uci_sample_1500.csv", index=False)
    priv.to_csv(f"{BASE}/private_phyphox_500.csv", index=False)
    combined.to_csv(f"{BASE}/combined_dataset_2000.csv", index=False)
    print(f"public : {len(pub)}  Empty={(pub.Occupied==0).sum()} Occupied={(pub.Occupied==1).sum()}")
    print(f"private: {len(priv)}  Empty={(priv.Occupied==0).sum()} Occupied={(priv.Occupied==1).sum()}")
    print(f"combined: {len(combined)}  Empty={(combined.Occupied==0).sum()} Occupied={(combined.Occupied==1).sum()}")
    print(combined.groupby("Source")["Occupied"].value_counts())
    print(combined[["Light_Lux", "Sound_dB"]].describe().round(2))

if __name__ == "__main__":
    main()
