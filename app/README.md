# RoomGuard — Room Occupancy Classification Web App

RoomGuard is a web app that classifies whether a room is **Occupied** or **Empty**
from two smartphone ambient-sensor signals: light intensity (Lux) and sound level (dB).
The model is a Decision Tree trained on 2,000 rows — 1,500 public rows from the
UCI Room Occupancy Estimation dataset (dataset 864) and 500 private rows recorded
with Phyphox on group members' smartphones (30-second logging interval, manual labels).

## Model

| Item | Value |
| --- | --- |
| Algorithm | Decision Tree (criterion=gini, max_depth=5, min_samples_leaf=10) |
| Split | 70/15/15 — train 1,400 / validation 300 / test 300 |
| CV accuracy | 97.29% |
| Test accuracy | 95.33% (precision 95.95%, recall 94.67%, F1 95.30%) |
| Confusion matrix (test) | [[144, 6], [8, 142]] (TN 144, FP 6, FN 8, TP 142) |
| Feature importance | Sound_dB 0.9006, Light_Lux 0.0994 |
| Class balance | 1,000 Occupied / 1,000 Empty |

## How to run

```bash
cd ~/workspace/roomguard/app
python -m venv .venv
# On this machine the project venv already exists at ~/workspace/roomguard/.venv
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000> in your browser.

## Pages

- `/` — dashboard with recent predictions and key metrics
- `/predict` — enter Light (0–600 Lux) and Sound (25–85 dB), get a prediction + recommendation
- `/api/predict` — JSON API, `POST {"Light_Lux": 180, "Sound_dB": 58}` → `{"prediction": "Occupied", ...}`
- `/history` — every prediction saved in SQLite (`roomguard.db`); edit label/note or delete
- `/dataset` — first 50 rows of the 2,000-row dataset + summary statistics
- `/model` — metrics, confusion matrix and the decision-tree rules
- `/about` — project background, dataset sources and method

## Files

- `app.py` — Flask app (model loading, routes, SQLite history)
- `templates/` — Jinja templates (English UI)
- `static/style.css` — styling (teal/navy + amber accent)
- `roomguard.db` — created automatically on first run; stores prediction history
