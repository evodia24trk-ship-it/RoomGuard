"""RoomGuard — Web-based Room Occupancy Classification.

Decision Tree (gini, max_depth=5, min_samples_leaf=10) trained on
2,000 rows: 1,500 public (UCI Room Occupancy Estimation, dataset 864)
+ 500 private (Phyphox smartphone sensor recordings).
Features: Light_Lux, Sound_dB. Target: 0 = Empty, 1 = Occupied.
"""
import json
import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, g, jsonify, redirect, render_template, request, url_for, flash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "decision_tree_roomguard.pkl")
METRICS_PATH = os.path.join(PROJECT_DIR, "models", "metrics.json")
DATASET_PATH = os.path.join(PROJECT_DIR, "data", "processed", "dataset_roomguard_2000.csv")
DB_PATH = os.path.join(BASE_DIR, "roomguard.db")

app = Flask(__name__)
app.secret_key = "roomguard-dev-key"

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

# ---------------------------------------------------------------- DB
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                light_lux REAL NOT NULL,
                sound_db REAL NOT NULL,
                result TEXT NOT NULL,
                probability REAL NOT NULL,
                note TEXT DEFAULT ''
            )
        """)
        conn.commit()

init_db()

# ---------------------------------------------------------------- helpers
def validate_inputs(light, sound):
    """Return (light, sound) floats or raise ValueError with a clear message."""
    try:
        light = float(light)
        sound = float(sound)
    except (TypeError, ValueError):
        raise ValueError("Light and Sound must be numbers.")
    if not (0 <= light <= 600):
        raise ValueError("Light must be between 0 and 600 Lux.")
    if not (25 <= sound <= 85):
        raise ValueError("Sound must be between 25 and 85 dB.")
    return light, sound

def predict_occupancy(light, sound):
    """Run the model. Returns dict with label, probability, recommendation."""
    proba = model.predict_proba([[light, sound]])[0]
    pred = int(model.predict([[light, sound]])[0])
    prob = float(proba[pred])
    label = "Occupied" if pred == 1 else "Empty"
    if pred == 1:
        recommendation = ("Room is occupied. Keep the lights as needed and "
                          "check the air conditioning — no need to switch everything off.")
    else:
        recommendation = ("Room looks empty. Turn off the lights and AC to save energy, "
                          "and check again in a few minutes before locking up.")
    return {"label": label, "label_id": pred, "probability": prob,
            "probability_pct": round(prob * 100, 1),
            "recommendation": recommendation,
            "light": light, "sound": sound}

def save_prediction(light, sound, result, note=""):
    db = get_db()
    db.execute(
        "INSERT INTO predictions (timestamp, light_lux, sound_db, result, probability, note)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), light, sound,
         result["label"], result["probability"], note),
    )
    db.commit()
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]

# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    occupied = db.execute("SELECT COUNT(*) c FROM predictions WHERE result='Occupied'").fetchone()["c"]
    recent = db.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 5").fetchall()
    return render_template("index.html", total=total, occupied=occupied,
                           empty=total - occupied, recent=recent, metrics=METRICS)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            light, sound = validate_inputs(request.form.get("light"), request.form.get("sound"))
        except ValueError as e:
            flash(str(e), "error")
            return render_template("predict.html", result=None)
        result = predict_occupancy(light, sound)
        note = request.form.get("note", "").strip()
        save_prediction(light, sound, result, note)
        flash("Prediction saved to history.", "success")
    return render_template("predict.html", result=result)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    # Accept both Light_Lux/Sound_dB and light/sound keys
    light = data.get("Light_Lux", data.get("light"))
    sound = data.get("Sound_dB", data.get("sound"))
    try:
        light, sound = validate_inputs(light, sound)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    result = predict_occupancy(light, sound)
    save_prediction(light, sound, result, data.get("note", ""))
    return jsonify({
        "Light_Lux": light, "Sound_dB": sound,
        "prediction": result["label"], "prediction_id": result["label_id"],
        "probability": round(result["probability"], 4),
        "recommendation": result["recommendation"],
    })

@app.route("/history")
def history():
    db = get_db()
    rows = db.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    return render_template("history.html", rows=rows)

@app.route("/history/<int:pid>/edit", methods=["POST"])
def history_edit(pid):
    label = request.form.get("result", "").strip()
    note = request.form.get("note", "").strip()
    if label not in ("Occupied", "Empty"):
        flash("Label must be Occupied or Empty.", "error")
        return redirect(url_for("history"))
    db = get_db()
    db.execute("UPDATE predictions SET result=?, note=? WHERE id=?", (label, note, pid))
    db.commit()
    flash(f"Record #{pid} updated.", "success")
    return redirect(url_for("history"))

@app.route("/history/<int:pid>/delete", methods=["POST"])
def history_delete(pid):
    db = get_db()
    db.execute("DELETE FROM predictions WHERE id=?", (pid,))
    db.commit()
    flash(f"Record #{pid} deleted.", "success")
    return redirect(url_for("history"))

@app.route("/dataset")
def dataset():
    df = pd.read_csv(DATASET_PATH)
    stats = {
        "total": len(df),
        "public": int((df["Source"] == "Public (UCI 864)").sum()),
        "private": int((df["Source"] == "Private (Phyphox)").sum()),
        "occupied": int((df["Occupancy"] == "Terisi").sum()),
        "empty": int((df["Occupancy"] == "Kosong").sum()),
        "light_mean": round(float(df["Light_Lux"].mean()), 1),
        "light_min": float(df["Light_Lux"].min()),
        "light_max": float(df["Light_Lux"].max()),
        "sound_mean": round(float(df["Sound_dB"].mean()), 1),
        "sound_min": float(df["Sound_dB"].min()),
        "sound_max": float(df["Sound_dB"].max()),
    }
    preview = df.head(50).to_dict(orient="records")
    return render_template("dataset.html", stats=stats, rows=preview, columns=list(df.columns))

@app.route("/model")
def model_page():
    from sklearn.tree import export_text
    rules = export_text(model, feature_names=["Light_Lux", "Sound_dB"])
    cm = METRICS["confusion_matrix_test"]  # [[TN, FP], [FN, TP]]
    return render_template("model.html", metrics=METRICS, cm=cm, rules=rules)

@app.route("/about")
def about():
    return render_template("about.html", metrics=METRICS)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
