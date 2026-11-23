"""RoomGuard — Web Room-Occupancy Classification (Decision Tree, Light + Sound).
Run:  pip install -r requirements.txt ; python app/app.py   -> http://127.0.0.1:5000
Backend: Flask + scikit-learn Decision Tree (model/decision_tree_model.pkl)
         + SQLite history (app/history.db) with full CRUD.
Frontend: English, responsive, no framework CDN dependency beyond local CSS/JS.
"""
import json, os, sqlite3, sys
from datetime import datetime
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
app = Flask(__name__, template_folder="templates", static_folder="static")
MODEL_PATH = os.path.join(BASE, "model", "decision_tree_model.pkl")
METRICS_PATH = os.path.join(BASE, "model", "metrics.json")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db")
FEATURES = ["Light_Lux", "Sound_dB"]
# Indoor sensor plausibility limits (validation, Bab III.7). Values outside
# are rejected with 400, not silently clipped — the user must fix the input.
LIMITS = {"Light_Lux": (0, 2000), "Sound_dB": (20, 120)}

model = joblib.load(MODEL_PATH)
with open(METRICS_PATH) as f:
    METRICS = json.load(f)

def db():
    c = sqlite3.connect(DB_PATH); c.row_factory = sqlite3.Row; return c

def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            light_lux REAL NOT NULL, sound_db REAL NOT NULL,
            prediction TEXT NOT NULL, probability REAL NOT NULL,
            true_label TEXT, note TEXT, source TEXT DEFAULT 'manual',
            created_at TEXT NOT NULL)""")

def decision_path(light, sound):
    """Human-readable trace of the tree path for one input (explainability)."""
    node = 0; steps = []
    tree = model.tree_
    while tree.children_left[node] != -1:
        f = FEATURES[tree.feature[node]]; thr = tree.threshold[node]
        val = light if f == "Light_Lux" else sound
        go_left = val <= thr
        steps.append(f"{f} = {val} {'<=' if go_left else '>'} {thr:.2f}")
        node = tree.children_left[node] if go_left else tree.children_right[node]
    leaf = tree.value[node][0]
    return steps, ("Occupied" if leaf[1] >= leaf[0] else "Empty")

def describe_and_recommend(pred, light, sound):
    if pred == "Occupied":
        return ("The room is occupied. Light and sound levels match the pattern "
                "of people being present in the room.",
                "Keep lights and ventilation on. If this is a study room, do not "
                "switch off the air conditioner. Log the occupancy for energy-use records.")
    return ("The room is empty. Light and sound levels match the pattern "
            "of an unoccupied room.",
            "Switch off lights and air conditioner to save energy. "
            "If the room should be empty, this also confirms no unauthorised use.")

def validate(light, sound):
    errs = []
    for name, val in (("Light_Lux", light), ("Sound_dB", sound)):
        if val is None or (isinstance(val, float) and val != val):
            errs.append(f"{name} is required.")
        elif not isinstance(val, (int, float)):
            errs.append(f"{name} must be a number.")
        else:
            lo, hi = LIMITS[name]
            if not (lo <= val <= hi):
                errs.append(f"{name} must be between {lo} and {hi} (got {val}).")
    return errs

def predict_one(light, sound):
    proba = model.predict_proba([[light, sound]])[0]
    idx = int(proba.argmax())
    pred = "Occupied" if idx == 1 else "Empty"
    steps, _ = decision_path(float(light), float(sound))
    desc, rec = describe_and_recommend(pred, light, sound)
    return {"prediction": pred, "probability": round(float(proba[idx]), 4),
            "probabilities": {"Empty": round(float(proba[0]), 4),
                              "Occupied": round(float(proba[1]), 4)},
            "decision_path": steps, "description": desc, "recommendation": rec}

@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)

@app.route("/api/model-info")
def model_info():
    return jsonify({"model": "DecisionTreeClassifier", "features": FEATURES,
                    "classes": ["Empty", "Occupied"], "metrics": METRICS,
                    "dataset": {"total": 2000, "public_uci_864": 1500, "private_phyphox": 500,
                                "train": 1400, "test": 600, "split": "70% train / 30% test"}})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    try:
        light = float(data.get("light_lux")) if data.get("light_lux") not in (None, "") else None
        sound = float(data.get("sound_db")) if data.get("sound_db") not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"error": "Light_Lux and Sound_dB must be numbers."}), 400
    errs = validate(light, sound)
    if errs:
        return jsonify({"error": " ".join(errs)}), 400
    out = predict_one(light, sound)
    with db() as c:
        cur = c.execute("INSERT INTO predictions (light_lux, sound_db, prediction, probability, source, created_at) "
                        "VALUES (?,?,?,?,?,?)", (light, sound, out["prediction"], out["probability"],
                                                 data.get("source", "manual"), datetime.now().isoformat(timespec="seconds")))
        out["id"] = cur.lastrowid
    return jsonify(out)

@app.route("/api/batch", methods=["POST"])
def api_batch():
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "No CSV file uploaded. The file must have Light_Lux and Sound_dB columns."}), 400
    f = request.files["file"]
    if f.content_length and f.content_length > 5 * 1024 * 1024:
        return jsonify({"error": "File too large (max 5 MB)."}), 400
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400
    if not {"Light_Lux", "Sound_dB"}.issubset(df.columns):
        return jsonify({"error": "CSV must contain Light_Lux and Sound_dB columns."}), 400
    if len(df) > 5000:
        return jsonify({"error": "Too many rows (max 5000)."}), 400
    rows, errors = [], []
    for i, r in df.iterrows():
        try:
            light, sound = float(r["Light_Lux"]), float(r["Sound_dB"])
        except (TypeError, ValueError):
            errors.append({"row": int(i) + 2, "error": "non-numeric value"}); continue
        errs = validate(light, sound)
        if errs:
            errors.append({"row": int(i) + 2, "error": " ".join(errs)}); continue
        out = predict_one(light, sound)
        out.update({"light_lux": light, "sound_db": sound})
        if "Occupied" in df.columns:
            try: out["true_label"] = "Occupied" if int(r["Occupied"]) == 1 else "Empty"
            except (TypeError, ValueError): pass
        rows.append(out)
        with db() as c:
            c.execute("INSERT INTO predictions (light_lux, sound_db, prediction, probability, true_label, source, created_at) "
                      "VALUES (?,?,?,?,?,?,?)", (light, sound, out["prediction"], out["probability"],
                                                 out.get("true_label"), "batch_csv", datetime.now().isoformat(timespec="seconds")))
    correct = sum(1 for r in rows if r.get("true_label") == r["prediction"])
    labelled = sum(1 for r in rows if "true_label" in r)
    return jsonify({"results": rows, "errors": errors,
                    "batch_accuracy": round(correct / labelled, 4) if labelled else None,
                    "n_labelled": labelled, "n_predicted": len(rows)})

@app.route("/api/history", methods=["GET"])
def history_list():
    with db() as c:
        rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 200").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/history/<int:pid>", methods=["PUT"])
def history_update(pid):
    data = request.get_json(force=True, silent=True) or {}
    note, true_label = data.get("note"), data.get("true_label")
    if true_label is not None and true_label not in ("Empty", "Occupied", ""):
        return jsonify({"error": "true_label must be Empty or Occupied."}), 400
    with db() as c:
        cur = c.execute("UPDATE predictions SET note=?, true_label=? WHERE id=?",
                        (note, true_label or None, pid))
        if cur.rowcount == 0:
            return jsonify({"error": f"No prediction with id {pid}."}), 404
    return jsonify({"ok": True, "id": pid})

@app.route("/api/history/<int:pid>", methods=["DELETE"])
def history_delete(pid):
    with db() as c:
        cur = c.execute("DELETE FROM predictions WHERE id=?", (pid,))
        if cur.rowcount == 0:
            return jsonify({"error": f"No prediction with id {pid}."}), 404
    return jsonify({"ok": True, "id": pid})

@app.route("/model/<path:name>")
def model_file(name):
    return send_from_directory(os.path.join(BASE, "model"), name)

if __name__ == "__main__":
    init_db()
    print("RoomGuard running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
