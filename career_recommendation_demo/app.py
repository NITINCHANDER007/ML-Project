import pickle, numpy as np, json, urllib.parse, urllib.request
import sqlite3, hashlib, os, secrets
from datetime import datetime
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# ─── Database Setup ───────────────────────────────────────────────────────────
DB_PATH = "career_ai.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT    NOT NULL,
            email     TEXT    UNIQUE NOT NULL,
            password  TEXT    NOT NULL,
            created   TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            domain      TEXT,
            top_matches TEXT,
            inputs      TEXT,
            created     TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()

init_db()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ─── Load AI Models ───────────────────────────────────────────────────────────
try:
    xgb_model  = pickle.load(open("ensemble_xgb.pkl",  "rb"))
    nn_model   = tf.keras.models.load_model("ensemble_nn.h5")
    le         = pickle.load(open("ensemble_le.pkl",    "rb"))
    scaler     = pickle.load(open("ensemble_scaler.pkl","rb"))
    st_model   = SentenceTransformer('all-MiniLM-L6-v2')
    MODELS_LOADED = True
except Exception as e:
    print(f"[WARN] Could not load models: {e}")
    MODELS_LOADED = False

API_KEY = "864tf-QuKjE-zseDp-LWbRI"

def get_onet(path):
    req = urllib.request.Request(
        f"https://api-v2.onetcenter.org{path}",
        headers={"X-API-Key": API_KEY, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())

# ─── Auth Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.json
    name  = data.get("name",  "").strip()
    email = data.get("email", "").strip().lower()
    pwd   = data.get("password", "")

    if not all([name, email, pwd]):
        return jsonify({"error": "All fields are required."}), 400

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (name, email, password, created) VALUES (?,?,?,?)",
            (name, email, hash_password(pwd), datetime.utcnow().isoformat())
        )
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        session["user_id"]   = user["id"]
        session["user_name"] = user["name"]
        return jsonify({"message": "Account created!", "name": name})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered."}), 409
    finally:
        conn.close()

@app.route("/api/login", methods=["POST"])
def login():
    data  = request.json
    email = data.get("email", "").strip().lower()
    pwd   = data.get("password", "")

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()

    if not user or user["password"] != hash_password(pwd):
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_id"]   = user["id"]
    session["user_name"] = user["name"]
    return jsonify({"message": "Login successful!", "name": user["name"]})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out."})

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html", user_name=session["user_name"])

# ─── Recommendation Route ─────────────────────────────────────────────────────
@app.route("/recommend", methods=["POST"])
def recommend():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated."}), 401

    if not MODELS_LOADED:
        return jsonify({"error": "AI models not loaded. Please check server setup."}), 503

    data = request.json
    prog_val = {"Low": 20, "Medium": 55, "High": 90}.get(data.get("Programming_Skill","Medium"), 50)

    # 1. Numerical Stream (XGBoost + NN Ensemble)
    raw_feats = np.array([[
        float(data["Math_Score"]),
        prog_val,
        float(data["R"]), float(data["I"]), float(data["A"]),
        float(data["S"]), float(data["E"]), float(data["C"])
    ]])

    xgb_probs    = xgb_model.predict_proba(raw_feats)[0]
    scaled_feats = scaler.transform(raw_feats)
    nn_probs     = nn_model.predict(scaled_feats, verbose=0)[0]

    final_probs  = (xgb_probs + nn_probs) / 2
    top_indices  = np.argsort(final_probs)[-3:][::-1]
    top_domains  = le.inverse_transform(top_indices)

    # 2. Hybrid Semantic Stream
    all_candidates = []
    text_search = get_onet(f"/online/search?keyword={urllib.parse.quote(data['Interest_Desc'])}")
    all_candidates.extend(text_search.get("occupation", []))
    domain_search = get_onet(f"/online/search?keyword={urllib.parse.quote(top_domains[0])}")
    all_candidates.extend(domain_search.get("occupation", []))

    unique_jobs  = list({v["code"]: v for v in all_candidates}.values())
    user_emb     = st_model.encode([data["Interest_Desc"]])

    final_matches = []
    if unique_jobs:
        job_titles = [j["title"] for j in unique_jobs]
        job_embs   = st_model.encode(job_titles)
        scores     = cosine_similarity(user_emb, job_embs)[0]
        for i, job in enumerate(unique_jobs):
            final_matches.append({"title": job["title"], "match": round(float(scores[i] * 100), 1)})
        final_matches = sorted(final_matches, key=lambda x: x["match"], reverse=True)[:5]

    result = {"domain": top_domains[0], "top_matches": final_matches, "alt_domains": list(top_domains[1:])}

    # Persist to DB
    conn = get_db()
    conn.execute(
        "INSERT INTO recommendations (user_id, domain, top_matches, inputs, created) VALUES (?,?,?,?,?)",
        (session["user_id"], top_domains[0], json.dumps(final_matches), json.dumps(data), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    return jsonify(result)

# ─── History Route ─────────────────────────────────────────────────────────────
@app.route("/api/history")
def history():
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated."}), 401

    conn = get_db()
    rows = conn.execute(
        "SELECT domain, top_matches, inputs, created FROM recommendations WHERE user_id=? ORDER BY created DESC LIMIT 10",
        (session["user_id"],)
    ).fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "domain":      r["domain"],
            "top_matches": json.loads(r["top_matches"]),
            "inputs":      json.loads(r["inputs"]),
            "created":     r["created"]
        })
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)