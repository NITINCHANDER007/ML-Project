import pandas as pd
import numpy as np
import xgboost as xgb
import pickle, urllib.request, json, time, os
import tensorflow as tf
from keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
import concurrent.futures

API_KEY = "864tf-QuKjE-zseDp-LWbRI"
CACHE_FILE = "onet_cache.csv"

def get_onet(path):
    url = f"https://api-v2.onetcenter.org{path}"
    req = urllib.request.Request(url, headers={'X-API-Key': API_KEY, 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req) as r: return json.loads(r.read().decode())
    except: return {}

def fetch_single_career(occ):
    code, name = occ['code'], occ['title']
    s_data = get_onet(f"/online/occupations/{code}/details/skills")
    i_data = get_onet(f"/online/occupations/{code}/details/interests")
    s_map = {e.get('element_name'): e.get('value', 50) for e in s_data.get('element', []) if 'element_name' in e}
    i_map = {e.get('element_name'): e.get('value', 30) for e in i_data.get('element', []) if 'element_name' in e}
    return {'Code': code, 'Career': name, 'Math': s_map.get('Mathematics', 50), 'Prog': s_map.get('Programming', 10),
            'R': i_map.get('Realistic', 30), 'I': i_map.get('Investigative', 30), 'A': i_map.get('Artistic', 30),
            'S': i_map.get('Social', 30), 'E': i_map.get('Enterprising', 30), 'C': i_map.get('Conventional', 30)}

# --- STEP 1: INCREMENTAL DISCOVERY ---
if os.path.exists(CACHE_FILE):
    existing_df = pd.read_csv(CACHE_FILE)
    known_codes = set(existing_df['Code'].astype(str).tolist())
    print(f"📂 Memory Check: {len(known_codes)} careers already known.")
else:
    existing_df, known_codes = pd.DataFrame(), set()

print("🌐 Scanning O*NET for new career paths...")
all_occ = get_onet("/online/occupations/?start=1&end=500").get('occupation', [])
new_occupations = [o for o in all_occ if o['code'] not in known_codes]

if new_occupations:
    print(f"✨ Found {len(new_occupations)} NEW careers. Learning...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch_single_career, new_occupations))
    full_df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
    full_df.to_csv(CACHE_FILE, index=False)
else:
    print("✅ Knowledge base is up to date.")
    full_df = existing_df

# --- STEP 2: ENSEMBLE RETRAINING ---
training_rows = []
for _, row in full_df.iterrows():
    profile = row.drop(['Career', 'Code']).values
    for _ in range(25): # Data Augmentation
        noisy = [p + np.random.randint(-12, 12) for p in profile]
        training_rows.append(list(noisy) + [row['Career']])

df_train = pd.DataFrame(training_rows, columns=['Math','Prog','R','I','A','S','E','C','Career'])
le = LabelEncoder()
y = le.fit_transform(df_train['Career'])
X = df_train.drop('Career', axis=1).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("🚀 Retraining Ensemble (XGBoost + Neural Network)...")
xgb_model = xgb.XGBClassifier(n_estimators=100, objective='multi:softprob')
xgb_model.fit(X, y)

nn_model = tf.keras.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(le.classes_), activation='softmax')
])
nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
nn_model.fit(X_scaled, y, epochs=15, batch_size=64, verbose=0)

# Save Assets
pickle.dump(xgb_model, open("ensemble_xgb.pkl", "wb"))
nn_model.save("ensemble_nn.h5")
pickle.dump(le, open("ensemble_le.pkl", "wb"))
pickle.dump(scaler, open("ensemble_scaler.pkl", "wb"))
print("🏁 Update Complete.")