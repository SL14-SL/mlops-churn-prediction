import pandas as pd
import json
import subprocess
import random

# CONFIG
N_SAMPLES = 20
API_URL = "http://localhost:8000/predict"
API_KEY = "secret-token-123"
VAL_PATH = "data/raw/simulation_ground_truth.csv"  # ggf. anpassen

# 1. Load data
df = pd.read_csv(VAL_PATH)

# 2. Sample random rows
df_sample = df.sample(n=N_SAMPLES, random_state=random.randint(0, 10000))

# 3. Drop target column (wichtig!)
TARGET_COL = "churn"  # ggf. anpassen!
if TARGET_COL in df_sample.columns:
    df_sample = df_sample.drop(columns=[TARGET_COL])

# 4. Convert to API format
inputs = df_sample.to_dict(orient="records")

payload = {
    "inputs": inputs,
    "context": {"request_id": "val-test-batch"}
}

# 5. Call API via curl
curl_cmd = [
    "curl",
    "-X", "POST", API_URL,
    "-H", f"X-API-KEY: {API_KEY}",
    "-H", "Content-Type: application/json",
    "-d", json.dumps(payload)
]

print("🚀 Sending request with", N_SAMPLES, "samples...\n")

result = subprocess.run(curl_cmd, capture_output=True, text=True)

print("📥 Response:\n")
print(result.stdout)
response = json.loads(result.stdout)

actions = [p["action"] for p in response["predictions"]]
print("\n📊 Action distribution:")
print(pd.Series(actions).value_counts())
