import os
import requests
import toml
import time
from src.agent import run_agent, predict_risk

print("=== LLM DIAGNOSTIC TOOL ===")

# Task 3: Validate Token Usage
def load_token():
    token = None
    try:
        secrets = toml.load('.streamlit/secrets.toml')
        token = secrets.get('HF_TOKEN')
        print(f"Token read from .streamlit/secrets.toml: {'Yes (Length ' + str(len(token)) + ')' if token else 'No'}")
    except Exception as e:
        print("Could not read secrets.toml:", e)
    
    if not token:
        token = os.environ.get('HF_TOKEN')
        print(f"Token read from environment: {'Yes' if token else 'No'}")
    
    return token

hf_token = load_token()

if not hf_token or "paste_your_token" in hf_token:
    print("❌ ERROR: Valid HF_TOKEN not found! System will use Fallback.")
else:
    print("✅ HF_TOKEN found and starting with 'hf_'")

# Task 1, 2, 4, 5: Validate LLM Connectivity, Inspect Response, Model Compatibility, Debug Prompt
URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
print(f"\n--- Testing Free Serverless Inference on {URL.split('/')[-1]} ---")
headers = {"Authorization": f"Bearer {hf_token}"}
payload = {
    "inputs": "Respond with ONLY valid JSON: {\"status\": \"working\"}",
    "parameters": {"max_new_tokens": 20, "temperature": 0.1}
}

try:
    response = requests.post(URL, headers=headers, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("✅ SUCCESS (200 OK): Model is active and token is working!")
        res_json = response.json()
        print(res_json)
    else:
        print(f"❌ FAILURE ({response.status_code})")
        print(f"Raw Response: {response.text[:200]}")
        
        # Diagnostics
        if response.status_code == 401:
            print("Root Cause: 401 Unauthorized -> Token is invalid.")
        elif response.status_code == 403:
            print("Root Cause: 403 Forbidden -> Token lacks 'Inference' permissions.")
        elif response.status_code == 404:
            print("Root Cause: 404 Not Found -> Model is not actively hosted on free tier right now, OR token is totally unauthorized/missing.")
        elif response.status_code == 429:
            print("Root Cause: 429 Too Many Requests -> Rate Limited.")
        elif response.status_code == 503:
            print("Root Cause: 503 Service Unavailable -> Model loading. Try again limit.")
except Exception as e:
    print(f"❌ Exception occurred: {e}")

# Alternative Model Test (if primary fails)
if 'response' in locals() and response.status_code != 200:
    alt_model = "HuggingFaceH4/zephyr-7b-beta"
    print(f"\n--- Testing Alternative Model: {alt_model} ---")
    alt_url = f"https://api-inference.huggingface.co/models/{alt_model}"
    try:
        r_alt = requests.post(alt_url, headers=headers, json=payload, timeout=10)
        print(f"Status Code: {r_alt.status_code}")
        if r_alt.status_code == 200:
            print("✅ Alternative Model WORKS. Suggest switching.")
        else:
             print(f"Alternative model failed ({r_alt.status_code})")
    except Exception as e:
        print(f"Alternative model test failed: {e}")

# Task 6: Test Full Pipeline
print("\n--- Testing Full Agent Pipeline ---")
borrower = {
    'rev_util': 0.5, 'age': 35.0, 'late_30_59': 0.0,
    'debt_ratio': 0.5, 'monthly_inc': 5000.0, 'open_credit': 5.0,
    'late_90': 0.0, 'real_estate': 0.0, 'late_60_89': 0.0, 'dependents': 0.0,
}
t0 = time.time()
risk = predict_risk(borrower)
safe_risk = risk if risk is not None else 0.5
report = run_agent(borrower, safe_risk)
t1 = time.time()

print(f"Agent Engine completed in {t1-t0:.2f}s")
if report:
    print(f"Final Report Source: {report.get('_source')}")
    if report.get('_source') == 'llm':
        print("✅ Pipeline used LLM successfully.")
    else:
        print("⚠️ Pipeline fell back to Rule-Based engine.")
else:
    print("❌ Pipeline failed to return report.")

print("\n=== DIAGNOSTIC COMPLETE ===")
