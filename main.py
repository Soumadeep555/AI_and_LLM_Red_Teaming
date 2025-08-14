import importlib
import os
import sqlite3
import datetime
from art.estimators.classification.scikitlearn import ScikitlearnClassifier  # Example wrapper
# Import other ART wrappers as needed

# List of attack modules (filenames without .py)
ATTACK_MODULES = [
    "attribute_inference",
    "attribute_inference_regressor",
    "membership_inference_regressor",
    "label_only_membership_inference",
    "membership_inference_shadow_models",
    "membership_inference"
]

def wrap_model(model):
    # Wrap user-provided model in ART (assume scikit-learn for now)
    return ScikitlearnClassifier(model)  # Adapt for regressors if needed

def run_all_attacks(model, X_train, y_train, X_test, y_test):
    art_model = wrap_model(model)
    results = {}
    
    for module_name in ATTACK_MODULES:
        try:
            module = importlib.import_module(f"attacks.{module_name}")
            attack_func = getattr(module, f"run_{module_name.replace('_', '')}")
            result = attack_func(art_model, X_train, y_train, X_test, y_test)
            results[module_name] = result
            print(f"Attack {module_name} completed: Success={result['success']}, Score={result['score']}")
        except Exception as e:
            results[module_name] = {"success": False, "score": 0.0, "details": {"error": str(e)}}
            print(f"Attack {module_name} failed: {e}")
    
    return results

def store_results(results, model_id="default_model"):
    conn = sqlite3.connect("results.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attack_results (
            id INTEGER PRIMARY KEY,
            model_id TEXT,
            attack_name TEXT,
            success BOOLEAN,
            score REAL,
            details TEXT,
            timestamp DATETIME
        )
    """)
    
    for attack_name, result in results.items():
        cursor.execute("""
            INSERT INTO attack_results (model_id, attack_name, success, score, details, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (model_id, attack_name, result["success"], result["score"], str(result["details"]), datetime.datetime.now()))
    
    conn.commit()
    conn.close()

# Query function (optional)
def get_results(model_id="default_model"):
    conn = sqlite3.connect("results.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attack_results WHERE model_id=?", (model_id,))
    return cursor.fetchall()