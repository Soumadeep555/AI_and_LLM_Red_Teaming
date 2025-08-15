import importlib
import os
import sqlite3
import datetime
import logging
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators.regression.scikitlearn import ScikitlearnRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    # Wrap user-provided model in ART
    try:
        if hasattr(model, 'predict_proba'):  # Classifier
            return ScikitlearnClassifier(model)
        else:  # Regressor
            return ScikitlearnRegressor(model)
    except Exception as e:
        logging.error(f"Failed to wrap model: {e}")
        raise ValueError(f"Model wrapping failed: {e}")

def run_all_attacks(model, X_train, y_train, X_test, y_test):
    art_model = wrap_model(model)
    results = {}
    
    for module_name in ATTACK_MODULES:
        try:
            module = importlib.import_module(f"attacks.{module_name}")
            # Use the exact function name with underscores
            attack_func = getattr(module, f"run_{module_name}")
            result = attack_func(art_model, X_train, y_train, X_test, y_test)
            results[module_name] = result
            logging.info(f"Attack {module_name} completed: Success={result['success']}, Score={result['score']}, Details={result['details']}")
        except AttributeError as e:
            logging.error(f"Attack {module_name} failed: {e}")
            results[module_name] = {"success": False, "score": 0.0, "details": {"error": str(e)}}
        except Exception as e:
            logging.error(f"Attack {module_name} failed with unexpected error: {e}")
            results[module_name] = {"success": False, "score": 0.0, "details": {"error": str(e)}}
    
    return results

def store_results(results, model_id="default_model"):
    try:
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
        logging.info("Results stored in database")
    except Exception as e:
        logging.error(f"Failed to store results: {e}")
    finally:
        conn.close()

def get_results(model_id="default_model"):
    try:
        conn = sqlite3.connect("results.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attack_results WHERE model_id=?", (model_id,))
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        logging.error(f"Failed to retrieve results: {e}")
        return []