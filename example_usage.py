import joblib
from engine import run_all_attacks, store_results
from art.utils import load_nursery
import numpy as np

# Load a sample dataset (e.g., Nursery dataset) in the backend
(X_train, y_train), (X_test, y_test), _, _ = load_nursery(test_set=0.5)

# User provides only the model path
model_path = "data/model_input/DecisionTreeClassifier.pkl"  # Replace with actual model path
model = joblib.load(model_path)

# Run engine with preloaded dataset
results = run_all_attacks(model, X_train, y_train, X_test, y_test)
store_results(results, model_id=model_path)

# Print results
print(results)