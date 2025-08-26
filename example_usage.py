import numpy as np
from sklearn.ensemble import RandomForestClassifier
from engine import run_all_attacks, store_results
from art.utils import load_nursery
import joblib  # Add this import for loading the pickle

# Load example dataset with transform_social=True to match the model's training data
(X_train, y_train), (X_test, y_test), _, _ = load_nursery(test_set=0.5, transform_social=True)

# User provides only the model path
model_path = "data/model_input/DecisionTreeClassifier.pkl"  # Replace with actual model path
model = joblib.load(model_path)

# Run engine
results = run_all_attacks(model, X_train, y_train, X_test, y_test)
store_results(results, model_id="decision_tree_nursery")

# Print results
for attack_name, result in results.items():
    print(f"Attack: {attack_name}")
    print(f"  Success: {result['success']}")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Details: {result['details']}")