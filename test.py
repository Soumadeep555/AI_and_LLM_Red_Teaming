import numpy as np
from sklearn.ensemble import RandomForestClassifier
from engine import run_all_attacks, store_results

# User provides this
X_train = np.random.rand(100, 10)  # Example data
y_train = np.random.randint(0, 2, 100)
X_test = np.random.rand(50, 10)
y_test = np.random.randint(0, 2, 50)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Run engine
results = run_all_attacks(model, X_train, y_train, X_test, y_test)
store_results(results, model_id="my_model_1")

# View results
print(results)