import numpy as np
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox, AttributeInferenceBaseline
from art.estimators.regression.scikitlearn import ScikitlearnRegressor  # Import for type checking

def run_attribute_inference_regressor(art_model, X_train, y_train, X_test, y_test):
    if not isinstance(art_model, ScikitlearnRegressor):
        return {
            "success": False,
            "score": 0.0,
            "details": {"error": "Model is not a regressor; skipping attack."}
        }
    
    attack_feature = 3  # Example: bmi feature from your notebook
    attack_train_ratio = 0.75
    attack_train_size = int(len(X_train) * attack_train_ratio)
    attack_test_size = int(len(X_test) * attack_train_ratio)
    
    attack_x_train = X_train[:attack_train_size]
    attack_x_test = X_test[:attack_test_size]
    
    attack_x_test_predictions = art_model.predict(attack_x_test)
    attack_x_test_feature = attack_x_test[:, attack_feature].copy().reshape(-1, 1)
    x_test_for_attack = np.delete(attack_x_test, attack_feature, 1)
    
    bb_attack = AttributeInferenceBlackBox(art_model, attack_feature=attack_feature)
    bb_attack.fit(attack_x_train)
    
    # Infer for numerical feature (MSE)
    inferred = bb_attack.infer(x_test_for_attack, pred=attack_x_test_predictions)
    mse = np.sum((attack_x_test_feature - inferred) ** 2) / len(inferred)
    
    # Baseline
    baseline_attack = AttributeInferenceBaseline(attack_feature=attack_feature)
    baseline_attack.fit(attack_x_train)
    inferred_baseline = baseline_attack.infer(x_test_for_attack)
    baseline_mse = np.sum((attack_x_test_feature - inferred_baseline) ** 2) / len(inferred_baseline)
    
    # Success if MSE is lower than baseline (lower is better for MSE)
    success = mse < baseline_mse * 0.9  # 10% better than baseline
    score = min(1.0, max(0.0, (baseline_mse - mse) / baseline_mse))  # Normalized score
    
    return {
        "success": success,
        "score": score,
        "details": {"attack_mse": mse, "baseline_mse": baseline_mse}
    }