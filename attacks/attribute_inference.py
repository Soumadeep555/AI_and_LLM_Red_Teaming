import numpy as np
from art.attacks.inference.attribute_inference import AttributeInferenceBlackBox, AttributeInferenceBaseline
from art.utils import to_categorical  # If needed for one-hot

def run_attribute_inference(art_model, X_train, y_train, X_test, y_test):
    # Adapt from Attribute_Inference_Attack_ART.ipynb
    # Assume attack on a specific feature (e.g., index 0); make configurable if needed
    attack_feature = 0  # Example: first feature
    attack_train_ratio = 0.75
    attack_train_size = int(len(X_train) * attack_train_ratio)
    attack_test_size = int(len(X_test) * attack_train_ratio)
    
    attack_x_train = X_train[:attack_train_size]
    attack_y_train = y_train[:attack_train_size]
    attack_x_test = X_test[:attack_test_size]
    attack_y_test = y_test[:attack_test_size]
    
    attack_x_test_predictions = art_model.predict(attack_x_test)
    attack_x_test_feature = attack_x_test[:, attack_feature].copy().reshape(-1, 1)
    x_test_for_attack = np.delete(attack_x_test, attack_feature, 1)
    
    bb_attack = AttributeInferenceBlackBox(art_model, attack_feature=attack_feature)
    bb_attack.fit(attack_x_train)
    
    # Infer and compute accuracy
    values = np.unique(attack_x_test_feature)  # Assume binary/categorical
    inferred = bb_attack.infer(x_test_for_attack, attack_x_test_predictions, values=values)
    acc = np.sum(inferred == attack_x_test_feature.reshape(1, -1)) / len(inferred)
    
    # Baseline
    baseline_attack = AttributeInferenceBaseline(attack_feature=attack_feature)
    baseline_attack.fit(attack_x_train)
    inferred_baseline = baseline_attack.infer(x_test_for_attack, values=values)
    baseline_acc = np.sum(inferred_baseline == attack_x_test_feature.reshape(1, -1)) / len(inferred_baseline)
    
    # Success if acc > baseline_acc + threshold (e.g., 0.1)
    success = acc > baseline_acc + 0.1
    score = min(1.0, max(0.0, (acc - baseline_acc) / (1.0 - baseline_acc)))  # Normalized vulnerability score
    
    return {
        "success": success,
        "score": score,
        "details": {"attack_acc": acc, "baseline_acc": baseline_acc}
    }