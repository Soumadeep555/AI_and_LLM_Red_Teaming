import numpy as np
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

def run_label_only_membership_inference(art_model, X_train, y_train, X_test, y_test):
    try:
        attack_train_ratio = 0.5
        attack_train_size = int(len(X_train) * attack_train_ratio)
        attack_test_size = int(len(X_test) * attack_train_ratio)
        
        bb_attack = MembershipInferenceBlackBox(art_model, attack_model_type='rf')
        bb_attack.fit(X_train[:attack_train_size], y_train[:attack_train_size],
                      X_test[:attack_test_size], y_test[:attack_test_size])
        
        inferred_train = bb_attack.infer(X_train.astype(np.float32), y_train)
        inferred_test = bb_attack.infer(X_test.astype(np.float32), y_test)
        
        train_acc = np.sum(inferred_train) / len(inferred_train)
        test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
        acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
        
        baseline_acc = 0.5  # Random guessing
        success = acc > baseline_acc + 0.1
        score = min(1.0, max(0.0, (acc - baseline_acc) / (1.0 - baseline_acc)))
        
        return {
            "success": success,
            "score": score,
            "details": {"attack_acc": acc, "baseline_acc": baseline_acc}
        }
    except Exception as e:
        return {
            "success": False,
            "score": 0.0,
            "details": {"error": f"Attack not applicable (e.g., model type mismatch): {e}"}
        }