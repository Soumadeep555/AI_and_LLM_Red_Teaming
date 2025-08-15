import numpy as np
from art.attacks.inference.membership_inference import ShadowModels, MembershipInferenceBlackBox
from art.utils import to_categorical

def run_membership_inference_shadow_models(art_model, X_train, y_train, X_test, y_test):
    try:
        target_train_size = len(X_train) // 2
        x_target_train = X_train[:target_train_size]
        y_target_train = y_train[:target_train_size]
        x_target_test = X_train[target_train_size:]
        y_target_test = y_train[target_train_size:]
        
        shadow_models = ShadowModels(art_model, num_shadow_models=3)
        shadow_dataset = shadow_models.generate_shadow_dataset(X_test, to_categorical(y_test, nb_classes=len(np.unique(y_test))))
        (member_x, member_y, member_predictions), (nonmember_x, nonmember_y, nonmember_predictions) = shadow_dataset
        
        attack = MembershipInferenceBlackBox(art_model, attack_model_type="rf")
        attack.fit(member_x, member_y, nonmember_x, nonmember_y, member_predictions, nonmember_predictions)
        
        member_infer = attack.infer(x_target_train, y_target_train)
        nonmember_infer = attack.infer(x_target_test, y_target_test)
        
        member_acc = np.sum(member_infer) / len(x_target_train)
        nonmember_acc = 1 - np.sum(nonmember_infer) / len(x_target_test)
        acc = (member_acc * len(x_target_train) + nonmember_acc * len(x_target_test)) / (len(x_target_train) + len(x_target_test))
        
        baseline_acc = 0.5
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
            "details": {"error": f"Attack failed: {e}"}
        }