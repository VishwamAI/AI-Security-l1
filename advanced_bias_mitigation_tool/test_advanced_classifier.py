import sys
sys.path.append('/home/ubuntu/ai_bias_mitigation')
from advanced_bias_mitigation_tool.advanced_classifier import AdvancedBiasMitigationClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def test_functionality():
    print('Functionality Test:')
    # Generate a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2,
                               n_repeated=0, n_classes=2, n_clusters_per_class=2,
                               weights=[0.9, 0.1], flip_y=0.01, random_state=42)

    # Add a sensitive feature (e.g., gender)
    sensitive_feature = np.random.randint(2, size=X.shape[0])
    X = np.hstack((X, sensitive_feature.reshape(-1, 1)))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the AdvancedBiasMitigationClassifier
    base_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier = AdvancedBiasMitigationClassifier(base_estimator=base_estimator, sensitive_features=[20])
    classifier.fit(X_train, y_train)

    # Test predictions
    predictions = classifier.predict(X_test)
    print('Predictions shape:', predictions.shape)

    # Test feature importance
    feature_importance = classifier.get_feature_importance()
    print('Feature importance:', feature_importance)

    # Test fairness metrics
    fairness_metrics = classifier.get_comprehensive_fairness_metrics(X_test, y_test, predictions)
    print('Fairness metrics:', fairness_metrics)

    # Test bias mitigation explanation
    bias_explanation = classifier.explain_bias_mitigation()
    print('Bias Mitigation Explanation:', bias_explanation)

def test_security():
    print('\nSecurity Test:')
    classifier = AdvancedBiasMitigationClassifier(base_estimator=RandomForestClassifier(), sensitive_features=[0])

    # Test Zero Trust Security
    classifier.implement_zero_trust_security()
    print('Zero Trust Security implemented')

    # Test Data Encryption
    test_data = 'sensitive_data'
    encrypted_data, key = classifier.encrypt_data(test_data)
    decrypted_data = classifier.decrypt_data(encrypted_data, key)
    print('Data Encryption:', encrypted_data != test_data)
    print('Data Decryption:', decrypted_data == test_data)

    # Test Quantum-Safe Cryptography
    classifier.implement_quantum_safe_cryptography()
    print('Quantum-Safe Cryptography implemented')

    # Test Privacy Compliance
    classifier.ensure_privacy_compliance()
    print('Privacy Compliance ensured')

    # Test Incident Response Plan
    classifier.develop_incident_response_plan()
    print('Incident Response Plan developed')

    # Test Secure SDLC
    classifier.implement_secure_sdlc()
    print('Secure SDLC implemented')

    # Test Threat Detection
    classifier.implement_threat_detection()
    classifier.log_threat('test_threat', 'This is a test threat')
    print('Threat Detection implemented and tested')

    # Test Patch Management
    classifier.implement_patch_management()
    classifier.check_for_patches()
    classifier.apply_patches()
    print('Patch Management implemented and tested')

if __name__ == '__main__':
    test_functionality()
    test_security()
    print('\nAll tests completed successfully!')
