import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

class AdaptiveFairnessMetrics:
    def __init__(self, sensitive_features):
        self.sensitive_features = sensitive_features
        self.fairness_thresholds = {feature: 0.1 for feature in sensitive_features}
        self.historical_disparities = {feature: [] for feature in sensitive_features}

    def update_fairness_thresholds(self):
        """
        Update fairness thresholds based on historical disparities.
        """
        for feature in self.sensitive_features:
            if len(self.historical_disparities[feature]) > 0:
                mean_disparity = np.mean(self.historical_disparities[feature])
                self.fairness_thresholds[feature] = max(0.05, min(0.2, mean_disparity * 0.9))

    def calculate_disparity(self, y_true, y_pred, sensitive_feature):
        """
        Calculate disparity in predictions across different groups of a sensitive feature.
        """
        groups = pd.unique(sensitive_feature)
        group_disparities = []

        for group in groups:
            mask = sensitive_feature == group
            group_pred = y_pred[mask]
            group_true = y_true[mask]

            tn, fp, fn, tp = confusion_matrix(group_true, group_pred).ravel()
            group_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            group_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            group_disparities.append((group_fpr + group_fnr) / 2)

        max_disparity = max(group_disparities) - min(group_disparities)
        return max_disparity

    def evaluate_fairness(self, X, y_true, y_pred):
        """
        Evaluate fairness across all sensitive features.
        """
        fairness_results = {}

        for feature in self.sensitive_features:
            disparity = self.calculate_disparity(y_true, y_pred, X[feature])
            self.historical_disparities[feature].append(disparity)

            is_fair = disparity <= self.fairness_thresholds[feature]
            fairness_results[feature] = {
                'disparity': disparity,
                'threshold': self.fairness_thresholds[feature],
                'is_fair': is_fair
            }

        self.update_fairness_thresholds()
        return fairness_results

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    """
    df = pd.read_csv(file_path)

    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']):
        df[column] = le.fit_transform(df[column])

    return df

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('your_dataset.csv')

    # Define features, target, and sensitive attributes
    sensitive_features = ['race', 'gender', 'age_group']  # Replace with actual column names
    X = df.drop('target_column', axis=1)
    y = df['target_column']

    # Initialize AdaptiveFairnessMetrics
    afm = AdaptiveFairnessMetrics(sensitive_features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate fairness
    fairness_results = afm.evaluate_fairness(X_test, y_test, y_pred)

    print("Fairness Evaluation Results:")
    for feature, result in fairness_results.items():
        print(f"{feature}:")
        print(f"  Disparity: {result['disparity']:.4f}")
        print(f"  Threshold: {result['threshold']:.4f}")
        print(f"  Is Fair: {'Yes' if result['is_fair'] else 'No'}")
        print()

    print("Updated Fairness Thresholds:")
    for feature, threshold in afm.fairness_thresholds.items():
        print(f"{feature}: {threshold:.4f}")

if __name__ == "__main__":
    main()

# Note: This script assumes you have a dataset named 'your_dataset.csv'.
# You'll need to replace this with your actual dataset and adjust column names accordingly.
