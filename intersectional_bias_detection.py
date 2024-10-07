import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    """
    df = pd.read_csv(file_path)

    # Encode categorical variables
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']):
        df[column] = le.fit_transform(df[column])

    return df

def train_model(X, y):
    """
    Train a Random Forest model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print classification report.
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def detect_intersectional_bias(model, X, sensitive_features):
    """
    Detect intersectional bias in the model predictions.
    """
    predictions = model.predict(X)

    # Create intersectional groups
    intersectional_groups = X[sensitive_features].apply(lambda x: '_'.join(x.astype(str)), axis=1)

    # Calculate prediction rates for each intersectional group
    group_predictions = pd.DataFrame({
        'group': intersectional_groups,
        'prediction': predictions
    })

    prediction_rates = group_predictions.groupby('group')['prediction'].mean()

    # Calculate overall prediction rate
    overall_rate = predictions.mean()

    # Calculate bias scores
    bias_scores = (prediction_rates - overall_rate).abs()

    return bias_scores

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('your_dataset.csv')

    # Define features and target
    X = df.drop('target_column', axis=1)
    y = df['target_column']

    # Train model
    model, X_test, y_test = train_model(X, y)

    # Evaluate model
    print("Model Evaluation:")
    evaluate_model(model, X_test, y_test)

    # Detect intersectional bias
    sensitive_features = ['race', 'gender', 'age_group']  # Replace with actual column names
    bias_scores = detect_intersectional_bias(model, X, sensitive_features)

    print("\nIntersectional Bias Scores:")
    print(bias_scores.sort_values(ascending=False))

if __name__ == "__main__":
    main()

# Note: This script assumes you have a dataset named 'your_dataset.csv'.
# You'll need to replace this with your actual dataset and adjust column names accordingly.
