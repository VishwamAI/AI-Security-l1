import tensorflow as tf
from tensorflow import keras
import numpy as np
import shap

def build_explainable_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate some dummy data for demonstration
input_shape = (20,)
X_train = np.random.rand(1000, *input_shape)
y_train = np.random.randint(0, 2, (1000, 1))

# Create and train the model
model = build_explainable_model(input_shape)
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Create a SHAP explainer
explainer = shap.DeepExplainer(model, X_train[:100])

# Generate SHAP values for a sample
X_sample = X_train[:10]
shap_values = explainer.shap_values(X_sample)

# Print SHAP values for the first sample
print("SHAP values for the first sample:")
print(shap_values[0][0])

# Save the model
model.save("explainable_ai_security_bias_model.h5")
print("Explainable AI security and bias model created and saved.")

# Function to explain predictions
def explain_prediction(model, explainer, input_data):
    prediction = model.predict(input_data)
    shap_values = explainer.shap_values(input_data)

    print(f"Prediction: {prediction[0][0]}")
    print("Feature importance:")
    for i, importance in enumerate(shap_values[0][0]):
        print(f"Feature {i}: {importance}")

# Example usage of the explanation function
print("\nExplanation for a sample prediction:")
explain_prediction(model, explainer, X_sample[0:1])
