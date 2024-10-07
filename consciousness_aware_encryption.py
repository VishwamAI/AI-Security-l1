import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data for testing
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    return X, y

# Consciousness-Aware Encryption
def consciousness_aware_encryption(X, y):
    # Simulating encryption by adding noise to the input
    def add_noise(x):
        return x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.1)

    input_layer = keras.layers.Input(shape=(X.shape[1],))
    noise_layer = keras.layers.Lambda(add_noise)(input_layer)
    hidden_layer = keras.layers.Dense(64, activation='relu')(noise_layer)
    output_layer = keras.layers.Dense(1, activation='sigmoid')(hidden_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main execution
if __name__ == "__main__":
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = consciousness_aware_encryption(X_train_scaled, y_train)
    history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, verbose=1)

    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save the model
    model.save('consciousness_aware_encryption_model.h5')
    print("Model saved as consciousness_aware_encryption_model.h5")
