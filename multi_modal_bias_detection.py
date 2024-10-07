import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic multi-modal data for testing
def generate_synthetic_multi_modal_data(n_samples=1000):
    np.random.seed(42)
    X_text = np.random.randn(n_samples, 5)
    X_image = np.random.randn(n_samples, 5)
    X = np.concatenate((X_text, X_image), axis=1)
    y = ((X_text[:, 0] + X_image[:, 0]) > 0).astype(int)
    return X, y

# Multi-Modal Bias Detection and Correction
def multi_modal_bias_detection(X, y):
    text_input = keras.layers.Input(shape=(5,))
    image_input = keras.layers.Input(shape=(5,))

    text_features = keras.layers.Dense(32, activation='relu')(text_input)
    image_features = keras.layers.Dense(32, activation='relu')(image_input)

    combined = keras.layers.concatenate([text_features, image_features])

    # Bias detection layer
    bias_detection = keras.layers.Dense(16, activation='relu')(combined)
    bias_detection = keras.layers.Dense(1, activation='sigmoid', name='bias_output')(bias_detection)

    # Main classification layer
    main_output = keras.layers.Dense(16, activation='relu')(combined)
    main_output = keras.layers.Dense(1, activation='sigmoid', name='main_output')(main_output)

    model = keras.Model(inputs=[text_input, image_input], outputs=[main_output, bias_detection])
    model.compile(optimizer='adam',
                  loss={'main_output': 'binary_crossentropy', 'bias_output': 'binary_crossentropy'},
                  loss_weights={'main_output': 1.0, 'bias_output': 0.2},
                  metrics={'main_output': ['accuracy'], 'bias_output': ['accuracy']})
    return model

# Main execution
if __name__ == "__main__":
    X, y = generate_synthetic_multi_modal_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_text = X_train_scaled[:, :5]
    X_train_image = X_train_scaled[:, 5:]
    X_test_text = X_test_scaled[:, :5]
    X_test_image = X_test_scaled[:, 5:]

    model = multi_modal_bias_detection(X_train_scaled, y_train)
    history = model.fit([X_train_text, X_train_image], [y_train, y_train], epochs=50, validation_split=0.2, verbose=1)

    test_loss, main_loss, bias_loss, main_accuracy, bias_accuracy = model.evaluate([X_test_text, X_test_image], [y_test, y_test])
    print(f"Main task test accuracy: {main_accuracy:.4f}")
    print(f"Bias detection test accuracy: {bias_accuracy:.4f}")

    # Save the model
    model.save('multi_modal_bias_detection_model.h5')
    print("Model saved as multi_modal_bias_detection_model.h5")
