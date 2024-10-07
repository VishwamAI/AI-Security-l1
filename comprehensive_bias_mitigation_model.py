import tensorflow as tf
from tensorflow import keras
import numpy as np

def build_comprehensive_bias_mitigation_model(text_shape, image_shape, demographic_shape):
    # Text input
    text_input = keras.Input(shape=text_shape, name="text_input")
    text_features = keras.layers.Dense(64, activation="relu")(text_input)
    text_features = keras.layers.Dense(32, activation="relu")(text_features)

    # Image input
    image_input = keras.Input(shape=image_shape, name="image_input")
    image_features = keras.layers.Conv2D(32, (3, 3), activation="relu")(image_input)
    image_features = keras.layers.MaxPooling2D((2, 2))(image_features)
    image_features = keras.layers.Flatten()(image_features)
    image_features = keras.layers.Dense(32, activation="relu")(image_features)

    # Demographic input
    demographic_input = keras.Input(shape=demographic_shape, name="demographic_input")
    demographic_features = keras.layers.Dense(16, activation="relu")(demographic_input)

    # Combine all features
    combined_features = keras.layers.concatenate([text_features, image_features, demographic_features])

    # Main output
    main_output = keras.layers.Dense(16, activation="relu")(combined_features)
    main_output = keras.layers.Dense(1, activation="sigmoid", name="main_output")(main_output)

    # Bias detection output
    bias_output = keras.layers.Dense(16, activation="relu")(combined_features)
    bias_output = keras.layers.Dense(1, activation="sigmoid", name="bias_output")(bias_output)

    model = keras.Model(
        inputs=[text_input, image_input, demographic_input],
        outputs=[main_output, bias_output]
    )

    model.compile(
        optimizer="adam",
        loss={"main_output": "binary_crossentropy", "bias_output": "binary_crossentropy"},
        loss_weights={"main_output": 1.0, "bias_output": 0.5},
        metrics={"main_output": "accuracy", "bias_output": "accuracy"}
    )

    return model

# Generate some dummy data for demonstration
text_shape = (100,)
image_shape = (50, 50, 3)
demographic_shape = (3,)

X_text = np.random.rand(1000, *text_shape)
X_image = np.random.rand(1000, *image_shape)
X_demographic = np.random.rand(1000, *demographic_shape)
y_main = np.random.randint(0, 2, (1000, 1))
y_bias = np.random.randint(0, 2, (1000, 1))

# Create and train the model
model = build_comprehensive_bias_mitigation_model(text_shape, image_shape, demographic_shape)
model.fit(
    {"text_input": X_text, "image_input": X_image, "demographic_input": X_demographic},
    {"main_output": y_main, "bias_output": y_bias},
    epochs=10,
    validation_split=0.2
)

# Save the model
model.save("comprehensive_bias_mitigation_model.h5")
print("Comprehensive bias mitigation model created and saved.")
