import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=tf.TensorSpec(shape=(None, 784), dtype=tf.float32),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Generate some dummy data for demonstration
NUM_CLIENTS = 3
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100

def create_dummy_data():
    return tf.data.Dataset.from_tensor_slices((
        np.random.rand(100, 784).astype(np.float32),
        np.random.randint(0, 10, (100,))
    )).map(lambda x, y: (x, tf.cast(y, tf.int32)))

train_data = [create_dummy_data() for _ in range(NUM_CLIENTS)]

def preprocess(dataset):
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)

preprocessed_train_data = [preprocess(ds) for ds in train_data]

# Create and run the federated learning process
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0)
)

state = trainer.initialize()

for round_num in range(1, 11):
    result = trainer.next(state, preprocessed_train_data)
    state = result.state
    metrics = result.metrics
    print(f'Round {round_num}')
    print(f'Loss: {metrics["client_work"]["train"]["loss"]}')
    print(f'Accuracy: {metrics["client_work"]["train"]["sparse_categorical_accuracy"]}')

# Save the final global model
keras_model = create_keras_model()
tff.learning.models.assign_weights_to_keras_model(keras_model, state.global_model_weights)
keras_model.save('federated_learning_privacy_preserving_model.h5')
print("Federated learning privacy-preserving model created and saved.")
