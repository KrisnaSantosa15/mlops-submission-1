import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

NUMERIC_FEATURES = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
CATEGORICAL_FEATURES = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
LABEL_KEY = "Obesity"

def transformed_name(key):
    return key + '_xf'

def tuner_fn(fn_args: FnArgs):
    """Build the tuner to find the best hyperparameters."""
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    hp = tf.keras.hp.HyperParameters()
    
    # Model architecture hyperparameters
    hp.Int('num_hidden_layers', 2, 4)
    hp.Int('embedding_dim', 8, 32, step=8)
    hp.Int('dense_units', 32, 256, step=32)
    hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    
    # Training hyperparameters
    hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    hp.Int('batch_size', 32, 128, step=32)
    
    tuner = tf.keras.tuners.Hyperband(
        hypermodel=lambda hp: model_builder(hp, tf_transform_output),
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory=fn_args.working_dir,
        project_name='obesity_tuning')
    
    return tuner

def model_builder(hp, tf_transform_output):
    """Build model with given hyperparameters."""
    inputs = []
    feature_layers = []

    # Numeric features
    for feature_name in NUMERIC_FEATURES:
        numeric_input = layers.Input(
            shape=(1,), name=transformed_name(feature_name))
        inputs.append(numeric_input)
        feature_layers.append(numeric_input)

    # Categorical features
    for feature_name in CATEGORICAL_FEATURES:
        vocab_size = tf_transform_output.vocabulary_size_by_name(feature_name)
        embedding_dim = hp.get('embedding_dim')
        categorical_input = layers.Input(
            shape=(1,), name=transformed_name(feature_name), dtype=tf.int64)
        inputs.append(categorical_input)
        embedding = layers.Embedding(vocab_size, embedding_dim)(categorical_input)
        embedding_flat = layers.Flatten()(embedding)
        feature_layers.append(embedding_flat)

    # Concatenate features
    features = layers.concatenate(feature_layers)

    # Hidden layers
    for i in range(hp.get('num_hidden_layers')):
        features = layers.Dense(hp.get('dense_units'), activation='relu')(features)
        features = layers.Dropout(hp.get('dropout_rate'))(features)

    outputs = layers.Dense(7, activation='softmax')(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model