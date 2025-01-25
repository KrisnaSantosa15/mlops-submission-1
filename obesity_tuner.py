import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
import keras_tuner as kt

NUMERIC_FEATURES = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
CATEGORICAL_FEATURES = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
LABEL_KEY = "Obesity"

def transformed_name(key):
    return key + '_xf'

def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Generates features and label for tuning/training."""
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        label_key=transformed_name(LABEL_KEY)
    )

    def split_features(features, label):
        """Split the features into individual tensors."""
        inputs = {}
        for feature_name in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            inputs[transformed_name(feature_name)] = features[transformed_name(feature_name)]
        return inputs, label

    dataset = dataset.map(split_features)

    # Remove .repeat() if it exists
    # dataset = dataset.repeat()  # Comment out or remove this line

    return dataset

def tuner_fn(fn_args: FnArgs):
    """Build the tuner to find the best hyperparameters."""
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    hp = kt.HyperParameters()
    
    # Model architecture hyperparameters
    hp.Int('num_hidden_layers', 2, 4)
    hp.Int('embedding_dim', 8, 32, step=8)
    hp.Int('dense_units', 32, 256, step=32)
    hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    
    # Training hyperparameters
    hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    hp.Int('batch_size', 32, 128, step=32)
    
    tuner = kt.Hyperband(
        hypermodel=lambda hp: model_builder(hp, tf_transform_output),
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory=fn_args.working_dir,
        project_name='obesity_tuning'
    )
    
    # Define fit_kwargs with steps_per_epoch
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=hp.get('batch_size'))
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=hp.get('batch_size'))

    # Calculate steps_per_epoch based on the dataset size
    steps_per_epoch = len(list(train_dataset))  # Adjust this based on your dataset size
    validation_steps = len(list(eval_dataset))  # Adjust this based on your dataset size

    fit_kwargs = {
        "x": train_dataset,
        "validation_data": eval_dataset,
        "epochs": 10,
        "steps_per_epoch": steps_per_epoch,  # Specify steps_per_epoch
        "validation_steps": validation_steps,  # Specify validation_steps
        "callbacks": [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)]
    }
    
    # Return an object with 'tuner' and 'fit_kwargs' attributes
    return type('TunerResult', (), {'tuner': tuner, 'fit_kwargs': fit_kwargs})

def model_builder(hp, tf_transform_output):
    """Build model with given hyperparameters."""
    inputs = []
    feature_layers = []

    
        # Model architecture hyperparameters
    hp.Int('num_hidden_layers', 2, 4)
    hp.Int('embedding_dim', 8, 32, step=8)
    hp.Int('dense_units', 32, 256, step=32)
    hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    
    # Training hyperparameters
    hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    hp.Int('batch_size', 32, 128, step=32)
    
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
