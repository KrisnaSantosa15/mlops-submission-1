import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

NUMERIC_FEATURES = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
CATEGORICAL_FEATURES = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
LABEL_KEY = "Obesity"

def transformed_name(key):
    return key + '_xf'

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, batch_size=64):
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY))
    return dataset

def run_fn(fn_args: FnArgs):
    """Train the model based on given args."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)
    
    # Load best hyperparameters
    hp = fn_args.hyperparameters['values']
    
    # Build model with best hyperparameters
    model = build_model(hp, tf_transform_output)
    
    # Train model
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=10,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3
        )]
    )
    
    # Save model
    model.save(fn_args.serving_model_dir, save_format='tf')

def build_model(hp, tf_transform_output):
    """Build the model using hyperparameters from tuning."""
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