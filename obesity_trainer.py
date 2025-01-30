import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

NUMERIC_FEATURES = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
CATEGORICAL_FEATURES = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
LABEL_KEY = "Obesity"

def transformed_name(key):
    return key + '_xf'

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, batch_size=128)->tf.data.Dataset:
    """Creates an input function that loads and preprocesses the dataset."""
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=None,
        shuffle=True,
        shuffle_buffer_size=1000,
        label_key=transformed_name(LABEL_KEY))
    
    return dataset

def _get_serve_tf_examples_fn(model, tf_transform_output):

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):

        feature_spec = tf_transform_output.raw_feature_spec()

        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        # get predictions using the transformed features
        return model(transformed_features)

    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    """Train the model based on given args."""
    
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir, update_freq='batch'
    )
    
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Get dataset and compute steps per epoch
    batch_size = 128
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size)
    
    # Load best hyperparameters
    hp = fn_args.hyperparameters['values']
    
    # Build model with best hyperparameters
    model = build_model(hp, tf_transform_output)
    
    steps_per_epoch = 26
    validation_steps = 7
    
    # Train model
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=30,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )]
    )
    
    # Save model
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples'))
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

def build_model(hp, tf_transform_output):
    inputs = []
    encoded_features = []
    
    # Numeric features with BatchNormalization
    for feature_name in NUMERIC_FEATURES:
        input_layer = layers.Input(shape=(1,), name=transformed_name(feature_name))
        inputs.append(input_layer)
        norm_layer = layers.BatchNormalization()(input_layer)
        encoded_features.append(norm_layer)

    # Categorical features with embeddings
    for feature_name in CATEGORICAL_FEATURES:
        vocab_size = tf_transform_output.vocabulary_size_by_name(feature_name)
        input_layer = layers.Input(shape=(1,), name=transformed_name(feature_name), dtype=tf.int64)
        inputs.append(input_layer)
        
        safe_input = tf.where(
            tf.logical_or(input_layer < 0, input_layer >= vocab_size),
            tf.zeros_like(input_layer),
            input_layer
        )
        
        embedding = layers.Embedding(
            vocab_size,
            hp.get('embedding_dim'),
            mask_zero=True
        )(safe_input)
        
        embedding_flat = layers.Flatten()(embedding)
        encoded_features.append(embedding_flat)

    concat_features = layers.concatenate(encoded_features)
    
    # Dynamic hidden layers from tuner
    for i in range(hp.get('num_hidden_layers')):
        units = hp.get(f'units_{i}')
        dropout_rate = hp.get(f'dropout_{i}')
        concat_features = layers.Dense(units, activation='relu')(concat_features)  # Added activation
        concat_features = layers.BatchNormalization()(concat_features)
        concat_features = layers.Dropout(dropout_rate)(concat_features)

    outputs = layers.Dense(7, activation='softmax')(concat_features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    
    
    return model
