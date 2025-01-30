import tensorflow as tf
import tensorflow_transform as tft
from kerastuner.engine import base_tuner 
from tensorflow.keras import layers
from typing import NamedTuple, Dict, Text, Any 
from tfx.components.trainer.fn_args_utils import FnArgs
import kerastuner as kt

NUMERIC_FEATURES = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
CATEGORICAL_FEATURES = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
LABEL_KEY = "Obesity"

def transformed_name(key):
    return key + '_xf'

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                            ('fit_kwargs', Dict[Text,Any])])

def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern, tf_transform_output, batch_size=128):
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=1,
        shuffle=True,
        label_key=transformed_name(LABEL_KEY)
    )
    
    return dataset

def model_builder(hp, tf_transform_output):
    inputs = []
    encoded_features = []
    
    # Numeric features
    for feature_name in NUMERIC_FEATURES:
        input_layer = layers.Input(shape=(1,), name=transformed_name(feature_name))
        inputs.append(input_layer)
        norm_layer = layers.BatchNormalization()(input_layer)
        encoded_features.append(norm_layer)

    # Categorical features
    embedding_dim = hp.Int('embedding_dim', 8, 32, step=8)
    for feature_name in CATEGORICAL_FEATURES:
        vocab_size = tf_transform_output.vocabulary_size_by_name(feature_name)
        input_layer = layers.Input(shape=(1,), name=transformed_name(feature_name), dtype=tf.int64)
        inputs.append(input_layer)
        
        # Add handling for out-of-vocabulary indices
        safe_input = tf.where(
            tf.logical_or(input_layer < 0, input_layer >= vocab_size),
            tf.zeros_like(input_layer),
            input_layer
        )
        
        embedding = layers.Embedding(
            vocab_size,
            embedding_dim,
            mask_zero=True,
            name=f'embedding_{feature_name}'
        )(safe_input)
        
        embedding_flat = layers.Flatten()(embedding)
        encoded_features.append(embedding_flat)

    concat_features = layers.concatenate(encoded_features)
    
    num_hidden_layers = hp.Int('num_hidden_layers', 2, 4)
    for i in range(num_hidden_layers):
        units = hp.Int(f'units_{i}', 32, 256, step=32)
        dropout_rate = hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)
        concat_features = layers.Dense(units, activation='relu')(concat_features)
        concat_features = layers.BatchNormalization()(concat_features)
        concat_features = layers.Dropout(dropout_rate)(concat_features)

    outputs = layers.Dense(7, activation='softmax')(concat_features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    tuner = kt.Hyperband(
        hypermodel=lambda hp: model_builder(hp, tf_transform_output),
        objective='val_accuracy',
        max_epochs=30,
        factor=3,
        directory=fn_args.working_dir,
        project_name='obesity_tuning'
    )
    
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=3,
        restore_best_weights=True
    )
    
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'callbacks': [early_stopping]
        }
    )
