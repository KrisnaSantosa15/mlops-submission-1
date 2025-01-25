import tensorflow as tf
import tensorflow_transform as tft

NUMERIC_FEATURES = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
CATEGORICAL_FEATURES = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
LABEL_KEY = "Obesity"

def transformed_name(key):
    return key + '_xf'

def preprocessing_fn(inputs):
    """Preprocess input features into transformed features."""
    outputs = {}
    
    # Scale numeric features
    for feature_name in NUMERIC_FEATURES:
        outputs[transformed_name(feature_name)] = tft.scale_to_z_score(
            inputs[feature_name])
    
    # Convert categorical features to indices
    for feature_name in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature_name)] = tft.compute_and_apply_vocabulary(
            inputs[feature_name], vocab_filename=feature_name)
    
    # Convert label to index
    outputs[transformed_name(LABEL_KEY)] = tft.compute_and_apply_vocabulary(
        inputs[LABEL_KEY], vocab_filename=LABEL_KEY)
    
    return outputs