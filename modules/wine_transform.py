"""TFX Transform module for the wine quality pipeline."""

import tensorflow as tf
import tensorflow_transform as tft


FEATURE_KEYS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]

LABEL_KEY = "good_quality"


def transformed_name(key: str) -> str:
    return f"{key}_xf"


def _fill_in_missing(value):
    if isinstance(value, tf.SparseTensor):
        default_value = "" if value.dtype == tf.string else 0
        dense_shape = [value.dense_shape[0], 1]
        value = tf.sparse.to_dense(
            tf.SparseTensor(value.indices, value.values, dense_shape),
            default_value,
        )
    return value


def preprocessing_fn(inputs):
    """Transform raw wine features into model-ready tensors."""
    outputs = {}

    for key in FEATURE_KEYS:
        feature = tf.cast(_fill_in_missing(inputs[key]), tf.float32)
        outputs[transformed_name(key)] = tft.scale_to_z_score(feature)

    label = tf.cast(_fill_in_missing(inputs[LABEL_KEY]), tf.int64)
    outputs[transformed_name(LABEL_KEY)] = label

    return outputs

