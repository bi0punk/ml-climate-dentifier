import numpy as np
import pytest

from training.model import (
    IMG_SIZE,
    NUM_TIME,
    NUM_WEATHER,
    compile_model,
    get_model,
    masked_categorical_crossentropy,
)


def _model():
    return get_model(backbone="cnn", input_shape=(IMG_SIZE, IMG_SIZE, 3))


def test_cnn_model_creation():
    model = _model()
    assert model is not None
    assert len(model.outputs) == 2
    # Keras 3 expone los nombres a través de output_names
    assert set(model.output_names) == {"time", "weather"}


def test_cnn_output_shapes():
    model = compile_model(_model())
    x = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    preds = model.predict(x, verbose=0)
    assert len(preds) == 2
    assert preds[0].shape == (1, NUM_TIME)
    assert preds[1].shape == (1, NUM_WEATHER)
    # Softmax outputs should sum to 1
    assert abs(float(preds[0].sum()) - 1.0) < 1e-5
    assert abs(float(preds[1].sum()) - 1.0) < 1e-5


def test_mobilenetv2_model():
    model = get_model(backbone="mobilenetv2")
    assert model is not None
    assert len(model.outputs) == 2
    assert set(model.output_names) == {"time", "weather"}


def test_single_output_time_names():
    # Verifica que el primer head sea 'time' y el segundo 'weather'
    model = _model()
    names = model.output_names
    assert names[0] == "time"
    assert names[1] == "weather"


@pytest.mark.parametrize("backbone", ["cnn", "mobilenetv2"])
def test_invalid_backbone_falls_back_to_cnn(backbone):
    assert get_model(backbone=backbone) is not None


def test_backbone_unknown_falls_back_to_cnn():
    model = get_model(backbone="unknown")
    assert model is not None
    assert len(model.outputs) == 2


def test_masked_categorical_crossentropy_single_label():
    import tensorflow as tf

    y_true = tf.constant([[1.0, 0.0, 0.0]])
    y_pred = tf.constant([[0.8, 0.1, 0.1]])
    loss = masked_categorical_crossentropy(y_true, y_pred)
    assert loss.shape == (1,)
    assert float(loss[0]) > 0


def test_masked_categorical_crossentropy_zeroed_label_leaves_loss():
    """Las etiquetas con fila de ceros (unknown) deben aportar pérdida de
    entropía cruzada sobre probabilidad uniforme, no NaN."""
    import tensorflow as tf

    y_true = tf.constant([[0.0, 0.0, 0.0]])
    y_pred = tf.constant([[0.8, 0.1, 0.1]])
    loss = masked_categorical_crossentropy(y_true, y_pred)
    assert not np.isnan(float(loss[0]))