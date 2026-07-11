import pytest

try:
    from training.model import get_model, compile_model, IMG_SIZE, NUM_TIME, NUM_WEATHER
    HAS_TF = True
except ImportError:
    HAS_TF = False


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not available")
def test_cnn_model_creation():
    model = get_model(backbone="cnn", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    assert model is not None
    assert len(model.outputs) == 2
    # Output names should be 'time' and 'weather'
    output_names = [o.name for o in model.outputs]
    assert any("time" in n for n in output_names)
    assert any("weather" in n for n in output_names)


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not available")
def test_cnn_output_shapes():
    model = get_model(backbone="cnn")
    model = compile_model(model)
    import numpy as np
    x = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    preds = model.predict(x, verbose=0)
    assert len(preds) == 2
    assert preds[0].shape == (1, NUM_TIME)
    assert preds[1].shape == (1, NUM_WEATHER)
    # Softmax outputs should sum to 1
    assert abs(float(preds[0].sum()) - 1.0) < 1e-5
    assert abs(float(preds[1].sum()) - 1.0) < 1e-5


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow not available")
def test_mobilenetv2_model():
    model = get_model(backbone="mobilenetv2")
    assert model is not None
    assert len(model.outputs) == 2
