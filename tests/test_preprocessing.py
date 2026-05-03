"""Tests for preprocess_image."""

import pytest

from vlm_grader.preprocessing import PreprocessConfig, preprocess_image


def test_no_op_when_pillow_unavailable_or_input_passthrough():
    """When Pillow isn't installed (or input already an Image), call is safe."""
    try:
        from PIL import Image
    except ImportError:
        # Pillow not available — preprocess_image must be a no-op.
        result = preprocess_image("not-an-image-and-not-checked")
        assert result == "not-an-image-and-not-checked"
        return

    # Pillow available: should validate type and resize.
    img = Image.new("RGB", (100, 200), color="red")
    out = preprocess_image(img, PreprocessConfig(target_size=(64, 64)))
    assert out.size == (64, 64)


def test_preprocess_rejects_non_pil_when_pil_available():
    try:
        from PIL import Image  # noqa
    except ImportError:
        pytest.skip("Pillow not installed")
    with pytest.raises(TypeError):
        preprocess_image("not an image")


def test_letterbox_pads_to_target():
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    img = Image.new("RGB", (200, 100), color="red")
    out = preprocess_image(img, PreprocessConfig(target_size=(128, 128), keep_aspect_ratio=True))
    assert out.size == (128, 128)


def test_no_aspect_ratio_resize():
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    img = Image.new("RGB", (200, 100), color="red")
    out = preprocess_image(img, PreprocessConfig(target_size=(64, 64), keep_aspect_ratio=False))
    assert out.size == (64, 64)


def test_default_config():
    """Default config sets target_size=(448, 448)."""
    cfg = PreprocessConfig()
    assert cfg.target_size == (448, 448)
    assert cfg.convert_mode == "RGB"
    assert cfg.keep_aspect_ratio is True
