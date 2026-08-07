import pytest
import numpy as np

from bio_oncograph.research.histograph.features import (
    MorphologyFeatureExtractor,
    IntensityFeatureExtractor,
    TextureFeatureExtractor
)


def test_morphology_extractor_init() -> None:
    """Test MorphologyFeatureExtractor initialization."""
    extractor = MorphologyFeatureExtractor()
    assert extractor is not None


def test_morphology_extractor_compute() -> None:
    """Test MorphologyFeatureExtractor on synthetic mask."""
    extractor = MorphologyFeatureExtractor()
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    features = extractor.extract(mask)
    assert isinstance(features, np.ndarray)
    assert len(features) > 0


def test_morphology_extractor_empty_mask() -> None:
    """Test MorphologyFeatureExtractor on empty mask."""
    extractor = MorphologyFeatureExtractor()
    mask = np.zeros((10, 10), dtype=np.uint8)
    features = extractor.extract(mask)
    assert np.all(features == 0)


def test_morphology_extractor_invalid_shape() -> None:
    """Test MorphologyFeatureExtractor raises on invalid shape."""
    extractor = MorphologyFeatureExtractor()
    mask = np.zeros((10,), dtype=np.uint8)
    with pytest.raises(ValueError):
        extractor.extract(mask)


def test_intensity_extractor_init() -> None:
    """Test IntensityFeatureExtractor initialization."""
    extractor = IntensityFeatureExtractor()
    assert extractor is not None


def test_intensity_extractor_compute() -> None:
    """Test IntensityFeatureExtractor on synthetic image."""
    extractor = IntensityFeatureExtractor()
    img = np.ones((10, 10, 3), dtype=np.float32)
    mask = np.ones((10, 10), dtype=np.uint8)
    features = extractor.extract(img, mask)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] > 0


def test_intensity_extractor_empty_mask() -> None:
    """Test IntensityFeatureExtractor handles empty mask gracefully."""
    extractor = IntensityFeatureExtractor()
    img = np.ones((10, 10, 3), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=np.uint8)
    features = extractor.extract(img, mask)
    assert np.all(features == 0)


def test_intensity_extractor_shape_mismatch() -> None:
    """Test IntensityFeatureExtractor raises on shape mismatch."""
    extractor = IntensityFeatureExtractor()
    img = np.ones((10, 10, 3), dtype=np.float32)
    mask = np.ones((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        extractor.extract(img, mask)


def test_intensity_extractor_invalid_image_shape() -> None:
    """Test IntensityFeatureExtractor raises on invalid image shape."""
    extractor = IntensityFeatureExtractor()
    img = np.ones((10, 10), dtype=np.float32)
    mask = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        extractor.extract(img, mask)


def test_texture_extractor_init() -> None:
    """Test TextureFeatureExtractor initialization."""
    extractor = TextureFeatureExtractor()
    assert extractor is not None


def test_texture_extractor_compute() -> None:
    """Test TextureFeatureExtractor on synthetic image."""
    extractor = TextureFeatureExtractor()
    img = np.random.rand(20, 20).astype(np.float32)
    mask = np.ones((20, 20), dtype=np.uint8)
    features = extractor.extract(img, mask)
    assert isinstance(features, np.ndarray)


def test_texture_extractor_empty_mask() -> None:
    """Test TextureFeatureExtractor on empty mask."""
    extractor = TextureFeatureExtractor()
    img = np.random.rand(20, 20).astype(np.float32)
    mask = np.zeros((20, 20), dtype=np.uint8)
    features = extractor.extract(img, mask)
    assert np.all(features == 0)


def test_texture_extractor_shape_mismatch() -> None:
    """Test TextureFeatureExtractor raises on shape mismatch."""
    extractor = TextureFeatureExtractor()
    img = np.random.rand(20, 20).astype(np.float32)
    mask = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        extractor.extract(img, mask)


def test_all_extractors_deterministic() -> None:
    """Test feature extractors produce deterministic results."""
    m_ext = MorphologyFeatureExtractor()
    i_ext = IntensityFeatureExtractor()
    t_ext = TextureFeatureExtractor()

    np.random.seed(42)
    img = np.random.rand(15, 15, 3).astype(np.float32)
    mask = (np.random.rand(15, 15) > 0.5).astype(np.uint8)

    m1 = m_ext.extract(mask)
    m2 = m_ext.extract(mask)
    np.testing.assert_array_equal(m1, m2)

    i1 = i_ext.extract(img, mask)
    i2 = i_ext.extract(img, mask)
    np.testing.assert_array_equal(i1, i2)

    # Texture often expects 2D
    img_gray = img[:, :, 0]
    t1 = t_ext.extract(img_gray, mask)
    t2 = t_ext.extract(img_gray, mask)
    np.testing.assert_array_equal(t1, t2)


def test_feature_types() -> None:
    """Test output data types are consistently float32 or float64."""
    extractor = MorphologyFeatureExtractor()
    mask = np.ones((10, 10), dtype=np.uint8)
    features = extractor.extract(mask)
    assert features.dtype in (np.float32, np.float64)


def test_intensity_extractor_zero_variance_image() -> None:
    """Test IntensityFeatureExtractor handles flat/zero-variance images."""
    extractor = IntensityFeatureExtractor()
    img = np.zeros((10, 10, 3), dtype=np.float32)
    mask = np.ones((10, 10), dtype=np.uint8)
    features = extractor.extract(img, mask)
    assert not np.any(np.isnan(features))
