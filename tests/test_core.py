import numpy as np
import shrinkdenoise as sd


def _image(h=90, w=100):
    y, x = np.mgrid[0:h, 0:w]
    return np.clip(60 + 1.2 * x + 0.8 * y + 25 * np.sin(x / 6.0), 1, 254).astype(np.uint8)   # smooth, no 0/255


def test_noise_density():
    I = _image(); N = sd.add_salt_pepper(I, 0.3, 0)
    assert abs(sd.noise_fraction(N) - 0.3) < 0.03


def test_clean_pixels_unchanged_and_improvement():
    I = _image(); N = sd.add_salt_pepper(I, 0.3, 1)
    clean = (N != 0) & (N != 255)
    for base in "MC":
        for est in ("E1", "E6"):
            O = sd.denoise(N, base, est)
            assert O.dtype == np.uint8 and O.shape == I.shape
            assert (O[clean] == N[clean]).all()                      # switching filter: clean pixels untouched
            assert sd.psnr(I, O) > sd.psnr(I, sd.classical(N, base)) - 0.5
            assert sd.psnr(I, O) > sd.psnr(I, N) + 10


def test_tiling_is_exact():
    I = _image(); N = sd.add_salt_pepper(I, 0.2, 2)
    assert (sd.denoise(N, "M", "E6", tile=1000) == sd.denoise(N, "M", "E6", tile=40)).all()


def test_colour_and_progress():
    I = np.stack([_image(), _image()[::-1], _image()[:, ::-1]], -1)
    N = sd.add_salt_pepper(I, 0.2, 3); seen = []
    O = sd.denoise(N, "C", "E6", progress=seen.append)
    assert O.shape == I.shape and seen and abs(seen[-1] - 1.0) < 1e-9


def test_all_estimators_run():
    N = sd.add_salt_pepper(_image(), 0.2, 4)
    for e in sd.ESTIMATORS:
        assert sd.denoise(N, "M", e).shape == N.shape


def test_no_crash_on_saturated_areas_and_high_noise():
    # photos with pure white/black regions + heavy noise give windows without any usable data (singular systems)
    rng = np.random.default_rng(0)
    I = np.full((120, 120), 128, np.uint8); I[:60, :] = 255; I[:, :25] = 0
    I[40:100, 40:100] = rng.integers(40, 200, (60, 60)).astype(np.uint8)
    N = sd.add_salt_pepper(I, 0.7, 0)
    for base in "MC":
        for est in sd.ESTIMATORS:
            assert sd.denoise(N, base, est).shape == I.shape
