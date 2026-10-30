"""shrinkdenoise: shrinkage estimators E1-E6 for salt-and-pepper noise removal (median and CFM bases)."""
from .core import (denoise, classical, add_salt_pepper, psnr, to_gray, noise_fraction,
                   ESTIMATORS, BASES)
__all__ = ["denoise", "classical", "add_salt_pepper", "psnr", "to_gray", "noise_fraction", "ESTIMATORS", "BASES"]
__version__ = "1.0.0"
