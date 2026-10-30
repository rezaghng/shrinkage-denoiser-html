"""Entry point of the desktop app (also the PyInstaller entry point)."""
import os, sys

if getattr(sys, "frozen", False):          # installed .exe: keep the numba cache in a writable per-user folder
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(base, "ShrinkageDenoiser", "numba_cache"))


def selftest(path):
    import numpy as np
    import shrinkdenoise as sd
    y, x = np.mgrid[0:60, 0:60]
    img = np.clip(80 + 2 * x + y, 1, 254).astype(np.uint8)
    noisy = sd.add_salt_pepper(img, 0.3, 0)
    out = sd.denoise(noisy, "C", "E6")
    ok = sd.psnr(img, out) > sd.psnr(img, noisy) + 10
    with open(path, "w") as f:
        f.write("selftest ok" if ok else "selftest FAILED")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(sys.argv[sys.argv.index("--selftest") + 1])
    else:
        from shrinkdenoise.gui import main
        main()
