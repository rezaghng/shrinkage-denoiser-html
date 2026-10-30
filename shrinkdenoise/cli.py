"""Command line:  python -m shrinkdenoise input.png -o out.png --noise 20 --base C --estimator E6"""
import argparse, sys, time
import numpy as np
from PIL import Image
from . import denoise, classical, add_salt_pepper, psnr, to_gray, noise_fraction, ESTIMATORS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shrinkdenoise", description="Salt-and-pepper denoising with shrinkage estimators E1-E6.")
    ap.add_argument("input"); ap.add_argument("-o", "--output", default="denoised.png")
    ap.add_argument("--noise", type=float, default=None, metavar="PERCENT",
                    help="add salt-and-pepper noise of this percentage first; omit if the input is already noisy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", choices=["M", "C"], default="M", help="M: median initial estimate, C: CFM initial estimate")
    ap.add_argument("--estimator", choices=list(ESTIMATORS), default="E6")
    ap.add_argument("--gray", action="store_true", help="convert colour input to gray (the setting of the paper)")
    ap.add_argument("--save-noisy", default=None, help="also save the noisy image here")
    a = ap.parse_args(argv)
    img = np.array(Image.open(a.input).convert("RGB"))
    if a.gray:
        img = to_gray(img)
    clean = img
    if a.noise is not None:
        noisy = add_salt_pepper(img, a.noise / 100.0, a.seed)
    else:
        noisy = img
    print(f"image {noisy.shape}, pixels at 0/255: {100 * noise_fraction(noisy):.1f}%")
    if a.save_noisy:
        Image.fromarray(noisy).save(a.save_noisy)
    t = time.time()
    out = denoise(noisy, a.base, a.estimator, progress=lambda p: print(f"\r{100 * p:5.1f}%", end="", file=sys.stderr))
    print(f"\n{a.estimator}-{a.base} done in {time.time() - t:.1f}s")
    Image.fromarray(out).save(a.output)
    if a.noise is not None:
        gray_ok = noisy.ndim == 2
        print(f"PSNR noisy {psnr(clean, noisy):.2f} dB -> denoised {psnr(clean, out):.2f} dB")
        if gray_ok:
            print(f"PSNR classical ({'median' if a.base == 'M' else 'CFM'} E0): {psnr(clean, classical(noisy, a.base)):.2f} dB")
    print("saved", a.output)


if __name__ == "__main__":
    main()
