"""Tkinter GUI: open an image, choose the noise percentage, denoise with the shrinkage estimators of the paper."""
from __future__ import annotations
import os, queue, threading, time, traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

from . import core

APP_TITLE = "Shrinkage Denoiser - salt-and-pepper noise removal"
EST_LABELS = {"E1": "E1 - least squares (no shrinkage)", "E2": "E2 - James-Stein", "E3": "E3 - grouped James-Stein",
              "E4": "E4 - ridge, fixed penalty", "E5": "E5 - empirical-Bayes ridge (k=1)",
              "E6": "E6 - empirical-Bayes ridge (k=3), best"}


class Panel(ttk.Frame):
    """A titled image view that rescales its image to the available space."""
    def __init__(self, master, title):
        super().__init__(master, padding=2)
        self.title = ttk.Label(self, text=title, font=("TkDefaultFont", 10, "bold"), anchor="center")
        self.title.pack(fill="x")
        self.caption = ttk.Label(self, text="", anchor="center", foreground="#444")
        self.caption.pack(fill="x")
        self.view = tk.Label(self, bg="#202020")
        self.view.pack(fill="both", expand=True)
        self._img = None; self._tk = None; self._after = None
        self.view.bind("<Configure>", lambda e: self._schedule())

    def set(self, img: np.ndarray | None, caption: str = ""):
        self._img = None if img is None else Image.fromarray(img)
        self.caption.config(text=caption)
        self._render()

    def _schedule(self):
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(60, self._render)

    def _render(self):
        self._after = None
        if self._img is None:
            self.view.config(image="", text="")
            return
        w, h = max(self.view.winfo_width(), 50), max(self.view.winfo_height(), 50)
        s = min(w / self._img.width, h / self._img.height)
        size = (max(int(self._img.width * s), 1), max(int(self._img.height * s), 1))
        res = Image.NEAREST if s >= 1 else Image.LANCZOS
        self._tk = ImageTk.PhotoImage(self._img.resize(size, res))
        self.view.config(image=self._tk)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x700"); self.minsize(900, 640)
        self.src = None            # uint8 array as loaded (RGB or gray)
        self.src_path = None
        self.noisy = None; self.result = None; self.clean_ref = None
        self.q = queue.Queue(); self.busy = False
        self._build()
        self.after(100, self._poll)

    # ---------------------------------------------------------------- layout
    def _build(self):
        side = ttk.Frame(self, padding=8); side.pack(side="left", fill="y")
        main = ttk.Frame(self, padding=4); main.pack(side="left", fill="both", expand=True)

        f = ttk.LabelFrame(side, text="1. Image", padding=6); f.pack(fill="x", pady=3)
        ttk.Button(f, text="Open image...", command=self.open_image).pack(fill="x")
        self.lbl_file = ttk.Label(f, text="no image loaded", wraplength=250, foreground="#555"); self.lbl_file.pack(fill="x", pady=(4, 0))

        f = ttk.LabelFrame(side, text="2. Noise", padding=6); f.pack(fill="x", pady=3)
        self.noise_mode = tk.StringVar(value="add")
        ttk.Radiobutton(f, text="Add salt-and-pepper noise to this image", variable=self.noise_mode, value="add",
                        command=self._sync).pack(anchor="w")
        row = ttk.Frame(f); row.pack(fill="x", padx=18, pady=2)
        self.density = tk.DoubleVar(value=20.0)
        self.scale = ttk.Scale(row, from_=1, to=90, variable=self.density, command=lambda v: self._on_scale())
        self.scale.pack(side="left", fill="x", expand=True)
        self.spin = ttk.Spinbox(row, from_=1, to=90, width=5, increment=1, command=self._on_spin)
        self.spin.set("20"); self.spin.pack(side="left", padx=(6, 0)); ttk.Label(row, text="%").pack(side="left")
        self.spin.bind("<Return>", lambda e: self._on_spin()); self.spin.bind("<FocusOut>", lambda e: self._on_spin())
        row2 = ttk.Frame(f); row2.pack(fill="x", padx=18)
        ttk.Label(row2, text="random seed").pack(side="left")
        self.seed = ttk.Spinbox(row2, from_=0, to=99999, width=7); self.seed.set("0"); self.seed.pack(side="left", padx=6)
        ttk.Radiobutton(f, text="The image is already noisy (0/255 impulses)", variable=self.noise_mode, value="given",
                        command=self._sync).pack(anchor="w", pady=(6, 0))

        f = ttk.LabelFrame(side, text="3. Method", padding=6); f.pack(fill="x", pady=3)
        ttk.Label(f, text="Initial estimate E0 (classical algorithm):").pack(anchor="w")
        self.base = tk.StringVar(value="M")
        ttk.Radiobutton(f, text="Median algorithm (base M)", variable=self.base, value="M").pack(anchor="w", padx=12)
        ttk.Radiobutton(f, text="CFM algorithm (base C)", variable=self.base, value="C").pack(anchor="w", padx=12)
        ttk.Label(f, text="Shrinkage estimator:").pack(anchor="w", pady=(6, 0))
        self.est = tk.StringVar(value=EST_LABELS["E6"])
        ttk.Combobox(f, textvariable=self.est, values=list(EST_LABELS.values()), state="readonly", width=34).pack(fill="x")
        self.gray = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Convert to gray (setting of the paper)", variable=self.gray).pack(anchor="w", pady=(6, 0))
        ttk.Label(f, text="Unchecked: colour images are denoised\nchannel by channel (not evaluated in the paper).",
                  foreground="#666", font=("TkDefaultFont", 8)).pack(anchor="w", padx=18)

        f = ttk.Frame(side); f.pack(fill="x", pady=8)
        self.btn_run = ttk.Button(f, text="Denoise", command=self.run); self.btn_run.pack(fill="x", ipady=6)
        self.btn_save = ttk.Button(f, text="Save denoised image...", command=lambda: self.save(self.result), state="disabled")
        self.btn_save.pack(fill="x", pady=(6, 0))
        self.btn_savn = ttk.Button(f, text="Save noisy image...", command=lambda: self.save(self.noisy), state="disabled")
        self.btn_savn.pack(fill="x", pady=(3, 0))
        self.prog = ttk.Progressbar(side, mode="determinate", maximum=100); self.prog.pack(fill="x")
        self.status = ttk.Label(side, text="Open an image to start.", wraplength=250, foreground="#333"); self.status.pack(fill="x", pady=4)

        main.columnconfigure((0, 1, 2), weight=1, uniform="p"); main.rowconfigure(0, weight=1)
        self.p_noisy = Panel(main, "Noisy input"); self.p_class = Panel(main, "Classical algorithm (E0)")
        self.p_res = Panel(main, "Shrinkage estimator")
        for i, p in enumerate((self.p_noisy, self.p_class, self.p_res)):
            p.grid(row=0, column=i, sticky="nsew", padx=3)
        self._sync()

    # --------------------------------------------------------------- helpers
    def _on_scale(self):
        self.spin.set(str(int(round(self.density.get()))))

    def _on_spin(self):
        try:
            v = min(max(float(self.spin.get()), 1.0), 90.0)
        except ValueError:
            v = 20.0
        self.density.set(v); self.spin.set(f"{v:g}")

    def _sync(self):
        st = "normal" if self.noise_mode.get() == "add" else "disabled"
        self.scale.config(state=st); self.spin.config(state=st); self.seed.config(state=st)

    def set_status(self, text):
        self.status.config(text=text)

    def open_image(self):
        path = filedialog.askopenfilename(title="Open image", filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")])
        if path:
            self.load(path)

    def load(self, path):
        try:
            im = Image.open(path)
            im = im.convert("L") if im.mode in ("L", "1", "I;16", "I", "F") else im.convert("RGB")
            arr = np.array(im, dtype=np.uint8)
        except Exception as e:
            messagebox.showerror("Cannot open image", str(e)); return
        self.src, self.src_path = arr, path
        self.noisy = self.result = None
        mp = arr.shape[0] * arr.shape[1] / 1e6
        self.lbl_file.config(text=f"{os.path.basename(path)}\n{arr.shape[1]} x {arr.shape[0]} px ({mp:.2f} MP)")
        self.p_noisy.set(arr, "original"); self.p_class.set(None); self.p_res.set(None)
        self.btn_save.config(state="disabled"); self.btn_savn.config(state="disabled")
        self.set_status(f"Loaded. Estimated time about {max(mp * 16, 1):.0f} s (+ about 1 min compilation on the very first run).")

    def save(self, arr):
        if arr is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif")])
        if path:
            Image.fromarray(arr).save(path); self.set_status(f"Saved {path}")

    # ------------------------------------------------------------------- run
    def run(self):
        if self.busy:
            return
        if self.src is None:
            messagebox.showinfo("No image", "Please open an image first."); return
        est = next(k for k, v in EST_LABELS.items() if v == self.est.get())
        base = self.base.get(); add = self.noise_mode.get() == "add"
        try:
            rho = float(self.spin.get()) / 100.0; seed = int(float(self.seed.get()))
        except ValueError:
            messagebox.showerror("Invalid value", "Noise percentage and seed must be numbers."); return
        img = self.src
        if self.gray.get() and img.ndim == 3:
            img = core.to_gray(img)
        self.busy = True; self.btn_run.config(state="disabled"); self.prog["value"] = 0
        self.set_status("Working... (first run compiles the numba kernels, please wait)")
        threading.Thread(target=self._worker, args=(img, add, rho, seed, base, est), daemon=True).start()

    def _worker(self, img, add, rho, seed, base, est):
        try:
            t0 = time.time()
            noisy = core.add_salt_pepper(img, rho, seed) if add else img
            ref = img if add else None
            self.q.put(("noisy", noisy, ref))
            cls = self._classical(noisy, base)
            self.q.put(("classical", cls, ref))
            out = core.denoise(noisy, base, est, progress=lambda p: self.q.put(("progress", p)))
            self.q.put(("done", out, ref, time.time() - t0, est, base))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    @staticmethod
    def _classical(noisy, base):
        if noisy.ndim == 2:
            return core.classical(noisy, base)
        return np.stack([core.classical(np.ascontiguousarray(noisy[..., c]), base) for c in range(noisy.shape[2])], -1)

    def _cap(self, arr, ref, label=""):
        s = label
        if ref is not None:
            s += f"PSNR {core.psnr(ref, arr):.2f} dB"
        return s

    def _poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                k = m[0]
                if k == "progress":
                    self.prog["value"] = 100 * m[1]
                elif k == "noisy":
                    _, arr, ref = m; self.noisy = arr
                    frac = core.noise_fraction(arr)
                    self.p_noisy.set(arr, self._cap(arr, ref, f"{100 * frac:.1f}% pixels at 0/255   "))
                    self.btn_savn.config(state="normal")
                    if frac < 0.005:
                        self.set_status("Warning: almost no 0/255 pixels found - the image seems noise-free.")
                elif k == "classical":
                    _, arr, ref = m
                    self.p_class.set(arr, self._cap(arr, ref))
                elif k == "done":
                    _, arr, ref, dt, est, base = m; self.result = arr
                    self.p_res.title.config(text=f"{est}-{base} ({core.ESTIMATORS[est][0]})")
                    self.p_class.title.config(text=f"Classical {'median' if base == 'M' else 'CFM'} (E0 of base {base})")
                    self.p_res.set(arr, self._cap(arr, ref))
                    self.btn_save.config(state="normal"); self.btn_run.config(state="normal"); self.busy = False
                    self.prog["value"] = 100
                    extra = ""
                    if ref is not None:
                        extra = f"  Gain over the classical E0: {core.psnr(ref, arr) - core.psnr(ref, self._classical_cache(m)):+.2f} dB."
                    self.set_status(f"Done in {dt:.1f} s.{extra}")
                elif k == "error":
                    self.busy = False; self.btn_run.config(state="normal")
                    self.set_status("Error - see message."); messagebox.showerror("Error", m[1])
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _classical_cache(self, _m):
        # the classical image currently shown in the middle panel
        return np.array(self.p_class._img)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
