

IMAGE_PATH = None            # <<< PUT THE PATH TO YOUR CLEAN IMAGE HERE, e.g. r"C:\images\lena.png"
OUTPUT_DIR = "results"       # figures, images and CSV are saved here
MAX_SIDE = 1024              # larger images are down-scaled (keeps FFT fast); None = no resize
SEED = 42                    # fixed seed -> reproducible noise
SHOW_PLOTS = True            # False = only save figures

# --- noise generator settings -----------------------------------------
GAUSSIAN_VARIANCE = 0.01     # variance on the [0,1] scale (sigma = 0.1)
SALT_PROB = 0.05             # probability that a pixel becomes white
PEPPER_PROB = 0.05           # probability that a pixel becomes black
PERIODIC_PARAMS = dict(frequency=30.0,       # cycles per image along the wave direction
                       orientation_deg=30.0, # direction of the wave vector
                       amplitude=0.15,       # peak amplitude on the [0,1] scale
                       phase_deg=0.0)

# --- Wiener (Gaussian-noise) filter settings -----------------------------------
WIENER_COLOR_SPACE = "YCrCb"   # "YCrCb" (recommended) or "RGB" (independent R,G,B channels)
WIENER_WINDOW = 5              # local window for R,G,B / luminance
WIENER_CHROMA_WINDOW = 7       # local window for the chroma channels (YCrCb mode)
WIENER_REFINE = True           # second, pilot-guided Wiener pass (empirical Wiener)

# --- enhancement settings ----------------------------------------------
CONTRAST_PERCENTILES = (1.0, 99.0)   # lower / upper percentile for stretching
CONTRAST_MAX_GAIN = 1.1              # limits amplification of contrast
CONTRAST_SKIP_ABOVE = 0.6            # image already spans > this range (1-99 %) -> no stretching
SHARPEN_STRENGTH = 0.6               # Laplacian sharpening strength (0 = off)

# --- detector thresholds (see detect_noise_type for the meaning) ---------
T_IMPULSE = 0.008      # min fraction of isolated extreme pixels -> salt & pepper
T_PERIODIC_PROM = 2.0  # min log-magnitude prominence (ln units) of a spectral peak
T_GAUSS_SIGMA = 0.03   # min estimated noise sigma (on [0,1] scale) -> Gaussian
T_IMPULSE_VS_NOISE = 0.4  # impulses must be >= this * (robust sigma): strong Gaussian noise
                           # that is merely CLIPPED at 0/255 also creates a few extreme pixels
T_AMBIGUITY = 0.2      # winner must beat runner-up by this relative margin

# --- repeated-trial classification test (confusion matrix) ---------------
N_TRIALS_PER_CLASS = 10   # 0 disables the test

# ======================================================================
import os
import sys
import csv
import time
import warnings

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage, signal
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from skimage.metrics import (mean_squared_error,
                             peak_signal_noise_ratio,
                             structural_similarity)

LABELS = ["Gaussian", "Salt-and-Pepper", "Periodic", "Clean/Unknown"]
FILTER_FOR = {"Gaussian": "Wiener filter",
              "Salt-and-Pepper": "Median filter",
              "Periodic": "Notch-reject filter (FFT)",
              "Clean/Unknown": "Fallback: mild bilateral filter"}
STAGES = ["Noisy", "Denoised", "Enhanced"]


# ======================================================================
# 1. IMAGE INPUT
# ======================================================================
def get_image_path():
    """Return the image path: config variable -> CLI argument -> upload/file dialog."""
    if IMAGE_PATH:
        return IMAGE_PATH
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return sys.argv[1]
    try:  # Google Colab upload
        from google.colab import files  # type: ignore
        up = files.upload()
        return next(iter(up.keys()))
    except Exception:
        pass
    try:  # desktop file chooser
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select the original (noise-free) color image",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
        root.destroy()
        if path:
            return path
    except Exception:
        pass
    return input("Enter the path of the clean image: ").strip().strip('"')


def _quantize(img):
    """Clip to [0,1] and round to 8-bit levels (a real image is stored in 8 bits)."""
    return np.round(np.clip(img, 0.0, 1.0) * 255.0) / 255.0


def load_original_image(path, max_side=MAX_SIDE):
    """
    Read JPG/JPEG/PNG, validate it, and return a float64 RGB array in [0,1], shape (H,W,3).
    Grayscale and RGBA inputs are converted to RGB with a warning.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path!r}")
    if os.path.splitext(path)[1].lower() not in (".jpg", ".jpeg", ".png"):
        warnings.warn("Unusual file extension; trying to read it anyway.")
    try:
        pil = Image.open(path)
        pil.load()
    except Exception as exc:
        raise ValueError(f"Cannot read '{path}' as an image: {exc}") from exc

    if pil.mode in ("L", "LA", "1", "I", "I;16", "F"):
        warnings.warn("Grayscale image detected - replicating it to 3 channels.")
    if pil.mode in ("RGBA", "LA", "P"):
        pil = pil.convert("RGBA")
        bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(bg, pil)
    if pil.mode.startswith("I") or pil.mode == "F":
        arr = np.asarray(pil, dtype=np.float64)
        arr = (arr - arr.min()) / max(np.ptp(arr), 1e-12) * 255.0
        pil = Image.fromarray(arr.astype(np.uint8))
    rgb = np.asarray(pil.convert("RGB"))

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {rgb.shape}")
    h, w = rgb.shape[:2]
    if min(h, w) < 32:
        raise ValueError(f"Image too small ({w}x{h}); need at least 32x32 pixels.")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)

    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        rgb = cv2.resize(rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        print(f"[info] Image resized from {w}x{h} to {rgb.shape[1]}x{rgb.shape[0]} for speed.")

    img = rgb.astype(np.float64) / 255.0          # normalisation to [0,1]
    if not np.isfinite(img).all():
        raise ValueError("Image contains NaN/Inf values.")
    return img


# ======================================================================
# 2. NOISE GENERATION  (each from the same clean image, never combined)
# ======================================================================
def add_gaussian_noise(img, variance=GAUSSIAN_VARIANCE, rng=None):
    """Zero-mean additive Gaussian noi  se; `variance` is on the [0,1] intensity scale."""
    rng = rng or np.random.default_rng(SEED)
    noise = rng.normal(0.0, np.sqrt(variance), img.shape)
    return _quantize(img + noise)


def add_salt_pepper_noise(img, salt_prob=SALT_PROB, pepper_prob=PEPPER_PROB, rng=None,
                          per_channel=False):
    """
    Random white (salt) and black (pepper) pixels.
    Total density = salt_prob + pepper_prob. per_channel=False -> whole pixel turns white/black.
    """
    if salt_prob < 0 or pepper_prob < 0 or salt_prob + pepper_prob > 1:
        raise ValueError("salt_prob and pepper_prob must be >= 0 and sum to <= 1.")
    rng = rng or np.random.default_rng(SEED)
    shape = img.shape if per_channel else img.shape[:2]
    r = rng.random(shape)
    salt = r < salt_prob
    pepper = (r >= salt_prob) & (r < salt_prob + pepper_prob)
    if not per_channel:
        salt, pepper = salt[..., None], pepper[..., None]
    out = img.copy()
    out = np.where(salt, 1.0, out)
    out = np.where(pepper, 0.0, out)
    return _quantize(out)


def add_periodic_noise(img, frequency=30.0, orientation_deg=30.0, amplitude=0.15, phase_deg=0.0):
    """
    2-D sinusoidal interference:  n(x,y) = A sin(2*pi*(fx*x/W + fy*y/H) + phase)
    with fx = f cos(theta), fy = f sin(theta) (cycles per image).
    It produces a symmetric pair of impulses at (+-fx, +-fy) in the 2-D FFT.
    """
    h, w = img.shape[:2]
    if frequency >= min(h, w) / 2:
        warnings.warn("Frequency is at/above the Nyquist limit and will alias.")
    th, ph = np.deg2rad(orientation_deg), np.deg2rad(phase_deg)
    fx, fy = frequency * np.cos(th), frequency * np.sin(th)
    y, x = np.mgrid[0:h, 0:w]
    pattern = amplitude * np.sin(2 * np.pi * (fx * x / w + fy * y / h) + ph)
    return _quantize(img + pattern[..., None])


# ======================================================================
# 3. AUTOMATIC NOISE DETECTION  (hybrid: spatial statistics + FFT)
# ======================================================================
def _gray(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def _impulse_features(img):
    """
    Spatial-domain impulse test, per colour channel (8-bit):
      * extreme pixel : value <= 5 or >= 250
      * isolated      : differs from the 3x3 median of its neighbourhood by > 0.3
    A genuine black/white region has extreme pixels whose neighbours are also extreme,
    so they are NOT isolated and are not counted. This is the key to avoiding false alarms.
    """
    u8 = np.round(img * 255).astype(np.uint8)
    iso_fracs, ext_fracs, salt, pepper = [], [], 0, 0
    for c in range(3):
        ch = u8[..., c]
        med = cv2.medianBlur(ch, 3)
        hi, lo = ch >= 250, ch <= 5
        extreme = hi | lo
        isolated = extreme & (np.abs(ch.astype(np.int16) - med.astype(np.int16)) > 0.3 * 255)
        iso_fracs.append(isolated.mean())
        ext_fracs.append(extreme.mean())
        salt += (isolated & hi).sum()
        pepper += (isolated & lo).sum()
    iso, ext = float(np.mean(iso_fracs)), float(np.mean(ext_fracs))
    return dict(impulse_ratio=iso, extreme_ratio=ext,
                purity=iso / ext if ext > 0 else 0.0,
                salt_share=salt / max(salt + pepper, 1))


def _estimate_sigma(gray):
    """
    Noise standard deviation (on [0,1] scale) from two edge-robust estimators:
      (a) Immerkaer's Laplacian-difference mask,
      (b) Donoho's MAD of the Haar diagonal detail band.
    The smaller one is used (conservative: image texture can only inflate both).
    Also returns the excess kurtosis of the Haar detail band (Gaussian noise ~ small,
    impulse noise ~ very large).
    """
    h, w = gray.shape
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray, cv2.CV_64F, k, borderType=cv2.BORDER_REFLECT_101)[1:-1, 1:-1]
    s_imm = np.sqrt(np.pi / 2) / 6.0 * np.mean(np.abs(conv))
    g = gray[: h // 2 * 2, : w // 2 * 2]
    hh = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2.0
    s_mad = np.median(np.abs(hh)) / 0.6745
    return float(min(s_imm, s_mad)), float(s_imm), float(s_mad), float(kurtosis(hh.ravel()))


def log_spectrum(gray, window=False):
    """Centred log-magnitude spectrum log(1+|F|) (zero frequency in the middle)."""
    g = gray - gray.mean()
    if window:
        g = g * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    return np.log1p(np.abs(fftshift(fft2(g))))


def _periodic_features(gray, prom_thresh=T_PERIODIC_PROM, bg_size=15, min_radius=6,
                       max_pairs=8, max_candidates=600, texture_pairs=12):
    """
    Frequency-domain test.
      1. Hann-window the image (reduces leakage / border streaks), 2-D FFT, fftshift.
      2. Prominence = log|F| - median-filtered log|F|  (local background removal).
      3. Candidate peaks: prominence > threshold, local maxima, away from the DC centre.
      4. Keep only candidates that (a) have a symmetric partner at (-u,-v) and
         (b) are compact spikes (not points on a long streak caused by an image edge).
    Acceptance (dominance test): the strongest peaks must be few (<= 8 within a factor ~2 of the top);
    many comparable peaks => texture/pattern in the real scene, not interference -> rejected.
    """
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = fftshift(fft2((gray - gray.mean()) * win))
    lm = np.log(np.abs(F) + 1e-6)
    prom = lm - ndimage.median_filter(lm, size=bg_size, mode="reflect")
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rad = np.hypot(yy - cy, xx - cx)
    cand = (prom > prom_thresh) & (rad > min_radius) & (lm == ndimage.maximum_filter(lm, size=5))
    ys, xs = np.nonzero(cand)
    n_cand = len(ys)
    if n_cand > max_candidates:                      # keep only the strongest candidates
        order = np.argsort(prom[ys, xs])[::-1][:max_candidates]
        ys, xs = ys[order], xs[order]
    pairs = []
    if True:
        for y, x in zip(ys, xs):
            dy, dx = y - cy, x - cx
            if not (dy > 0 or (dy == 0 and dx > 0)):
                continue                                   # handle each pair once
            py, px = cy - dy, cx - dx
            if not (0 <= py < h and 0 <= px < w):
                continue
            partner = prom[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2].max()
            if partner < 0.7 * prom_thresh:
                continue                                   # no symmetric counterpart
            y0, y1, x0, x1 = max(y - 5, 0), min(y + 6, h), max(x - 5, 0), min(x + 6, w)
            patch = lm[y0:y1, x0:x1].copy()
            patch[max(y - 2, 0) - y0:y + 3 - y0, max(x - 2, 0) - x0:x + 3 - x0] = -np.inf
            if lm[y, x] - patch.max() < 0.7:
                continue                                   # streak, not a spike
            pairs.append((float(lm[y, x]), float(prom[y, x]), int(dy), int(dx)))
    pairs.sort(reverse=True)                      # strongest ABSOLUTE spectral magnitude first
    n_pairs = len(pairs)
    valid, kept = False, []
    if n_pairs:
        top = pairs[0][0]
        mags = np.array([q[0] for q in pairs])
        dominant = [q for q in pairs if q[0] >= top - 0.8]          # within a factor ~2.2 of the strongest
        # Interference = a FEW peaks that clearly dominate. Real texture/scene patterns give MANY
        # comparable peaks. With many pairs, the top group must also stand out from the median.
        stands_out = n_pairs <= texture_pairs or (top - np.median(mags)) >= 0.7
        valid = len(dominant) <= max_pairs and stands_out
        if valid:   # also notch harmonics that are still strong relative to the top peak
            kept = [q for q in pairs if q[0] >= top - 1.2][:2 * max_pairs]
    return dict(valid=valid, n_pairs=n_pairs, n_candidates=n_cand,
                max_prominence=pairs[0][1] if pairs else 0.0,
                peaks=[(q[2], q[3]) for q in kept],
                peak_prominences=[q[1] for q in kept],
                all_peaks=[(q[2], q[3]) for q in pairs[:40]])


def detect_noise_type(noisy):
    """
    Classify the noise in `noisy` (float RGB in [0,1]). NO access to the clean image or to any label.

    Evidence (each converted to a normalised score n = measurement / threshold; n >= 1 'fires'):
      * Salt-and-Pepper : fraction of ISOLATED extreme pixels (extreme value AND very different
                          from the 3x3 median, so legitimate black/white regions are ignored)
      * Periodic        : strongest symmetric, compact spectral spike (prominence in ln units)
      * Gaussian        : robust noise-sigma estimate (only trusted when the two tests above
                          did not fire, because impulses/sinusoids inflate the sigma estimate)

    Decision:
      - nothing fires                      -> 'Clean/Unknown'  (fallback)
      - one fires                          -> that class
      - several fire                       -> highest score wins, flagged ambiguous; if the
                                              winner's margin over the runner-up is < T_AMBIGUITY
                                              the result is 'Clean/Unknown' (fallback)
    Confidence = n/(n+1) for a single detection (0.5 at the threshold, ->1 for strong evidence),
    multiplied by the relative margin when several classes fire. It is a heuristic score,
    NOT a calibrated probability.
    """
    if noisy.ndim != 3 or noisy.shape[2] != 3 or not np.isfinite(noisy).all():
        raise ValueError("detect_noise_type expects a finite (H, W, 3) array.")
    gray = _gray(noisy)
    imp = _impulse_features(noisy)
    per = _periodic_features(gray)
    sigma, s_imm, s_mad, kurt = _estimate_sigma(gray)

    # Gaussian noise that is clipped at 0/255 produces some isolated extremes too, but it also
    # raises the robust (MAD) sigma strongly, whereas impulse noise barely moves it.
    impulse_consistent = imp["impulse_ratio"] >= T_IMPULSE_VS_NOISE * s_mad
    n_sp = imp["impulse_ratio"] / T_IMPULSE if impulse_consistent else 0.0
    n_per = per["max_prominence"] / T_PERIODIC_PROM if per["valid"] else 0.0
    gauss_trusted = (n_sp < 1.0) and (n_per < 1.0)
    n_g = sigma / T_GAUSS_SIGMA if gauss_trusted else 0.0
    scores = {"Salt-and-Pepper": n_sp, "Periodic": n_per, "Gaussian": n_g}

    fired = {k: v for k, v in scores.items() if v >= 1.0}
    ambiguous, note = False, ""
    if not fired:
        label = "Clean/Unknown"
        confidence = float(np.clip(1.0 - max(scores.values()), 0.0, 1.0))
        note = "No noise signature exceeded its threshold (image is clean, or noise is too weak)."
    else:
        ranked = sorted(fired.items(), key=lambda kv: kv[1], reverse=True)
        label, top = ranked[0]
        if len(ranked) == 1:
            confidence = top / (top + 1.0)
        else:
            second = ranked[1][1]
            margin = (top - second) / top
            ambiguous = True
            confidence = top / (top + 1.0) * margin
            note = f"Several noise signatures fired ({', '.join(k for k, _ in ranked)})."
            if margin < T_AMBIGUITY:
                label, confidence = "Clean/Unknown", 0.0
                note += " Margin too small -> fallback."
    return dict(label=label, confidence=float(confidence), ambiguous=ambiguous, note=note,
                scores=scores,
                diagnostics=dict(impulse_ratio=imp["impulse_ratio"], extreme_ratio=imp["extreme_ratio"],
                                 impulse_purity=imp["purity"], salt_share=imp["salt_share"],
                                 sigma_est=sigma, sigma_immerkaer=s_imm, sigma_mad=s_mad,
                                 residual_excess_kurtosis=kurt,
                                 periodic_pairs=per["n_pairs"], periodic_candidates=per["n_candidates"],
                                 periodic_max_prominence=per["max_prominence"],
                                 gaussian_estimate_trusted=gauss_trusted),
                peaks=per["peaks"], peak_prominences=per["peak_prominences"])


# ======================================================================
# 4./5. FILTERS (each applied per RGB channel, then recombined)
# ======================================================================
def apply_median_filter(img, impulse_ratio=None, ksize=None):
    """
    Median filter per R, G, B channel (8-bit).  Median = order statistic -> it removes isolated
    0/255 outliers completely while preserving edges (unlike a mean filter).
    Kernel: 3x3 for light, 5x5 for medium, 7x7 for very heavy impulse density.
    """
    if ksize is None:
        d = impulse_ratio if impulse_ratio is not None else 0.1
        ksize = 3 if d < 0.12 else 5 if d < 0.25 else 7
    u8 = np.round(img * 255).astype(np.uint8)
    out = np.dstack([cv2.medianBlur(u8[..., c], ksize) for c in range(3)])
    return out.astype(np.float64) / 255.0, ksize


def _wiener_channel(x, sigma, window, refine=True):
    """
    Adaptive (locally-varying) Wiener filter for ONE channel.
    Pass 1 (scipy.signal.wiener):   y = mu + max(var_l - s^2, 0)/var_l * (x - mu)
    Pass 2 (empirical Wiener, optional): the local SIGNAL variance is re-estimated from the
        pass-1 result (a cleaner 'pilot' than the noisy input), giving a better gain
        g = var_signal / (var_signal + s^2)   ->   y = mean + g * (x - mu).
    Flat areas (g ~ 0) are smoothed strongly, edges (g ~ 1) are kept.
    """
    s2 = float(sigma) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        first = np.nan_to_num(signal.wiener(x, mysize=window, noise=s2))
    if not refine:
        return first
    k = (window, window)
    bd = cv2.BORDER_REFLECT_101
    mu = cv2.blur(x, k, borderType=bd)
    m1 = cv2.blur(first, k, borderType=bd)
    v1 = np.maximum(cv2.blur(first ** 2, k, borderType=bd) - m1 ** 2, 0.0)
    return m1 + v1 / (v1 + s2 + 1e-12) * (x - mu)


def apply_wiener_filter(img, sigma=None, window=WIENER_WINDOW, color_space=WIENER_COLOR_SPACE,
                        chroma_window=WIENER_CHROMA_WINDOW, refine=WIENER_REFINE):
    """
    Wiener filtering for Gaussian noise.
      * The noise sigma is estimated SEPARATELY for every processed channel (R,G,B or Y,Cr,Cb)
        from that channel itself. (Estimating it on a gray-scale mix of R,G,B under-estimates
        the per-channel noise by ~35 % and makes the filter far too weak.)
        `sigma` (a single value) can be given to override the estimate.
      * YCrCb mode: R,G,B noise is strongly correlated with the signal across channels, while in
        YCrCb the image energy sits in Y and the chroma planes are smooth, so chroma can be
        filtered with a larger window without blurring edges. The colour information is
        preserved because the transform is invertible and Y/Cr/Cb are recombined afterwards.
      * RGB mode: each of R, G, B is filtered independently with the same method.
    Returns (denoised_rgb, info_string).
    """
    x = img.astype(np.float32)
    if color_space.upper() == "YCRCB":
        planes = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb).astype(np.float64)
        wins, names = [window, chroma_window, chroma_window], "Y/Cr/Cb"
    else:
        planes = img.astype(np.float64)
        wins, names = [window] * 3, "R/G/B"
    out, sigmas = np.empty_like(planes), []
    for c in range(3):
        sg = float(sigma) if sigma is not None else _estimate_sigma(planes[..., c])[0]
        sg = max(sg, 1e-4)
        sigmas.append(sg)
        out[..., c] = _wiener_channel(planes[..., c], sg, wins[c], refine)
    if color_space.upper() == "YCRCB":
        out = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_YCrCb2RGB).astype(np.float64)
    out = np.clip(np.nan_to_num(out), 0, 1)
    desc = (f"{color_space} windows={'/'.join(str(w) for w in wins)}, refine={refine}, "
            f"est. sigma {names}=" + "/".join(f"{v:.3f}" for v in sigmas))
    return out, desc


def build_notch_mask(shape, peaks, sigma=3.5):
    """
    Gaussian notch-reject transfer function  H = prod_k [1 - exp(-D_k^2 / (2 sigma^2))]
    centred on every detected peak (u_k, v_k) AND its symmetric counterpart (-u_k, -v_k).
    Gaussian notches (instead of ideal ones) avoid ringing in the restored image.
    The DC term and all other frequencies keep gain 1, so legitimate content is preserved.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    v, u = np.ogrid[:h, :w]
    mask = np.ones((h, w), dtype=np.float64)
    for dy, dx in peaks:
        for s in (1, -1):
            d2 = (v - (cy + s * dy)) ** 2 + (u - (cx + s * dx)) ** 2
            mask *= 1.0 - np.exp(-d2 / (2.0 * sigma ** 2))
    return mask


def apply_notch_reject_filter(img, peaks, sigma=3.5):
    """
    FFT -> multiply by notch mask -> inverse FFT, on each RGB channel with the SAME mask
    (detected once on the luminance).  A linear filter applied identically to R, G and B keeps
    colour balance intact; filtering only the luminance would also work when the interference
    is identical in all channels, but per-channel filtering also covers colour-dependent patterns.
    """
    if not peaks:
        return img.copy(), None
    h, w = img.shape[:2]
    mask = build_notch_mask((h, w), peaks, sigma)
    out = np.empty_like(img)
    for c in range(3):
        F = fftshift(fft2(img[..., c]))
        out[..., c] = np.real(ifft2(ifftshift(F * mask)))
    return np.clip(out, 0, 1), mask


def _fallback_filter(img):
    """Conservative edge-preserving smoothing used when the noise type is unknown."""
    u8 = np.round(img * 255).astype(np.uint8)
    return cv2.bilateralFilter(u8, d=5, sigmaColor=15, sigmaSpace=3).astype(np.float64) / 255.0


def adaptive_denoise(noisy, detection=None):
    """
    Main adaptive filter: runs the detector on the noisy image (unless a detection is passed)
    and routes the image to the filter that matches the PREDICTED class.
    Returns (denoised, info_dict).
    """
    det = detection if detection is not None else detect_noise_type(noisy)
    label = det["label"]
    info = dict(label=label, filter=FILTER_FOR[label], params="", notch_mask=None)
    if label == "Gaussian":
        den, desc = apply_wiener_filter(noisy)          # sigma estimated per channel inside the filter
        info["params"] = desc
    elif label == "Salt-and-Pepper":
        den, k = apply_median_filter(noisy, impulse_ratio=det["diagnostics"]["impulse_ratio"])
        info["params"] = f"kernel={k}x{k}"
    elif label == "Periodic":
        den, mask = apply_notch_reject_filter(noisy, det["peaks"])
        info["params"] = f"{len(det['peaks'])} peak pair(s), notch sigma=3.5 px"
        info["notch_mask"] = mask
    else:
        den = _fallback_filter(noisy)
        info["params"] = "d=5, sigmaColor=15, sigmaSpace=3"
    return den, info


# ======================================================================
# 6. ENHANCEMENT
# ======================================================================
def contrast_stretch(img, percentiles=CONTRAST_PERCENTILES, max_gain=CONTRAST_MAX_GAIN):
    """
    Percentile-based linear contrast stretch.
      * The useful range [lo, hi] = luminance percentiles (default 1 % / 99 %).
      * gain = 1/(hi-lo) is capped at `max_gain` (no excessive amplification).
      * The stretch pivots around the MEDIAN luminance, so the overall brightness is preserved
        (a plain (x-lo)/(hi-lo) mapping shifts the mean and costs a lot of PSNR).
      * The same linear map is used for R, G and B -> colour ratios are preserved.
      * If the image already uses the full range (gain ~ 1) nothing is changed.
    """
    y = _gray(img)
    lo, hi = np.percentile(y, percentiles)
    if hi - lo < 1e-6:
        return img.copy()
    if hi - lo >= CONTRAST_SKIP_ABOVE:                # contrast already adequate: stretching would
        return img.copy()                              # only move pixels away from the true image
    gain = min(1.0 / (hi - lo), max_gain)
    if gain < 1.02:
        return img.copy()
    pivot = float(np.median(y))
    # largest gain that keeps the percentile limits inside [0,1] when pivoting at the median
    safe = min(gain, pivot / max(pivot - lo, 1e-6), (1.0 - pivot) / max(hi - pivot, 1e-6))
    gain = max(1.0, safe)
    return np.clip(pivot + gain * (img - pivot), 0, 1)


def laplacian_sharpen(img, strength=SHARPEN_STRENGTH, smooth_sigma=1.0, max_delta=0.06):
    """
    Noise-aware Laplacian sharpening on the luminance:
        out = img - strength * LoG(Y) * edge_mask
      * Gaussian pre-smoothing (Laplacian of Gaussian) stops residual noise being amplified.
      * The residual-noise level of the denoised image is estimated; the sharpening strength
        is reduced when noise remains, and responses below ~1.5 x that noise level are removed
        ("coring"), so flat areas are NOT re-noised.
      * An edge mask limits the correction to real edges; the correction is also bounded
        (max_delta) to prevent halos / ringing.
      * cv2 BORDER_REFLECT_101 handles the image boundaries.
    The same luminance correction is added to R, G and B -> no colour shifts.
    """
    if strength <= 0:
        return img.copy()
    y = _gray(img)
    sig_res = _estimate_sigma(y)[0]                              # residual noise after denoising
    eff = strength / (1.0 + (sig_res / 0.03) ** 2)               # less sharpening if still noisy
    ys = cv2.GaussianBlur(y, (0, 0), smooth_sigma, borderType=cv2.BORDER_REFLECT_101)
    lap = cv2.Laplacian(ys, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT_101)
    delta = -eff * lap
    coring = max(0.002, 0.15 * sig_res)
    delta = np.sign(delta) * np.maximum(np.abs(delta) - coring, 0.0)
    gx = cv2.Sobel(ys, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(ys, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    grad = np.hypot(gx, gy) / 8.0
    g0 = 0.5 * sig_res + 0.004
    mask = np.clip((grad - g0) / (2.0 * g0 + 1e-6), 0.0, 1.0)    # 0 in flat/noisy areas, 1 on edges
    delta = np.clip(delta * mask, -max_delta, max_delta)
    return np.clip(img + delta[..., None], 0, 1)


def enhance_image(img, strength=SHARPEN_STRENGTH):
    """Contrast stretching followed by Laplacian sharpening. Returns (final, stretched)."""
    stretched = contrast_stretch(img)
    return laplacian_sharpen(stretched, strength), stretched


# ======================================================================
# 7. METRICS
# ======================================================================
def calculate_metrics(reference, test):
    """MSE / PSNR / SSIM against the clean reference. Both converted to the same 0-255 scale."""
    if reference.shape != test.shape:
        raise ValueError(f"Shape mismatch {reference.shape} vs {test.shape}")
    if not (np.isfinite(reference).all() and np.isfinite(test).all()):
        return dict(MSE=np.nan, PSNR=np.nan, SSIM=np.nan)
    a = np.round(np.clip(reference, 0, 1) * 255).astype(np.uint8)
    b = np.round(np.clip(test, 0, 1) * 255).astype(np.uint8)
    mse = float(mean_squared_error(a, b))
    psnr = float(peak_signal_noise_ratio(a, b, data_range=255)) if mse > 0 else np.inf
    win = min(7, min(a.shape[:2]) // 2 * 2 - 1)
    ssim = float(structural_similarity(a, b, data_range=255, channel_axis=2, win_size=win))
    return dict(MSE=mse, PSNR=psnr, SSIM=ssim)


# ======================================================================
# FULL PROCESSING OF ONE NOISY IMAGE
# ======================================================================
def process_noisy_image(noisy):
    """Detector -> adaptive filter -> enhancement. Receives ONLY the noisy image."""
    t0 = time.perf_counter()
    det = detect_noise_type(noisy)
    denoised, info = adaptive_denoise(noisy, det)
    final, stretched = enhance_image(denoised)
    return dict(detection=det, info=info, denoised=denoised, stretched=stretched,
                final=final, time=time.perf_counter() - t0)


# ======================================================================
# 8./9. PLOTS
# ======================================================================
def _finish(fig, name, show=SHOW_PLOTS):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def display_inputs(original, noisy):
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    items = [("Original (clean)", original)] + [(f"{k} noise", v) for k, v in noisy.items()]
    for a, (t, im) in zip(ax, items):
        a.imshow(im); a.set_title(t); a.axis("off")
    fig.suptitle("Original and generated noisy images")
    _finish(fig, "01_original_and_noisy.png")


def display_detection(results, truth):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for a, name in zip(ax, results):
        d = results[name]["detection"]; g = d["diagnostics"]
        ok = "CORRECT" if d["label"] == truth[name] else "WRONG"
        txt = (f"Generated (ground truth): {truth[name]}\n"
               f"PREDICTED: {d['label']}  [{ok}]\n"
               f"Confidence (heuristic): {d['confidence']:.2f}   Ambiguous: {d['ambiguous']}\n"
               f"Selected filter: {results[name]['info']['filter']}\n"
               f"Filter params: {results[name]['info']['params']}\n\n"
               f"--- normalised scores (>=1 fires) ---\n"
               + "\n".join(f"{k:16s}: {v:6.2f}" for k, v in d["scores"].items()) +
               f"\n\n--- diagnostics ---\n"
               f"isolated-extreme ratio : {g['impulse_ratio']:.4f}\n"
               f"extreme-pixel ratio    : {g['extreme_ratio']:.4f}\n"
               f"extreme/isolated share : {g['impulse_purity']:.2f}\n"
               f"luminance sigma (Immerk.): {g['sigma_immerkaer']:.4f}\n"
               f"luminance sigma (MAD)   : {g['sigma_mad']:.4f}\n"
               f"residual excess kurt.  : {g['residual_excess_kurtosis']:.1f}\n"
               f"spectral peak pairs    : {g['periodic_pairs']}\n"
               f"max peak prominence    : {g['periodic_max_prominence']:.2f} (ln)\n"
               + (f"\nNote: {d['note']}" if d["note"] else ""))
        a.text(0.0, 1.0, txt, va="top", family="monospace", fontsize=8.6, wrap=True)
        a.set_title(f"Test image: {name} noise"); a.axis("off")
    fig.suptitle("Automatic noise detection - predicted class and diagnostics")
    _finish(fig, "02_detection_diagnostics.png")


def display_fft(noisy_periodic, res):
    """FFT spectrum of the periodic-noisy image (with detected peaks), the notch mask, and after filtering."""
    g_noisy, g_den = _gray(noisy_periodic), _gray(res["denoised"])
    s0, s1 = log_spectrum(g_noisy), log_spectrum(g_den)
    h, w = g_noisy.shape
    cy, cx = h // 2, w // 2
    vmax = s0.max()
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    ax[0].imshow(s0, cmap="gray", vmin=0, vmax=vmax)
    for dy, dx in res["detection"]["peaks"]:
        for s in (1, -1):
            ax[0].add_patch(plt.Circle((cx + s * dx, cy + s * dy), 14, fill=False, color="red", lw=1.4))
    ax[0].set_title("FFT log-magnitude: periodic-noisy image\n(red circles = detected peak pairs)")
    mask = res["info"]["notch_mask"]
    ax[1].imshow(mask if mask is not None else np.ones_like(s0), cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Notch-reject transfer function H(u,v)")
    ax[2].imshow(s1, cmap="gray", vmin=0, vmax=vmax)
    ax[2].set_title("FFT log-magnitude after notch filtering")
    for a in ax:
        a.axis("off")
    _finish(fig, "03_fft_spectrum_periodic.png")


def display_results(original, noisy, results, metrics, truth):
    names = list(results)
    fig, ax = plt.subplots(len(names), 4, figsize=(18, 4.6 * len(names)))
    for r, n in enumerate(names):
        panels = [("Original", original, None),
                  (f"Noisy ({n})", noisy[n], metrics[n]["Noisy"]),
                  ("Denoised (before enhancement)", results[n]["denoised"], metrics[n]["Denoised"]),
                  ("Final enhanced", results[n]["final"], metrics[n]["Enhanced"])]
        for c, (t, im, m) in enumerate(panels):
            ax[r, c].imshow(im); ax[r, c].axis("off")
            sub = "" if m is None else f"\nMSE {m['MSE']:.1f} | PSNR {m['PSNR']:.2f} dB | SSIM {m['SSIM']:.3f}"
            ax[r, c].set_title(t + sub, fontsize=10)
        d = results[n]
        ax[r, 0].text(0.0, -0.06, f"Predicted: {d['detection']['label']} -> {d['info']['filter']}",
                      transform=ax[r, 0].transAxes, fontsize=9, va="top", color="navy")
    fig.suptitle("Original vs noisy vs denoised vs enhanced", fontsize=14)
    _finish(fig, "04_comparison.png")


def plot_metric_comparisons(metrics):
    names = list(metrics)
    colors = {"Noisy": "#d95f02", "Denoised": "#1b9e77", "Enhanced": "#7570b3"}
    spec = [("MSE", "MSE (0-255 scale, lower is better)", "05_mse.png", None),
            ("PSNR", "PSNR (dB, higher is better)", "06_psnr.png", None),
            ("SSIM", "SSIM (0-1, higher is better)", "07_ssim.png", (0, 1.05))]
    for key, ylabel, fname, ylim in spec:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        x, wd = np.arange(len(names)), 0.26
        top = 0.0
        for i, st in enumerate(STAGES):
            vals = np.array([metrics[n][st][key] for n in names], dtype=float)
            shown = np.where(np.isfinite(vals), vals, 0.0)           # inf / NaN handled explicitly
            bars = ax.bar(x + (i - 1) * wd, shown, wd, label=st, color=colors[st])
            top = max(top, shown.max())
            for b, v in zip(bars, vals):
                txt = "n/a" if np.isnan(v) else ("inf" if np.isinf(v) else f"{v:.3f}" if key == "SSIM" else f"{v:.1f}")
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_xlabel("Noise type"); ax.set_ylabel(ylabel)
        ax.set_title(f"{key} comparison: noisy vs denoised vs enhanced")
        if ylim: ax.set_ylim(*ylim)
        else: ax.set_ylim(0, top * 1.15 if top > 0 else 1)
        ax.grid(axis="y", alpha=0.3); ax.legend(title="Processing stage")
        _finish(fig, fname)


def plot_confusion_matrix(cm, true_labels, pred_labels, acc):
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(pred_labels))); ax.set_xticklabels(pred_labels, rotation=25, ha="right")
    ax.set_yticks(range(len(true_labels))); ax.set_yticklabels(true_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Detector prediction"); ax.set_ylabel("Generated (ground truth)")
    ax.set_title(f"Noise classification confusion matrix (accuracy {acc * 100:.1f}%)")
    _finish(fig, "08_confusion_matrix.png")


# ======================================================================
# 11. CLASSIFIER EVALUATION OVER RANDOMISED TRIALS
# ======================================================================
def evaluate_classifier(clean, n_per_class=N_TRIALS_PER_CLASS, seed=SEED):
    """
    Random noise levels per trial. Ground-truth labels are stored separately and are only
    compared with the detector's output afterwards. 'Clean' images should map to 'Clean/Unknown'.
    """
    rng = np.random.default_rng(seed + 1000)
    true_labels = ["Gaussian", "Salt-and-Pepper", "Periodic", "Clean/Unknown"]
    cm = np.zeros((4, 4), dtype=int)
    for _ in range(n_per_class):
        sp = rng.uniform(0.02, 0.10)
        tests = [
            (0, add_gaussian_noise(clean, rng.uniform(0.004, 0.03), rng)),
            (1, add_salt_pepper_noise(clean, sp, sp, rng)),
            (2, add_periodic_noise(clean, rng.uniform(15, 70), rng.uniform(0, 180),
                                   rng.uniform(0.08, 0.25), rng.uniform(0, 360))),
            (3, clean.copy()),
        ]
        for ti, im in tests:
            pred = detect_noise_type(im)["label"]
            cm[ti, LABELS.index(pred)] += 1
    acc = np.trace(cm) / cm.sum()
    return cm, true_labels, acc


# ======================================================================
# MAIN
# ======================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- 1. input -------------------------------------------------------
    path = get_image_path()
    original = load_original_image(path)               # reference image - never modified
    print(f"[info] Loaded '{path}'  size={original.shape[1]}x{original.shape[0]}  "
          f"dtype={original.dtype}  range=[{original.min():.2f}, {original.max():.2f}]")
    fig = plt.figure(figsize=(6, 6)); plt.imshow(original); plt.title("Original clean image"); plt.axis("off")
    _finish(fig, "00_original.png")

    # ---- 2. generate noise (ground truth kept apart from the pipeline) -------
    noisy = {
        "Gaussian": add_gaussian_noise(original, GAUSSIAN_VARIANCE, rng),
        "Salt-and-Pepper": add_salt_pepper_noise(original, SALT_PROB, PEPPER_PROB, rng),
        "Periodic": add_periodic_noise(original, **PERIODIC_PARAMS),
    }
    truth = {k: k for k in noisy}                       # used ONLY for reporting accuracy
    display_inputs(original, noisy)

    # ---- 3-6. detect -> filter -> enhance ----------------------------------
    results, metrics = {}, {}
    for name, img in noisy.items():
        try:
            res = process_noisy_image(img)              # note: only the noisy image is passed in
            results[name] = res
            metrics[name] = {"Noisy": calculate_metrics(original, img),
                             "Denoised": calculate_metrics(original, res["denoised"]),
                             "Enhanced": calculate_metrics(original, res["final"])}
            for tag, im in (("noisy", img), ("denoised", res["denoised"]), ("enhanced", res["final"])):
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name.replace('-', '')}_{tag}.png"),
                            cv2.cvtColor(np.round(im * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        except Exception as exc:                        # graceful failure for one noise type
            print(f"[error] Processing failed for {name}: {exc}")

    if not results:
        print("Nothing could be processed."); return

    # ---- console report ---------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'Test image':<17}{'Predicted':<16}{'OK?':<5}{'Conf.':<7}{'Selected filter':<34}{'Time(s)':<8}")
    print("-" * 100)
    for n, r in results.items():
        ok = "yes" if r["detection"]["label"] == truth[n] else "NO"
        print(f"{n:<17}{r['detection']['label']:<16}{ok:<5}{r['detection']['confidence']:<7.2f}"
              f"{r['info']['filter']:<34}{r['time']:<8.2f}")
    print("=" * 100)

    rows = [["Noise Type", "Processing Stage", "MSE", "PSNR (dB)", "SSIM"]]
    print(f"\n{'Noise Type':<17}{'Processing Stage':<18}{'MSE':>12}{'PSNR (dB)':>12}{'SSIM':>9}")
    print("-" * 68)
    for n in metrics:
        for st in STAGES:
            m = metrics[n][st]
            print(f"{n:<17}{st:<18}{m['MSE']:>12.2f}{m['PSNR']:>12.2f}{m['SSIM']:>9.4f}")
            rows.append([n, st, f"{m['MSE']:.4f}", f"{m['PSNR']:.4f}", f"{m['SSIM']:.4f}"])
        print("-" * 68)
    with open(os.path.join(OUTPUT_DIR, "metrics.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # ---- visualisations ------------------------------------------------------
    display_detection(results, truth)
    if "Periodic" in results:
        display_fft(noisy["Periodic"], results["Periodic"])
    display_results(original, noisy, results, metrics, truth)
    plot_metric_comparisons(metrics)

    # ---- classifier reliability (confusion matrix) ------------------------------
    if N_TRIALS_PER_CLASS > 0:
        print(f"\n[info] Running classifier test ({N_TRIALS_PER_CLASS} random trials per class)...")
        cm, tl, acc = evaluate_classifier(original, N_TRIALS_PER_CLASS)
        print("Rows = ground truth, columns = prediction:", LABELS)
        print(cm)
        print(f"Overall classification accuracy: {acc * 100:.1f}%")
        plot_confusion_matrix(cm, tl, LABELS, acc)

    print(f"\nAll figures, images and metrics.csv were saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("Note: contrast stretching + sharpening can raise visual quality yet lower PSNR/SSIM, "
          "because they deliberately move pixel values away from the clean reference.")


if __name__ == "__main__":
    main()
