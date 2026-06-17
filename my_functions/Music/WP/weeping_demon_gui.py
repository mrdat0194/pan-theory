"""
weeping_demon_gui.py
====================
OpenCV-based interactive control panel + real-time spectrogram for the
Weeping Demon Wah pedal emulator.

The GUI is intentionally headless-friendly: if OpenCV cannot open a
display window it falls back to a no-op stub so the DSP still runs.
"""

import threading
import numpy as np

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

# ──────────────────────────────────────────────────────────────
#  Layout constants
# ──────────────────────────────────────────────────────────────

WIN_NAME   = "Weeping Demon Wah"
PANEL_W    = 700
SPEC_H     = 200          # pixels per spectrogram (input + output stacked)
CTRL_H     = 180          # height of the trackbar control area
TOTAL_H    = SPEC_H * 2 + CTRL_H
SPEC_BINS  = 256           # displayed FFT bins (x axis = frequency)
SPEC_FRAMES = PANEL_W      # scroll history length = window width

_PALETTE = None  # lazily built JET colourmap


def _build_palette():
    global _PALETTE
    if not _CV2_OK:
        return
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for v in range(256):
        colour = cv2.applyColorMap(np.array([[[v]]], dtype=np.uint8), cv2.COLORMAP_JET)
        lut[v, 0] = colour[0, 0]
    _PALETTE = lut


# ──────────────────────────────────────────────────────────────
#  Spectrogram buffer
# ──────────────────────────────────────────────────────────────

class _SpectrogramBuffer:
    """Ring-buffer of FFT magnitude columns."""
    def __init__(self, n_bins=SPEC_BINS, n_frames=SPEC_FRAMES):
        self.n_bins   = n_bins
        self.n_frames = n_frames
        self._buf     = np.zeros((n_bins, n_frames), dtype=np.float32)
        self._col     = 0

    def push(self, samples: np.ndarray):
        if len(samples) < 2:
            return
        fft_mag = np.abs(np.fft.rfft(samples, n=self.n_bins * 2))[:self.n_bins]
        fft_mag = np.log1p(fft_mag) / 10.0  # rough dB normalise
        self._buf[:, self._col] = fft_mag.astype(np.float32)
        self._col = (self._col + 1) % self.n_frames

    def image(self) -> np.ndarray:
        """Return (n_bins, n_frames) uint8 image (frequency = rows, time = cols)."""
        # Roll so newest column is on the right
        ordered = np.roll(self._buf, -self._col, axis=1)
        # Flip frequency axis so low = bottom
        ordered = np.flipud(ordered)
        img = np.clip(ordered * 255, 0, 255).astype(np.uint8)
        return img


# ──────────────────────────────────────────────────────────────
#  GUI class
# ──────────────────────────────────────────────────────────────

class WahGUI:
    """
    Interactive control panel for the Weeping Demon Wah emulator.

    Parameters are exposed as OpenCV trackbars; their values are read by
    weeping_demon_app.py and forwarded to the DSP engine each frame.

    Usage::

        gui = WahGUI()
        gui.start()                      # opens window in background thread
        while gui.is_open():
            params = gui.get_params()    # dict of current knob values
            gui.push_audio(dry, wet)     # update spectrograms
            gui.tick()                   # redraw (call from main thread)
        gui.stop()
    """

    # Trackbar ranges (integer, mapped to float in get_params)
    _KNOBS = {
        # name      : (int_min, int_max, default_int, scale_fn)
        "Wah"   : (5,    17,    7,    lambda v: float(v)),
        "Q"     : (100,  2500,  500,  lambda v: v * 100.0),
        "Level" : (100,  1000,  300,  lambda v: v * 10.0),
        "LO"    : (100,  1000,  500,  lambda v: v * 100.0),
        "Range" : (100,  2500,  500,  lambda v: float(v)),
        "Mode"  : (0,    1,     1,    lambda v: int(v)),
    }

    def __init__(self):
        self._open   = False
        self._lock   = threading.Lock()
        self._spec_in  = _SpectrogramBuffer()
        self._spec_out = _SpectrogramBuffer()
        self._canvas   = None
        _build_palette()

    # ------------------------------------------------------------------
    def start(self):
        if not _CV2_OK:
            print("[WahGUI] cv2 not available – GUI disabled.")
            return
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, PANEL_W, TOTAL_H)

        for name, (lo, hi, default, _) in self._KNOBS.items():
            cv2.createTrackbar(name, WIN_NAME, default, hi, lambda v: None)

        self._canvas = np.zeros((TOTAL_H, PANEL_W, 3), dtype=np.uint8)
        self._open   = True

    def stop(self):
        self._open = False
        if _CV2_OK:
            cv2.destroyWindow(WIN_NAME)

    def is_open(self) -> bool:
        if not _CV2_OK or not self._open:
            return False
        return cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) >= 1

    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        """Read current trackbar positions and return scaled float params."""
        if not _CV2_OK or not self._open:
            return {k: fn(d) for k, (_, _, d, fn) in self._KNOBS.items()}
        params = {}
        for name, (lo, hi, default, scale_fn) in self._KNOBS.items():
            raw = cv2.getTrackbarPos(name, WIN_NAME)
            raw = max(lo, min(hi, raw))
            params[name] = scale_fn(raw)
        return params

    # ------------------------------------------------------------------
    def push_audio(self, dry: np.ndarray, wet: np.ndarray):
        """Feed a processed block to the spectrogram buffers (thread-safe)."""
        with self._lock:
            self._spec_in.push(dry)
            self._spec_out.push(wet)

    # ------------------------------------------------------------------
    def tick(self) -> bool:
        """
        Redraw the window.  Call this from the main thread every ~30 ms.
        Returns False when the window has been closed.
        """
        if not _CV2_OK or not self._open:
            return False

        with self._lock:
            img_in  = self._spec_in.image()
            img_out = self._spec_out.image()

        canvas = self._canvas
        canvas[:] = 0

        # ---- input spectrogram (top half) ----
        coloured_in = cv2.applyColorMap(img_in.T, cv2.COLORMAP_JET)
        coloured_in = cv2.resize(coloured_in, (PANEL_W, SPEC_H))
        canvas[0:SPEC_H] = coloured_in

        # ---- label ----
        cv2.putText(canvas, "Input (dry)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ---- output spectrogram (middle) ----
        coloured_out = cv2.applyColorMap(img_out.T, cv2.COLORMAP_JET)
        coloured_out = cv2.resize(coloured_out, (PANEL_W, SPEC_H))
        canvas[SPEC_H:SPEC_H*2] = coloured_out
        cv2.putText(canvas, "Output (wet – Wah filtered)", (10, SPEC_H + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ---- separator ----
        cv2.line(canvas, (0, SPEC_H*2), (PANEL_W, SPEC_H*2), (80, 80, 80), 1)

        # ---- param readout ----
        params = self.get_params()
        y = SPEC_H*2 + 20
        for name, value in params.items():
            label = f"{name}: {value:.1f}" if isinstance(value, float) else f"{name}: {value}"
            cv2.putText(canvas, label, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)
            y += 22

        cv2.imshow(WIN_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # q or ESC closes
            self.stop()
            return False
        return True


# ──────────────────────────────────────────────────────────────
#  No-op stub when OpenCV is absent
# ──────────────────────────────────────────────────────────────

class _NoOpGUI:
    def start(self): pass
    def stop(self):  pass
    def is_open(self): return False
    def get_params(self):
        return {"Wah": 10.0, "Q": 5000.0, "Level": 3000.0,
                "LO": 50000.0, "Range": 500.0, "Mode": 1}
    def push_audio(self, dry, wet): pass
    def tick(self): return False
