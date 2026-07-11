"""
weeping_demon_dsp.py
====================
Python interface to the Weeping Demon Wah DSP engine.

Strategy
--------
1. Try to load the compiled C++ shared library (weeping_demon_dsp.dll).
2. If the DLL is missing, fall back to a pure-Python implementation that
   replicates WDEffect.cpp exactly using NumPy.

The public API is identical in both cases:

    wd = WahEffect(buffer_length=512, sample_rate=44100)
    wd.set_parameters(wah=10, q=5000, level=3000, lo=50000, range=500, mode=0)
    output = wd.process(input_array)   # np.ndarray float64, shape (N,)
"""

import ctypes
import os
import math
import numpy as np

# ──────────────────────────────────────────────────────────────
#  DLL loader
# ──────────────────────────────────────────────────────────────

_DLL_PATH = os.path.join(os.path.dirname(__file__), "weeping_demon_dsp.dll")

def _try_load_dll():
    """Return the loaded ctypes DLL, or None if not available."""
    if not os.path.exists(_DLL_PATH):
        return None
    try:
        dll = ctypes.CDLL(_DLL_PATH)
        # create_wd_effect(int buffer_length, int sample_rate) -> void*
        dll.create_wd_effect.restype  = ctypes.c_void_p
        dll.create_wd_effect.argtypes = [ctypes.c_int, ctypes.c_int]
        # destroy_wd_effect(void*)
        dll.destroy_wd_effect.restype  = None
        dll.destroy_wd_effect.argtypes = [ctypes.c_void_p]
        # set_parameters(void*, double x6, int mode)
        dll.set_parameters.restype  = None
        dll.set_parameters.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ]
        # process_audio(void*, double* in, double* out, int len)
        dll.process_audio.restype  = None
        dll.process_audio.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        return dll
    except OSError:
        return None


# ──────────────────────────────────────────────────────────────
#  Pure-Python fallback  (mirrors WDEffect.cpp exactly)
# ──────────────────────────────────────────────────────────────

class _PythonWahDSP:
    """
    Faithful Python port of WDEffect.cpp (Chet Gnegy, 2013).

    Resistor & capacitor values are the measured (not nominal) values from
    WDEffect.cpp:L97-L116; these already encode the WD_PotTapers data via
    the fitted curve WAH = m_fudge * 1.6933367e10 * theta^(-4.45855 * e_fudge).
    """

    # Circuit constants  (from WDEffect.cpp L99-L116)
    R108 = 9.79e3;   R109 = 21.3e3;  R110 = 23.9e3;  R111 = 46.4e3
    R113 = 197.0e3;  R114 = 10.97e3; R115 = 982.0;   R117 = 14.77e3
    R120 = 3.52e3;   R122 = 4.95e3;  R123 = 4.65e3
    C104 = 2.76e-9;  C105 = 10.3e-9; C118 = 19.5e-9; C119 = 10.1e-9
    VR6  = 0.0;      VR7  = 449e3

    M_FUDGE = 1.1;   E_FUDGE = 0.95

    def __init__(self, buffer_length: int = 512, sample_rate: int = 44100):
        self.buffer_length = buffer_length
        self.sample_rate   = sample_rate

        # Knob defaults (midpoint of ranges)
        self.WAH_theta = 0.3   # default from WDEffect (Knob(5,17,.3) -> .3*range)
        self.Q         = 5000.0
        self.LEVEL     = 3000.0
        self.LO        = 50000.0
        self.RANGE     = 500.0
        self.bass_     = 1     # 1 = bass mode default

        # IIR state
        self.a_ = np.zeros(4);  self.b_ = np.zeros(4)
        self.past_a_ = np.zeros(4); self.past_b_ = np.zeros(4)
        self.x_past_ = np.zeros(4); self.y_past_ = np.zeros(4)

        self._recalc()

    # ------------------------------------------------------------------
    # Knob-style taper  (WAH_theta is the raw angle sent from the pedal)
    # ------------------------------------------------------------------
    def _wah_resistance(self, theta: float) -> float:
        """Convert pedal angle to filter resistance (potentiometer taper fit)."""
        theta = max(theta, 1e-6)
        return self.M_FUDGE * 1.6933367e10 * math.pow(theta, -4.45855 * self.E_FUDGE)

    # ------------------------------------------------------------------
    # Coefficient calculation  (WDEffect::calculate_coefficients)
    # ------------------------------------------------------------------
    def _recalc(self):
        WAH = self._wah_resistance(self.WAH_theta)
        R = self  # shorthand

        if self.bass_:
            x0 = R.LO + R.R122 + R.R123
            x1 = R.C105 * R.R117 * x0
            x2 = R.Q + R.R114
            x3 = R.R109 + R.R110
            x4 = R.R113 + R.VR7
            x5 = WAH + x4
            x6 = R.R108 + x2
            x7 = R.R115 + R.RANGE + R.VR6
            B2 = 0.0
            A3 = 0.0
            B1 = R.LEVEL * x1 * x2 * x3 * x5
            B0 = R.R123 * x2 * x3 * x5 * (R.LEVEL + R.R120)
            A2 = R.R109*R.R120*x1*x6*(R.C104+R.C119)*(WAH*(x4+x7)+x4*x7)
            A1 = R.C105*R.R108*R.R117*R.R120*x0*x3*x5
            A0 = R.R110*R.R120*x0*x5*x6
        else:
            x0 = R.LO + R.R122 + R.R123
            x1 = R.C105 * R.R111 * R.R117 * x0
            x2 = R.Q + R.R114
            x3 = R.R109 + R.R110
            x4 = R.R113 + R.VR7
            x5 = WAH + x4
            x6 = x2 * x3 * x5
            x7 = R.C118 * R.R111 * R.R123
            x8 = R.C105 * R.R117
            x9 = x0 * x8
            x10 = R.R108 + x2
            x11 = R.R115 + R.RANGE + R.VR6
            x12 = WAH*(x11+x4) + x11*x4
            x13 = R.R110 * R.R111
            x14 = R.C118 * R.R108 * x5
            x15 = R.C104 * x12
            x16 = R.R120 * x0 * x5
            x17 = R.R108 + R.R111
            B2 = R.C118 * R.LEVEL * x1 * x2 * x3 * x5
            B1 = x6 * (R.LEVEL*(x7+x9) + R.R120*x7)
            B0 = R.R123 * x6 * (R.LEVEL + R.R120)
            A3 = R.C104*R.C118*R.R109*R.R120*x1*x10*x12
            A2 = R.R120*x9*(R.R109*(R.R111*(x14+x15)+x10*x15)+x13*x14)
            A1 = x16*(R.C118*x10*x13 + x17*x3*x8)
            A0 = R.R110*x16*(x17 + x2)

        # Bilinear transform  (WDEffect::calculate_coefficients L196-208)
        f  = float(self.sample_rate)
        f2 = f * f
        f3 = f2 * f

        denom = 8*A3*f3 + 4*A2*f2 + 2*A1*f + A0
        if abs(denom) < 1e-300:
            return  # degenerate state; keep old coefficients

        self.past_a_[:] = self.a_
        self.past_b_[:] = self.b_

        self.a_[0] = 1.0
        self.a_[1] = (-24*A3*f3 - 4*A2*f2 + 2*A1*f + 3*A0) / denom
        self.a_[2] = ( 24*A3*f3 - 4*A2*f2 - 2*A1*f + 3*A0) / denom
        self.a_[3] = ( -8*A3*f3 + 4*A2*f2 - 2*A1*f +   A0) / denom

        self.b_[0] = ( 4*B2*f2 + 2*B1*f + B0) / denom
        self.b_[1] = (-4*B2*f2 + 2*B1*f + 3*B0) / denom
        self.b_[2] = (-4*B2*f2 - 2*B1*f + 3*B0) / denom   # note: sign matches C++ L205
        self.b_[3] = ( 4*B2*f2 - 2*B1*f +   B0) / denom

    # ------------------------------------------------------------------
    # Sample-by-sample filtering  (WDEffect::filter_tick)
    # ------------------------------------------------------------------
    def _filter_tick(self, x_in: float) -> float:
        self.x_past_ = np.roll(self.x_past_, 1)
        self.x_past_[0] = x_in

        y = (  self.b_[0]*self.x_past_[0]
             + self.b_[1]*self.x_past_[1]
             + self.b_[2]*self.x_past_[2]
             + self.b_[3]*self.x_past_[3]
             - self.a_[1]*self.y_past_[1]
             - self.a_[2]*self.y_past_[2]
             - self.a_[3]*self.y_past_[3] )

        # Guard: clamp / reset on overflow or NaN (mirrors C++ NaN check)
        if not np.isfinite(y) or abs(y) > 15.0:
            y = 0.0
            self.x_past_[:] = 0.0
            self.y_past_[:] = 0.0


        self.y_past_ = np.roll(self.y_past_, 1)
        self.y_past_[0] = y
        return y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_parameters(self, wah: float, q: float, level: float,
                       lo: float, rang: float, mode: int):
        self.WAH_theta = max(wah, 1e-6)
        self.Q         = q
        self.LEVEL     = level
        self.LO        = lo
        self.RANGE     = rang
        self.bass_     = 1 if mode else 0
        self._recalc()

    def process(self, signal: np.ndarray) -> np.ndarray:
        out = np.empty_like(signal)
        for i, s in enumerate(signal):
            out[i] = self._filter_tick(s)
        return out


# ──────────────────────────────────────────────────────────────
#  DLL-backed implementation
# ──────────────────────────────────────────────────────────────

class _DllWahDSP:
    def __init__(self, dll, buffer_length: int, sample_rate: int):
        self._dll    = dll
        self._handle = dll.create_wd_effect(buffer_length, sample_rate)
        self._len    = buffer_length

    def __del__(self):
        if self._handle:
            self._dll.destroy_wd_effect(self._handle)
            self._handle = None

    def set_parameters(self, wah, q, level, lo, rang, mode):
        self._dll.set_parameters(self._handle, wah, q, level, lo, rang, int(mode))

    def process(self, signal: np.ndarray) -> np.ndarray:
        sig64 = signal.astype(np.float64)
        out64 = np.zeros(len(sig64), dtype=np.float64)
        in_ptr  = sig64.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_ptr = out64.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._dll.process_audio(self._handle, in_ptr, out_ptr, len(sig64))
        return out64


# ──────────────────────────────────────────────────────────────
#  Public factory
# ──────────────────────────────────────────────────────────────

def WahEffect(buffer_length: int = 512, sample_rate: int = 44100):
    """
    Factory that returns either the DLL-backed or pure-Python Wah DSP.

    Usage::

        wd = WahEffect(buffer_length=512, sample_rate=44100)
        wd.set_parameters(wah=10, q=5000, level=3000, lo=50000, range=500, mode=0)
        out = wd.process(input_samples)   # np.ndarray float64
    """
    _dll = _try_load_dll()
    if _dll is not None:
        print("[WahEffect] Using compiled C++ DLL backend.")
        return _DllWahDSP(_dll, buffer_length, sample_rate)
    else:
        print("[WahEffect] DLL not found – using pure-Python fallback.")
        return _PythonWahDSP(buffer_length, sample_rate)
