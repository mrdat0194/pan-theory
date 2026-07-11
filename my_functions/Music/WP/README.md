# Weeping Demon Wah Emulator — Python Port Walkthrough

## What Was Built

The Weeping Demon Wah Pedal emulator from [`WD_v_1.1/`](file:///C:/Users/mrdat/Desktop/archive/AIML/Other/WD_v_1.1) has been ported into a Python-callable package at:

```
C:\Users\mrdat\PycharmProjects\pan-theory\my_functions\Music\WP\
```

---

## `build.bat` vs the original `makefile` — They Are NOT the Same

| | `makefile` (original) | `build.bat` (new) |
|---|---|---|
| **Output** | Native executable `WeepingDemon` | Shared library `weeping_demon_dsp.dll` |
| **Target OS** | Linux / macOS (JACK / CoreAudio) | Windows (MinGW-w64 g++) |
| **Audio layer** | RtAudio + RtMidi (reads mic/speaker) | None (audio is handled by Python `sounddevice`) |
| **GUI layer** | OpenGL + GLUT + Knob + Spectrogram | None (GUI handled by Python OpenCV) |
| **Sources compiled** | `AudioWrapper` `DigitalFilter` `fft` `WDEffect` `RtAudio` `RtMidi` `Stk` `Thread` `Graphics` `Knob` `Spectrogram` `RgbImage` | **Only `weeping_demon_wrapper.cpp`** |
| **C++ header used** | Original `WDEffect.h` (needs Knob/OpenGL) | `WDEffect_headless.h` (self-contained DSP shim) |

The original `WDEffect.h` could **not** be reused directly because it depends on `Knob.h` → `Graphics.h` → OpenGL, which cannot be linked into a headless DLL. `WDEffect_headless.h` was created as a faithful mathematical copy of the DSP with those UI dependencies removed.

---

## File-by-File Explanation

```
WP/
├── WDEffect_headless.h          ← C++ DSP core (self-contained, no OpenGL)
├── weeping_demon_wrapper.cpp    ← Thin C-linkage API over WDEffect_headless
├── build.bat                    ← Compiles wrapper → weeping_demon_dsp.dll
├── weeping_demon_dsp.py         ← Python: loads DLL or falls back to pure-Python
├── weeping_demon_gui.py         ← Python: OpenCV GUI + dual spectrograms
└── weeping_demon_app.py         ← Python: main entry point (offline + real-time)
```

### `WDEffect_headless.h`
A self-contained C++ header that contains the complete DSP math from the original `WDEffect.cpp`:
- **Potentiometer taper**: The `WD_PotTapers.xlsx` data was already curve-fitted into this formula inside `WDEffect.cpp`:
  ```cpp
  WAH = 1.1 * 1.6933367e10 * pow(WAH_theta, -4.45855 * 0.95)
  ```
  This formula is copied verbatim — the `.xlsx` file itself is not used at runtime.
- **Circuit values**: All resistor/capacitor values (measured, not nominal) from the `WD_Schem/` schematics are hardcoded constants.
- **Bilinear transform**: Converts the analog s-domain transfer function to digital z-domain IIR coefficients each time knobs change.
- **Public setters**: Added `set_knob_wah()`, `set_knob_q()`, etc. so Python can control parameters without the OpenGL Knob objects.

### `weeping_demon_wrapper.cpp`
Exposes four C functions that Python calls via `ctypes`:
```c
void* create_wd_effect(int buffer_length, int sample_rate);
void  destroy_wd_effect(void* handle);
void  set_parameters(void* handle, double wah, q, level, lo, range, int mode);
void  process_audio(void* handle, double* in, double* out, int length);
```

### `weeping_demon_dsp.py`
Two backends, same public API:
- **`_DllWahDSP`**: wraps the compiled DLL via `ctypes` (maximum accuracy, C++ speed)
- **`_PythonWahDSP`**: pure NumPy fallback that ports every line of `WDEffect.cpp` math to Python

The `WahEffect()` factory picks the DLL if it exists, otherwise falls back silently.

### `weeping_demon_gui.py`
Replaces the original `Graphics.cpp` + `Knob.cpp` + `Spectrogram.cpp`:
- OpenCV trackbars for all 6 knobs (Wah, Q, Level, LO, Range, Mode)
- Two scrolling FFT spectrograms: dry (input) on top, wet (wah-filtered) on bottom
- Falls back to a no-op stub if `cv2` is not installed

### `weeping_demon_app.py`
Replaces `WeepingDemon.cpp` + `AudioWrapper.cpp` + `RtAudio.cpp`:
- **Offline mode**: reads WAV block-by-block, applies filter, writes output WAV
- **Real-time mode**: opens a `sounddevice` duplex stream (mic → wah → speaker)

---

## Usage

### Step 1 – Install Python dependencies
```bash
pip install numpy opencv-python sounddevice
```

### Step 2 – (Optional) Build the C++ DLL for maximum accuracy
> Requires MinGW-w64 (`winget install MinGW.MinGW` or MSYS2).
```bat
cd C:\Users\mrdat\PycharmProjects\pan-theory\my_functions\Music\WP
build.bat
```
If `weeping_demon_dsp.dll` is absent, Python automatically uses the pure-Python fallback — no build step needed.

### Step 3 – Run

**Offline WAV processing** (always works, no soundcard needed):
```bash
python weeping_demon_app.py --file guitar.wav
# Output saved to: guitar_wah.wav
```

**Custom output path:**
```bash
python weeping_demon_app.py --file guitar.wav --out output_wah.wav
```

**Real-time mic input** (requires sounddevice + audio driver):
```bash
python weeping_demon_app.py --rt
```

### Knob Reference

| Trackbar | Maps to | Range | Effect |
|---|---|---|---|
| **Wah** | Pedal sweep angle θ | 5 – 17 | Controls the wah filter center frequency via the taper curve |
| **Q** | Resonance factor Q | 100 – 250,000 | Filter sharpness / quack character |
| **Level** | Output gain resistor | 100 – 10,000 | Overall output volume |
| **LO** | Low-pass corner | 100 – 100,000 | Bass bleed-through |
| **Range** | Sweep range | 100 – 2,500 | Width of the wah frequency sweep |
| **Mode** | Bass / Normal | 0 or 1 | 1 = Bass wah (3rd-order), 0 = Normal wah (4th-order) |

### Keyboard shortcuts (GUI window)
| Key | Action |
|---|---|
| `q` or `ESC` | Close and stop |

---

## Architecture Diagram

```
 Original C++ Repo                      Python WP/ Package
 ---------------------                  ------------------------------
 WD_Schem/*.asc         --math-->       WDEffect_headless.h
 WD_PotTapers.xlsx      --fitted-->     (WAH = 1.1*1.693e10*theta^-4.23)
 WDEffect.cpp (DSP)     --port-->       WDEffect_headless.h

 WDEffect.cpp           ----------->    weeping_demon_wrapper.cpp
 Knob.cpp / Graphics    --REMOVED-->    weeping_demon_gui.py (OpenCV)
 RtAudio.cpp            --REMOVED-->    weeping_demon_app.py (sounddevice)
 WeepingDemon.cpp       --REMOVED-->    weeping_demon_app.py (argparse CLI)
```
