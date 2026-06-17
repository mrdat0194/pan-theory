"""
weeping_demon_app.py
====================
Main entry point for the Weeping Demon Wah Pedal emulator.

Modes
-----
  python weeping_demon_app.py --file   guitar.wav   # offline WAV → WAV
  python weeping_demon_app.py --rt                  # real-time mic input

The --file mode always works; --rt requires sounddevice to be installed.

Keyboard shortcuts (when GUI window is open)
  q / ESC  – quit
"""

import argparse
import sys
import os
import wave
import struct
import threading
import time
import numpy as np

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from weeping_demon_dsp import WahEffect
from weeping_demon_gui import WahGUI, _NoOpGUI

# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────

BUFFER_LEN  = 512
SAMPLE_RATE = 44100
GUI_FPS     = 30         # target GUI refresh rate
GUI_PERIOD  = 1.0 / GUI_FPS


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def _read_wav_mono(path: str):
    """Read a WAV file and return (samples_float64, sample_rate)."""
    with wave.open(path, 'rb') as wf:
        sr       = wf.getframerate()
        n_ch     = wf.getnchannels()
        sampw    = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw      = wf.readframes(n_frames)

    fmt = {1: 'b', 2: '<h', 4: '<i'}.get(sampw)
    if fmt is None:
        raise ValueError(f"Unsupported sample width: {sampw}")

    samples = np.array(struct.unpack(f'{n_frames * n_ch}{fmt}', raw), dtype=np.float64)
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)   # downmix to mono

    peak = float(2 ** (sampw * 8 - 1))
    return samples / peak, sr


def _write_wav_mono(path: str, samples: np.ndarray, sample_rate: int):
    """Write float64 mono samples to a 16-bit WAV file."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm     = (clipped * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _apply_params(wd, params: dict):
    wd.set_parameters(
        wah   = params["Wah"],
        q     = params["Q"],
        level = params["Level"],
        lo    = params["LO"],
        rang  = params["Range"],
        mode  = int(params["Mode"]),
    )


# ──────────────────────────────────────────────────────────────
#  Offline WAV processing
# ──────────────────────────────────────────────────────────────

def run_offline(input_path: str, output_path: str):
    print(f"[Offline] Reading  : {input_path}")
    samples, sr = _read_wav_mono(input_path)
    print(f"[Offline] {len(samples)} samples @ {sr} Hz  ({len(samples)/sr:.1f}s)")

    wd  = WahEffect(BUFFER_LEN, SAMPLE_RATE)
    gui = WahGUI()
    gui.start()

    n_blocks = len(samples) // BUFFER_LEN
    output   = np.zeros(n_blocks * BUFFER_LEN, dtype=np.float64)

    t_last_gui = time.perf_counter()

    for i in range(n_blocks):
        params = gui.get_params()
        _apply_params(wd, params)

        chunk = samples[i * BUFFER_LEN:(i + 1) * BUFFER_LEN]
        wet   = wd.process(chunk)
        output[i * BUFFER_LEN:(i + 1) * BUFFER_LEN] = wet

        gui.push_audio(chunk, wet)

        now = time.perf_counter()
        if now - t_last_gui >= GUI_PERIOD:
            if not gui.tick():
                print("[Offline] GUI closed early – writing partial output.")
                output = output[:( i + 1) * BUFFER_LEN]
                break
            t_last_gui = now

        if i % 100 == 0:
            pct = 100 * i / n_blocks
            print(f"\r[Offline] Progress: {pct:5.1f}%", end="", flush=True)

    gui.stop()
    print()
    _write_wav_mono(output_path, output, sr)
    print(f"[Offline] Saved to : {output_path}")


# ──────────────────────────────────────────────────────────────
#  Real-time streaming
# ──────────────────────────────────────────────────────────────

def run_realtime():
    try:
        import sounddevice as sd
    except ImportError:
        print("[RT] sounddevice not installed.  Run:  pip install sounddevice")
        sys.exit(1)

    wd  = WahEffect(BUFFER_LEN, SAMPLE_RATE)
    gui = WahGUI()
    gui.start()

    _params = gui.get_params()
    _lock   = threading.Lock()

    def callback(indata, outdata, frames, time_info, status):
        nonlocal _params
        with _lock:
            p = dict(_params)
        _apply_params(wd, p)

        mono = indata[:, 0].astype(np.float64)
        wet  = wd.process(mono)
        outdata[:, 0] = wet
        if outdata.shape[1] > 1:
            outdata[:, 1] = wet

        gui.push_audio(mono, wet)

    print(f"[RT] Opening audio stream: {SAMPLE_RATE} Hz, block={BUFFER_LEN}")
    with sd.Stream(samplerate=SAMPLE_RATE,
                   blocksize=BUFFER_LEN,
                   dtype='float64',
                   channels=1,
                   callback=callback):
        print("[RT] Stream running. Adjust sliders, press q/ESC to quit.")
        while gui.is_open():
            with _lock:
                _params = gui.get_params()
            if not gui.tick():
                break
            time.sleep(GUI_PERIOD)

    gui.stop()
    print("[RT] Done.")


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Weeping Demon Wah Pedal Emulator"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", metavar="INPUT.wav",
                       help="Offline mode: process INPUT.wav and save result")
    group.add_argument("--rt", action="store_true",
                       help="Real-time mode: process mic input via sounddevice")
    parser.add_argument("--out", metavar="OUTPUT.wav", default=None,
                        help="Output path for --file mode (default: <input>_wah.wav)")

    args = parser.parse_args()

    if args.file:
        out = args.out or os.path.splitext(args.file)[0] + "_wah.wav"
        run_offline(args.file, out)
    elif args.rt:
        run_realtime()


if __name__ == "__main__":
    main()
