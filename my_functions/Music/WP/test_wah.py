"""
test_wah.py  â€“  Self-contained test for the Weeping Demon Wah emulator.

Runs in three phases without any microphone or soundcard:
  1. Unit test  â€“ verify the pure-Python DSP produces plausible output
  2. WAV test   â€“ generate a synthetic guitar-like tone, apply the wah, save result
  3. GUI test   â€“ open the OpenCV control panel for 5 seconds (visual check)

Run from the WP/ directory:
    python test_wah.py
"""

import os, sys, wave, struct, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Phase 0 â€“ import check
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("Phase 0 â€“ Import check")
print("=" * 60)

try:
    from weeping_demon_dsp import WahEffect, _try_load_dll
    print("  âœ“  weeping_demon_dsp imported")
except Exception as e:
    print(f"  âœ—  FAILED to import weeping_demon_dsp: {e}")
    sys.exit(1)

try:
    from weeping_demon_gui import WahGUI, _CV2_OK
    print(f"  âœ“  weeping_demon_gui imported  (cv2 available: {_CV2_OK})")
except Exception as e:
    print(f"  âœ—  FAILED to import weeping_demon_gui: {e}")

dll_loaded = _try_load_dll() is not None
print(f"  {'âœ“' if dll_loaded else 'â—‹'}  DLL backend: {'loaded' if dll_loaded else 'not found â€“ using pure-Python fallback'}")
print()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Phase 1 â€“ DSP unit tests
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("Phase 1 â€“ DSP unit tests (pure-Python fallback)")
print("=" * 60)

SAMPLE_RATE = 44100
BUFFER_LEN  = 512

wd = WahEffect(buffer_length=BUFFER_LEN, sample_rate=SAMPLE_RATE)

def test(name, condition):
    status = "âœ“  PASS" if condition else "âœ—  FAIL"
    print(f"  {status}  {name}")
    return condition

all_pass = True

# 1a. Silent input â†’ near-silent output
silence = np.zeros(BUFFER_LEN)
out_silence = wd.process(silence)
all_pass &= test("Silent input produces near-zero output",
                 np.max(np.abs(out_silence)) < 0.01)

# 1b. Output length matches input length
tone = np.sin(2 * np.pi * 440 * np.arange(BUFFER_LEN) / SAMPLE_RATE) * 0.5
out_tone = wd.process(tone)
all_pass &= test("Output length == input length", len(out_tone) == len(tone))

# 1c. Filter actually changes the signal (not a pass-through)
all_pass &= test("Wah filter modifies the signal",
                 not np.allclose(tone, out_tone, atol=1e-6))

# 1d. Wah sweep changes output
wd.set_parameters(wah=5,  q=5000, level=3000, lo=50000, rang=500, mode=1)
out_low  = wd.process(tone.copy())
wd.set_parameters(wah=17, q=5000, level=3000, lo=50000, rang=500, mode=1)
out_high = wd.process(tone.copy())
all_pass &= test("Different Wah positions give different outputs",
                 not np.allclose(out_low, out_high, atol=1e-6))

# 1e. Mode toggle doesn't crash
wd.set_parameters(wah=10, q=5000, level=3000, lo=50000, rang=500, mode=0)
out_normal = wd.process(tone.copy())
wd.set_parameters(wah=10, q=5000, level=3000, lo=50000, rang=500, mode=1)
out_bass   = wd.process(tone.copy())
all_pass &= test("Normal mode (0) and Bass mode (1) produce different results",
                 not np.allclose(out_normal, out_bass, atol=1e-6))

# 1f. No NaN or Inf in output
all_pass &= test("No NaN/Inf in output", np.all(np.isfinite(out_tone)))

print()
print(f"  Phase 1 result: {'ALL PASS âœ“' if all_pass else 'SOME FAILURES âœ—'}")
print()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Phase 2 â€“ Offline WAV round-trip
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("Phase 2 â€“ Synthetic WAV â†’ Wah filter â†’ output WAV")
print("=" * 60)

DURATION  = 3.0   # seconds
N_SAMPLES = int(DURATION * SAMPLE_RATE)

# Generate a simple chord-like test tone (root + 3rd + 5th overtones)
t = np.arange(N_SAMPLES) / SAMPLE_RATE
signal = (
    0.3 * np.sin(2 * np.pi * 110 * t) +   # A2
    0.2 * np.sin(2 * np.pi * 165 * t) +   # E3
    0.15* np.sin(2 * np.pi * 220 * t) +   # A3
    0.1 * np.sin(2 * np.pi * 330 * t)     # E4
).astype(np.float64)

INPUT_WAV  = os.path.join(os.path.dirname(__file__), "test_input.wav")
OUTPUT_WAV = os.path.join(os.path.dirname(__file__), "test_output_wah.wav")

def write_wav_mono(path, samples, sr):
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1);  wf.setsampwidth(2)
        wf.setframerate(sr); wf.writeframes(pcm.tobytes())

write_wav_mono(INPUT_WAV, signal, SAMPLE_RATE)
print(f"  Wrote synthetic input : {INPUT_WAV}")

# Process block-by-block with a slow wah sweep
wd2 = WahEffect(buffer_length=BUFFER_LEN, sample_rate=SAMPLE_RATE)
n_blocks = N_SAMPLES // BUFFER_LEN
output   = np.zeros(n_blocks * BUFFER_LEN)

for i in range(n_blocks):
    # Slowly sweep Wah from 5 â†’ 17 across the duration
    wah_pos = 5.0 + 12.0 * (i / n_blocks)
    wd2.set_parameters(wah=wah_pos, q=5000, level=3000, lo=50000, rang=800, mode=1)
    chunk = signal[i * BUFFER_LEN:(i + 1) * BUFFER_LEN]
    output[i * BUFFER_LEN:(i + 1) * BUFFER_LEN] = wd2.process(chunk)

write_wav_mono(OUTPUT_WAV, output, SAMPLE_RATE)
print(f"  Wrote processed output: {OUTPUT_WAV}")

# Verify file exists and has content
size = os.path.getsize(OUTPUT_WAV)
print(f"  Output WAV size: {size:,} bytes")
test("Output WAV created and non-empty", size > 1000)

rms_in  = np.sqrt(np.mean(signal[:len(output)]**2))
rms_out = np.sqrt(np.mean(output**2))
print(f"  Input RMS : {rms_in:.4f}")
print(f"  Output RMS: {rms_out:.4f}")
test("Output has energy (not all zeros)", rms_out > 1e-6)
print()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Phase 3 â€“ GUI smoke test (5 second window)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("Phase 3 â€“ OpenCV GUI smoke test (auto-closes after 5 seconds)")
print("           Adjust any trackbar to see spectrograms update.")
print("           Press q or ESC to close early.")
print("=" * 60)

try:
    from weeping_demon_gui import WahGUI
    gui = WahGUI()
    gui.start()

    wd3 = WahEffect(buffer_length=BUFFER_LEN, sample_rate=SAMPLE_RATE)

    t_start = time.perf_counter()
    frame   = 0

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= 5.0:
            print("  5 seconds elapsed â€“ closing GUI automatically.")
            break
        if not gui.is_open():
            print("  GUI closed by user.")
            break

        # Feed live synthesised audio so spectrogram scrolls
        t_vec = np.arange(BUFFER_LEN) / SAMPLE_RATE + elapsed
        chunk = (0.3 * np.sin(2*np.pi*110*t_vec) +
                 0.2 * np.sin(2*np.pi*220*t_vec)).astype(np.float64)

        params = gui.get_params()
        wd3.set_parameters(
            wah=params["Wah"], q=params["Q"], level=params["Level"],
            lo=params["LO"],   rang=params["Range"], mode=int(params["Mode"]))
        wet = wd3.process(chunk)

        gui.push_audio(chunk, wet)
        if not gui.tick():
            break

        frame += 1

    gui.stop()
    print("  âœ“  GUI phase complete")
except Exception as e:
    print(f"  â—‹  GUI skipped: {e}")

print()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Summary
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("=" * 60)
print("DONE.")
print(f"  Input WAV  : {INPUT_WAV}")
print(f"  Output WAV : {OUTPUT_WAV}")
print()
print("  To hear the result in any media player:")
print(f"    start {OUTPUT_WAV}")
print()
print("  To run the full interactive app:")
print(f"    python weeping_demon_app.py --file test_input.wav")
print("=" * 60)

