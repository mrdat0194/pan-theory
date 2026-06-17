/*
 * weeping_demon_wrapper.cpp
 *
 * Exposes the WDEffect DSP core as a flat C API so Python can load
 * this DLL via ctypes.
 *
 * This file includes WDEffect_headless.h (in the same WP/ directory)
 * rather than the original WDEffect.h.  The headless header contains
 * the full DSP math but removes all OpenGL/Knob/Spectrogram dependencies.
 *
 * Build with:  build.bat   (Windows / MinGW-w64)
 */

#include "WDEffect_headless.h"   // self-contained DSP core

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/** Allocate a WDEffect instance.
 *  buffer_length : samples per processing block (512 recommended)
 *  sample_rate   : Hz (44100 recommended)
 *  Returns an opaque handle; must be freed with destroy_wd_effect(). */
EXPORT void* create_wd_effect(int buffer_length, int sample_rate) {
    return new WDEffect(buffer_length, sample_rate);
}

/** Free a previously allocated WDEffect. */
EXPORT void destroy_wd_effect(void* handle) {
    delete static_cast<WDEffect*>(handle);
}

// ---------------------------------------------------------------------------
// Parameter control  (mirrors WDEffect::check_knobs / knob ranges)
// ---------------------------------------------------------------------------

/** Set all six knob values at once.
 *
 *  wah   : pedal sweep angle [5 .. 17]          (mapped to WAH resistance)
 *  q     : resonance / Q factor [100 .. 250000]
 *  level : output gain resistor [100 .. 10000]
 *  lo    : low-pass corner [100 .. 100000]
 *  range : sweep range [100 .. 2500]
 *  mode  : 0 = normal (treble), 1 = bass
 */
EXPORT void set_parameters(void* handle,
                            double wah, double q, double level,
                            double lo,  double range, int mode) {
    WDEffect* wd = static_cast<WDEffect*>(handle);
    wd->set_knob_wah(wah);
    wd->set_knob_q(q);
    wd->set_knob_level(level);
    wd->set_knob_lo(lo);
    wd->set_knob_range(range);
    wd->set_knob_bass(mode ? 1.0 : 0.0);
}

// ---------------------------------------------------------------------------
// Audio processing
// ---------------------------------------------------------------------------

/** Process a block of audio in-place.
 *  input  : pointer to [length] doubles of dry signal  (-1..+1 range)
 *  output : pointer to [length] doubles for wet signal (caller-allocated)
 *  length : number of samples (must match the buffer_length used at create)
 */
EXPORT void process_audio(void* handle,
                           const double* input, double* output, int length) {
    WDEffect* wd = static_cast<WDEffect*>(handle);

    // WDEffect works with a mutable internal buffer
    double* buf = new double[length];
    for (int i = 0; i < length; ++i) buf[i] = input[i];

    wd->receive_buffer(length, buf);
    wd->process_buffer();
    wd->return_buffer(length, output);

    delete[] buf;
}

} // extern "C"
