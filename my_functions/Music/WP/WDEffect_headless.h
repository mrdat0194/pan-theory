/*
 * WDEffect_headless.h
 * ═══════════════════
 * Drop-in replacement header for WDEffect.h used when compiling the
 * Python DLL (USE_VISUALS=0 / no OpenGL / no Knob / no Spectrogram).
 *
 * Differences from the original WDEffect.h:
 *  - Removes all Knob*, Spectrogram*, Graphics* members
 *  - Adds public setters so weeping_demon_wrapper.cpp can configure
 *    knob values directly without going through OpenGL UI objects
 *  - Removes RtAudio / FFT includes (not needed for bare DSP)
 *
 * Author: auto-generated shim for the Python wrapper project
 */

#ifndef _WDEFFECT_HEADLESS_H_
#define _WDEFFECT_HEADLESS_H_

#include <stdio.h>
#include <math.h>
#include <string.h>   // memcpy

// ──────────────────────────────────────────────────────────────
//  Headless WDEffect – DSP core only
// ──────────────────────────────────────────────────────────────

class WDEffect {
public:

    WDEffect(int buffer_length, int sample_rate)
        : buffer_length_(buffer_length),
          sample_rate_(sample_rate),
          bass_(1)
    {
        buffer_ = new double[buffer_length_];

        for (int i = 0; i < 4; ++i) {
            a_[i] = b_[i] = past_a_[i] = past_b_[i] = 0.0;
            x_past_[i] = y_past_[i] = 0.0;
        }

        // Default knob values (same defaults as original Knob constructors)
        // Knob(min, max, default_fraction) → default = min + frac*(max-min)
        WAH_theta = 5.0  + 0.3 * (17.0  - 5.0);    // ~8.6
        Q         = 100.0 + 0.3 * (250e3 - 100.0);  // ~75100
        LEVEL     = 100.0 + 0.3 * (10e3  - 100.0);  // ~3070
        LO        = 100.0 + 0.3 * (100e3 - 100.0);  // ~30070
        RANGE     = 100.0 + 0.3 * (2500.0- 100.0);  // ~820

        calculate_coefficients(a_, b_);
    }

    ~WDEffect() { delete[] buffer_; }

    // ------------------------------------------------------------------
    // Public setters (replaces Knob UI in original code)
    // ------------------------------------------------------------------
    void set_knob_wah  (double v) { WAH_theta = v; }
    void set_knob_q    (double v) { Q         = v; }
    void set_knob_level(double v) { LEVEL     = v; }
    void set_knob_lo   (double v) { LO        = v; }
    void set_knob_range(double v) { RANGE     = v; }
    void set_knob_bass (double v) { bass_     = (v >= 0.5) ? 1 : 0; }

    // ------------------------------------------------------------------
    // Audio pipeline (same interface as original AudioModule)
    // ------------------------------------------------------------------
    void receive_buffer(int length, double* buf) {
        buffer_ = buf;
    }

    void process_buffer() {
        // Refresh coefficients from current knob values
        memcpy(past_a_, a_, sizeof(double) * 4);
        memcpy(past_b_, b_, sizeof(double) * 4);
        calculate_coefficients(a_, b_);

        for (int i = 0; i < buffer_length_; ++i) {
            buffer_[i] = filter_tick(buffer_[i]);
        }
    }

    void return_buffer(int length, double* out) {
        for (int i = 0; i < length; ++i) {
            out[i] = buffer_[i];
            if (buffer_[i] != buffer_[i])  { printf("NaN!\n");      return; }
            if (buffer_[i] >15||buffer_[i]<-15) { printf("OVERFLOW\n"); return; }
        }
    }

    int          get_buffer_length() { return buffer_length_; }
    unsigned int get_sample_rate()   { return (unsigned int)sample_rate_; }

    // ------------------------------------------------------------------
    // DSP internals (identical to original WDEffect.cpp)
    // ------------------------------------------------------------------
    void calculate_coefficients(double* a_, double* b_) {
        double R108=9.79e3,  R109=21.3e3,  R110=23.9e3, R111=46.4e3;
        double R113=197.0e3, R114=10.97e3, R115=982.0,  R117=14.77e3;
        double R120=3.52e3,  R122=4.95e3,  R123=4.65e3;
        double C104=2.76e-9, C105=10.3e-9, C118=19.5e-9,C119=10.1e-9;
        double VR6=0.0,      VR7=449e3;

        double m_fudge=1.1, e_fudge=0.95;
        double theta = (WAH_theta > 1e-9) ? WAH_theta : 1e-9;
        double WAH   = m_fudge * 1.6933367e10 * pow(theta, -4.45855 * e_fudge);

        double x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17;
        double B2,B1,B0,A3,A2,A1,A0;

        if (bass_) {
            x0=LO+R122+R123; x1=C105*R117*x0; x2=Q+R114;
            x3=R109+R110;    x4=R113+VR7;     x5=WAH+x4;
            x6=R108+x2;      x7=R115+RANGE+VR6;
            B2=0.0; A3=0.0;
            B1=LEVEL*x1*x2*x3*x5;
            B0=R123*x2*x3*x5*(LEVEL+R120);
            A2=R109*R120*x1*x6*(C104+C119)*(WAH*(x4+x7)+x4*x7);
            A1=C105*R108*R117*R120*x0*x3*x5;
            A0=R110*R120*x0*x5*x6;
        } else {
            x0=LO+R122+R123;   x1=C105*R111*R117*x0; x2=Q+R114;
            x3=R109+R110;      x4=R113+VR7;           x5=WAH+x4;
            x6=x2*x3*x5;       x7=C118*R111*R123;     x8=C105*R117;
            x9=x0*x8;          x10=R108+x2;           x11=R115+RANGE+VR6;
            x12=WAH*(x11+x4)+x11*x4; x13=R110*R111;   x14=C118*R108*x5;
            x15=C104*x12;      x16=R120*x0*x5;        x17=R108+R111;
            B2=C118*LEVEL*x1*x2*x3*x5;
            B1=x6*(LEVEL*(x7+x9)+R120*x7);
            B0=R123*x6*(LEVEL+R120);
            A3=C104*C118*R109*R120*x1*x10*x12;
            A2=R120*x9*(R109*(R111*(x14+x15)+x10*x15)+x13*x14);
            A1=x16*(C118*x10*x13+x17*x3*x8);
            A0=R110*x16*(x17+x2);
        }

        double f=sample_rate_, f2=f*f, f3=f2*f;
        double d = 8*A3*f3 + 4*A2*f2 + 2*A1*f + A0;
        if (fabs(d) < 1e-300) return;

        a_[0] = 1.0;
        a_[1] = (-24*A3*f3 - 4*A2*f2 + 2*A1*f + 3*A0) / d;
        a_[2] = ( 24*A3*f3 - 4*A2*f2 - 2*A1*f + 3*A0) / d;
        a_[3] = ( -8*A3*f3 + 4*A2*f2 - 2*A1*f +   A0) / d;
        b_[0] = ( 4*B2*f2 + 2*B1*f +   B0) / d;
        b_[1] = (-4*B2*f2 + 2*B1*f + 3*B0) / d;
        b_[2] = (-4*B2*f2 - 2*B1*f + 3*B0) / d;
        b_[3] = ( 4*B2*f2 - 2*B1*f +   B0) / d;
    }

    double filter_tick(double in) {
        x_past_[3]=x_past_[2]; x_past_[2]=x_past_[1]; x_past_[1]=x_past_[0];
        x_past_[0]=in;

        double y = b_[0]*x_past_[0]+b_[1]*x_past_[1]+b_[2]*x_past_[2]+b_[3]*x_past_[3]
                  -a_[1]*y_past_[1]-a_[2]*y_past_[2]-a_[3]*y_past_[3];

        y_past_[3]=y_past_[2]; y_past_[2]=y_past_[1]; y_past_[1]=y_past_[0];
        y_past_[0]=y;
        return y;
    }

private:
    int    buffer_length_;
    double sample_rate_;
    double* buffer_;
    int    bass_;

    double a_[4], b_[4], past_a_[4], past_b_[4];
    double x_past_[4], y_past_[4];

    double WAH_theta, Q, LEVEL, LO, RANGE;
};

#endif // _WDEFFECT_HEADLESS_H_
