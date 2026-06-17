import torch
import time
import sys
import os

# Add the parent directory to Python path if running directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_processor import MusicAudioProcessor
from espnet_processor import ESPnetAudioProcessor

def run_comparison():
    print("=" * 60)
    print("  ESPnet-Inspired vs. Official ESPnet Audio Processing Comparison  ")
    print("=" * 60)
    
    # 1. Generate a test waveform (e.g., 5 seconds of 440Hz sine wave mixed with some noise)
    fs = 16000
    duration = 5.0
    t = torch.linspace(0, duration, int(fs * duration))
    # Batch size of 2 waveforms
    waveform_1 = 0.5 * torch.sin(2 * torch.pi * 440 * t) + 0.1 * torch.randn_like(t)
    waveform_2 = 0.3 * torch.sin(2 * torch.pi * 880 * t) + 0.1 * torch.randn_like(t)
    waveforms = torch.stack([waveform_1, waveform_2]) # Shape: (2, 80000)
    
    # Varying lengths for batch testing (e.g., second waveform is shorter)
    ilens = torch.tensor([80000, 64000], dtype=torch.long)
    # Zero out padded part of second waveform
    waveforms[1, 64000:] = 0.0
    
    print(f"Generated test batch of {waveforms.size(0)} waveforms.")
    print(f"Waveform shape: {waveforms.shape}")
    print(f"Sample rate: {fs} Hz, Duration: {duration} seconds")
    print(f"Sequence lengths: {ilens.tolist()}\n")
    
    # 2. Instantiate both processors with matching parameters
    params = {
        "fs": fs,
        "n_fft": 512,
        "win_length": 400,
        "hop_length": 160,
        "n_mels": 80,
        "fmin": 0,
        "fmax": 8000,
        "norm_means": True,
        "norm_vars": True,
    }
    
    print("Instantiating Custom Processor...")
    custom_processor = MusicAudioProcessor(**params)
    custom_processor.eval()  # Disable spec augment/dropout for exact comparison
    
    print("Instantiating Official ESPnet Processor...")
    espnet_processor = ESPnetAudioProcessor(**params)
    espnet_processor.eval()
    
    # 3. Step-by-Step comparison
    print("\n--- STEP-BY-STEP TRACE AND COMPARISON ---")
    with torch.no_grad():
        # --- Step 3a: STFT ---
        custom_stft, custom_olens = custom_processor.stft(waveforms, ilens)
        
        # ESPnet STFT (via stft module)
        espnet_stft_raw, espnet_olens = espnet_processor.frontend.stft(waveforms, ilens)
        # Convert to PyTorch complex matching our format
        # espnet_stft_raw shape: (Batch, Frames, Freq, 2)
        espnet_stft = torch.complex(espnet_stft_raw[..., 0], espnet_stft_raw[..., 1])
        
        stft_diff = torch.abs(custom_stft - espnet_stft).max().item()
        print(f"STFT Maximum Difference: {stft_diff:.8e}")
        
        # --- Step 3b: Power Spectrum ---
        custom_power = custom_stft.real**2 + custom_stft.imag**2
        espnet_power = espnet_stft.real**2 + espnet_stft.imag**2
        power_diff = torch.abs(custom_power - espnet_power).max().item()
        print(f"Power Spec Maximum Difference: {power_diff:.8e}")
        
        # --- Step 3c: Log-Mel Spectrogram ---
        custom_logmel, _ = custom_processor.logmel(custom_power, custom_olens)
        
        # ESPnet LogMel
        espnet_logmel, _ = espnet_processor.frontend.logmel(espnet_power, espnet_olens)
        logmel_diff = torch.abs(custom_logmel - espnet_logmel).max().item()
        print(f"LogMel Spec Maximum Difference: {logmel_diff:.8e}")
        
        # --- Step 3d: MVN Normalization ---
        custom_mvn, _ = custom_processor.mvn(custom_logmel, custom_olens)
        espnet_mvn, _ = espnet_processor.mvn(espnet_logmel, espnet_olens)
        mvn_diff = torch.abs(custom_mvn - espnet_mvn).max().item()
        print(f"MVN Spec Maximum Difference: {mvn_diff:.8e}")

    # 4. Benchmark Custom Processor
    print("\nRunning Custom Processor...")
    start_time = time.perf_counter()
    with torch.no_grad():
        custom_feats, custom_lens = custom_processor(waveforms, ilens)
    custom_time = time.perf_counter() - start_time
    print(f"-> Custom processing time: {custom_time:.5f} seconds")
    
    # 5. Benchmark ESPnet Processor
    print("\nRunning Official ESPnet Processor...")
    start_time = time.perf_counter()
    with torch.no_grad():
        espnet_feats, espnet_lens = espnet_processor(waveforms, ilens)
    espnet_time = time.perf_counter() - start_time
    print(f"-> ESPnet processing time: {espnet_time:.5f} seconds")
    
    # 6. Numerical Equivalence Check
    # Ensure they have identical output sizes
    assert custom_feats.shape == espnet_feats.shape, "Shape mismatch between processors!"
    
    # Calculate difference
    diff = torch.abs(custom_feats - espnet_feats)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print("\n" + "=" * 60)
    print("  Numerical Comparison Results")
    print("=" * 60)
    print(f"Maximum absolute difference: {max_diff:.8e}")
    print(f"Mean absolute difference:    {mean_diff:.8e}")
    
    # Tolerance check
    tolerance = 1e-5
    if max_diff < tolerance:
        print("\nSUCCESS: Custom PyTorch implementation is mathematically equivalent to official ESPnet!")
    else:
        print(f"\nWARNING: Difference exceeds tolerance ({tolerance}). Inspect implementation details.")
        
    print("\n" + "=" * 60)
    print("  Pros and Cons Comparison")
    print("=" * 60)
    print("""
1. Native Custom PyTorch Implementation:
   [Pros]
   - Fully portable: Does not require installing ESPnet, which is extremely heavy and has deep, complex dependencies.
   - Zero-dependency: Only relies on torch, numpy, and librosa.
   - Traceability: Easier to export using torch.jit or ONNX for production deployment.
   - Readability: Clean, compact code (~200 lines) that is easy to customize or modify.
   [Cons]
   - Must be manually maintained if there are bugs or updates in the DSP logic.
   
2. Official ESPnet Import:
   [Pros]
   - Robustness: Extensively tested in research and production.
   - Updates: Automatically benefits from upstream ESPnet improvements and bugs fixes.
   - Richness: Full access to secondary modules (e.g. beamforming, WPE dereverberation, neural-network enhancement, complex multi-channel frontends).
   [Cons]
   - Heavy dependencies: Highly coupled with custom external tools (typeguard, humanfriendly, config wrappers).
   - High overhead: Includes initialization check overhead and imports hundreds of modules.
   - Difficult deployment: Complex to cross-compile or run in standalone edge devices.
    """)
    print("=" * 60)

if __name__ == "__main__":
    run_comparison()
