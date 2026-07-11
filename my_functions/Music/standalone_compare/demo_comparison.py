import torch
import librosa
import time
import sys
import os

# Add the current folder to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from audio_processor import MusicAudioProcessor
from espnet_processor import ESPnetAudioProcessor

def run_comparison():
    print("=" * 60)
    print("  ESPnet Standalone Audio Processing Comparison (Using Real Voice Sample) ")
    print("=" * 60)
    
    # 1. Load the copied sample.wav file
    wav_path = os.path.join(current_dir, "sample.wav")
    if not os.path.exists(wav_path):
        print(f"Error: Could not find sample file at {wav_path}")
        return
        
    fs = 16000
    y, sr = librosa.load(wav_path, sr=fs)
    print(f"Loaded sample voice file successfully.")
    print(f"Original shape: {y.shape}, sample rate: {sr} Hz")
    
    # Create batched sequences with different lengths (testing mask padding logic)
    waveform_1 = torch.from_numpy(y)
    len_1 = waveform_1.size(0)
    
    # Make second waveform shorter (80% of length) and pad it with zeros to length_1
    len_2 = int(len_1 * 0.8)
    waveform_2 = torch.zeros_like(waveform_1)
    waveform_2[:len_2] = waveform_1[:len_2]
    
    # Batch tensor: shape (2, Nsamples)
    waveforms = torch.stack([waveform_1, waveform_2])
    ilens = torch.tensor([len_1, len_2], dtype=torch.long)
    
    print(f"Batch waveforms shape: {waveforms.shape}")
    print(f"Lengths for evaluation: {ilens.tolist()}\n")
    
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
    
    print("Instantiating Custom Processor (compatibility mode = True)...")
    custom_processor = MusicAudioProcessor(compat_espnet=True, **params)
    custom_processor.eval()  # Disable SpecAugment for exact comparison
    
    print("Instantiating Official ESPnet Processor...")
    espnet_processor = ESPnetAudioProcessor(**params)
    espnet_processor.eval()
    
    # 3. Step-by-Step comparison
    print("\n--- STEP-BY-STEP TRACE AND COMPARISON ---")
    with torch.no_grad():
        # --- Step 3a: STFT ---
        custom_stft, custom_olens = custom_processor.stft(waveforms, ilens)
        
        espnet_stft_raw, espnet_olens = espnet_processor.frontend.stft(waveforms, ilens)
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
        
        espnet_logmel, _ = espnet_processor.frontend.logmel(espnet_power, espnet_olens)
        logmel_diff = torch.abs(custom_logmel - espnet_logmel).max().item()
        print(f"LogMel Spec Maximum Difference: {logmel_diff:.8e}")
        
        # --- Step 3d: MVN Normalization (ESPnet Compatibility Mode) ---
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
    assert custom_feats.shape == espnet_feats.shape, "Shape mismatch between processors!"
    
    diff = torch.abs(custom_feats - espnet_feats)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print("\n" + "=" * 60)
    print("  Numerical Comparison Results")
    print("=" * 60)
    print(f"Maximum absolute difference: {max_diff:.8e}")
    print(f"Mean absolute difference:    {mean_diff:.8e}")
    
    tolerance = 1e-5
    if max_diff < tolerance:
        print("\nSUCCESS: Custom PyTorch implementation is mathematically equivalent to official ESPnet!")
    else:
        print(f"\nWARNING: Difference exceeds tolerance ({tolerance}).")
    print("=" * 60)

if __name__ == "__main__":
    run_comparison()
