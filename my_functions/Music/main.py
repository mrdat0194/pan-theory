import os
import argparse
import time
import torch
import librosa
import numpy as np
import sys
# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from standalone_compare.audio_processor import MusicAudioProcessor

# Constants
SAMPLES_DIR = r"C:\Users\mrdat\PycharmProjects\pan-theory\my_functions\Music\samples"
GROUND_TRUTH_DIR = "Ground Truth (Human Speech)"

def get_processor():
    """Instantiate the ESPnet-inspired Log-Mel processor."""
    return MusicAudioProcessor(
        fs=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        compat_espnet=False,  # Use mathematically correct variance computation
    )

def process_single_file(file_path):
    """Process a single audio file and print its Log-Mel feature shape."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
        
    print(f"Loading '{file_path}'...")
    y, sr = librosa.load(file_path, sr=16000)
    waveform = torch.from_numpy(y).unsqueeze(0)  # Shape: (1, Nsamples)
    ilens = torch.tensor([waveform.size(1)], dtype=torch.long)
    
    processor = get_processor()
    with torch.no_grad():
        features, _ = processor(waveform, ilens)
        
    print(f"Successfully processed!")
    print(f"-> Waveform shape: {waveform.shape}")
    print(f"-> Extracted Log-Mel Spectrogram shape: {features.shape} (Batch, Frames, Mel-Bands)")
    print(f"-> Feature statistics - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
    
    out_path = os.path.splitext(file_path)[0] + "_features.pt"
    torch.save(features, out_path)
    print(f"-> Features saved to '{out_path}'")

def process_folder(folder_name):
    """Batch process all audio files in the specified samples folder."""
    target_path = os.path.join(SAMPLES_DIR, folder_name)
    if not os.path.exists(target_path) or not os.path.isdir(target_path):
        print(f"Error: Folder '{folder_name}' not found under samples path: {SAMPLES_DIR}")
        return
        
    wav_files = [f for f in os.listdir(target_path) if f.lower().endswith(".wav")]
    if not wav_files:
        print(f"No WAV files found in '{target_path}'.")
        return
        
    print(f"Batch processing {len(wav_files)} files in '{folder_name}'...")
    processor = get_processor()
    
    feature_dict = {}
    start_time = time.perf_counter()
    
    for i, file_name in enumerate(wav_files, 1):
        file_path = os.path.join(target_path, file_name)
        try:
            y, sr = librosa.load(file_path, sr=16000)
            waveform = torch.from_numpy(y).unsqueeze(0)
            ilens = torch.tensor([waveform.size(1)], dtype=torch.long)
            
            with torch.no_grad():
                features, _ = processor(waveform, ilens)
            feature_dict[file_name] = features.squeeze(0)  # Shape: (Frames, Mel-Bands)
        except Exception as e:
            print(f"  [{i}/{len(wav_files)}] Failed to process '{file_name}': {e}")
            
    elapsed = time.perf_counter() - start_time
    print(f"Processed {len(feature_dict)} files in {elapsed:.2f} seconds.")
    
    out_bundle_path = os.path.join(target_path, f"{folder_name.replace(' ', '_')}_features.pt")
    torch.save(feature_dict, out_bundle_path)
    print(f"Saved extracted feature bundle to '{out_bundle_path}'")

def evaluate_models():
    """Evaluate all speech synthesis models against the Ground Truth folder."""
    if not os.path.exists(SAMPLES_DIR):
        print(f"Error: Samples directory '{SAMPLES_DIR}' does not exist.")
        return
        
    gt_path = os.path.join(SAMPLES_DIR, GROUND_TRUTH_DIR)
    if not os.path.exists(gt_path):
        print(f"Error: Ground Truth directory '{gt_path}' not found.")
        return
        
    # List all subfolders representing models
    all_dirs = [d for d in os.listdir(SAMPLES_DIR) if os.path.isdir(os.path.join(SAMPLES_DIR, d))]
    model_dirs = [d for d in all_dirs if d != GROUND_TRUTH_DIR]
    
    if not model_dirs:
        print("No other model directories found under samples for evaluation.")
        return
        
    print("=" * 70)
    print("  Speech Synthesis Evaluation Mode: Spectral Distortion Benchmarking  ")
    print("=" * 70)
    print(f"Ground Truth Folder: {GROUND_TRUTH_DIR}")
    print(f"Synthesizer Models found: {len(model_dirs)}\n")
    
    processor = get_processor()
    rankings = []
    
    for model_name in model_dirs:
        model_path = os.path.join(SAMPLES_DIR, model_name)
        model_files = [f for f in os.listdir(model_path) if f.lower().endswith(".wav")]
        
        # Match with ground truth
        matching_files = []
        for f in model_files:
            gt_file_path = os.path.join(gt_path, f)
            if os.path.exists(gt_file_path):
                matching_files.append(f)
                
        if not matching_files:
            print(f"Skipping model '{model_name}': No matching WAV files in Ground Truth.")
            continue
            
        print(f"Evaluating model '{model_name}' using {len(matching_files)} matching files...")
        
        mse_list = []
        for f in matching_files:
            gt_wav_path = os.path.join(gt_path, f)
            model_wav_path = os.path.join(model_path, f)
            
            try:
                # Load audio
                y_gt, _ = librosa.load(gt_wav_path, sr=16000)
                y_mod, _ = librosa.load(model_wav_path, sr=16000)
                
                # Exract features
                wf_gt = torch.from_numpy(y_gt).unsqueeze(0)
                wf_mod = torch.from_numpy(y_mod).unsqueeze(0)
                
                with torch.no_grad():
                    feat_gt, _ = processor(wf_gt, torch.tensor([wf_gt.size(1)]))
                    feat_mod, _ = processor(wf_mod, torch.tensor([wf_mod.size(1)]))
                    
                feat_gt = feat_gt.squeeze(0)  # Shape: (Frames_gt, Mel-Bands)
                feat_mod = feat_mod.squeeze(0)  # Shape: (Frames_mod, Mel-Bands)
                
                # Align lengths (crop to match the shorter one)
                min_frames = min(feat_gt.size(0), feat_mod.size(0))
                feat_gt_aligned = feat_gt[:min_frames]
                feat_mod_aligned = feat_mod[:min_frames]
                
                # Compute Mel Spectrogram MSE
                mse = torch.mean((feat_gt_aligned - feat_mod_aligned) ** 2).item()
                mse_list.append(mse)
            except Exception as e:
                # Ignore failed file parses during batch evaluation
                pass
                
        if mse_list:
            mean_mse = np.mean(mse_list)
            std_mse = np.std(mse_list)
            rankings.append((model_name, len(mse_list), mean_mse, std_mse))
            print(f"  -> Done. Mean Mel Spectrogram MSE: {mean_mse:.5f}\n")
            
    # Sort rankings (lower MSE is better)
    rankings.sort(key=lambda x: x[2])
    
    print("\n" + "=" * 75)
    print("  Rankings Summary (Lower Mel-Spectrogram MSE represents higher quality)")
    print("=" * 75)
    print(f"{'Rank':<5} | {'Model Name':<35} | {'Matches':<8} | {'Mean MSE':<10} | {'Std MSE':<10}")
    print("-" * 75)
    for rank, (name, matches, mean, std) in enumerate(rankings, 1):
        print(f"{rank:<5} | {name:<35} | {matches:<8} | {mean:<10.5f} | {std:<10.5f}")
    print("=" * 75 + "\n")

def run_music_generation():
    """Original MusicGen generation fallback."""
    from audiocraft.models import MusicGen  # noqa: E402
    print("Loading Facebook MusicGen Small model...")
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    model.set_generation_params(duration=10)
    
    print("Generating music...")
    results = model.generate(['classical rock'])
    sampling_rate = model.sample_rate
    
    wav_filename = "rock.wav"
    print(f"Saving generated audio to {wav_filename}...")
    write(wav_filename, sampling_rate, (results[0].numpy()).T)
    
    # Process features
    process_single_file(wav_filename)

def main():
    parser = argparse.ArgumentParser(description="ESPnet-Inspired Audio Processing & Evaluation Utility")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true", help="Original fallback music generation mode")
    group.add_argument("--file", type=str, help="Process a single WAV file")
    group.add_argument("--folder", type=str, help="Batch process a folder of WAV files in samples/")
    group.add_argument("--eval", action="store_true", help="Evaluate synthesizer models against ground truth")
    
    args = parser.parse_args()
    
    if args.generate:
        run_music_generation()
    elif args.file:
        process_single_file(args.file)
    elif args.folder:
        process_folder(args.folder)
    elif args.eval:
        evaluate_models()

if __name__ == "__main__":
    main()