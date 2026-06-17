import torch
import librosa
from audiocraft.models import MusicGen  # noqa: E402
from scipy.io.wavfile import write  # noqa: E402
from audio_processor import MusicAudioProcessor

# 1. Generate music using AudioCraft's MusicGen
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=10)

print("Generating music...")
results = model.generate(['classical rock'])
sampling_rate = model.sample_rate

wav_filename = "rock.wav"
print(f"Saving generated audio to {wav_filename}...")
write(wav_filename, sampling_rate, (results[0].numpy()).T)

# 2. Load the generated audio and process it using our ESPnet-inspired pipeline
print("\nProcessing generated audio...")
# Load waveform (resampling to 16000 Hz as expected by speech/audio processors)
y, sr = librosa.load(wav_filename, sr=16000)
waveform = torch.from_numpy(y).unsqueeze(0)  # Shape: (Batch=1, Nsamples)
ilens = torch.tensor([waveform.size(1)], dtype=torch.long)

# Instantiate processor
processor = MusicAudioProcessor(
    fs=16000,
    n_fft=512,
    win_length=400,
    hop_length=160,
    n_mels=80,
    compat_espnet=False,  # Use mathematically correct variance computation
)

# Extract features
with torch.no_grad():
    features, olens = processor(waveform, ilens)

print(f"Processed features shape: {features.shape} (Batch, Frames, Mel-Bands)")
feature_filename = "rock_features.pt"
print(f"Saving extracted feature tensor to {feature_filename}...")
torch.save(features, feature_filename)
print("Finished successfully!")