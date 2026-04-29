from audiocraft.models import MusicGen  # noqa: E402

model = MusicGen.get_pretrained('facebook/musicgen-small')

model.set_generation_params(duration=10)

print("Generating music...")
results = model.generate(['classical rock'])
sampling_rate = model.sample_rate

from scipy.io.wavfile import write  # noqa: E402
write("rock.wav", sampling_rate, (results[0].numpy()).T)