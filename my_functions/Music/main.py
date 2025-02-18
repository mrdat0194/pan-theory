import audiocraft
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained('facebook/musicgen-small')

model.set_generation_params(duration=10)

print("Generating music...")
results = model.generate(['classical rock'])
sampling_rate = model.sample_rate

from scipy.io.wavfile import write
write("rock.wav", sampling_rate, (results[0].numpy()).T)