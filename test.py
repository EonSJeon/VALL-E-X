from utils.prompt_making import make_prompt

### Use given transcript
make_prompt(name="sample", audio_prompt_path="sample.wav",
                transcript="It was a bright cold day in April, and the clocks were striking thirteen.")

from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
preload_models()

text_prompt = """
Hey, Traveler, Listen to this, This machine has taken my voice, and now it can talk just like me!
"""
audio_array = generate_audio(text_prompt, prompt="sample")

write_wav("sample_cloned.wav", SAMPLE_RATE, audio_array)
