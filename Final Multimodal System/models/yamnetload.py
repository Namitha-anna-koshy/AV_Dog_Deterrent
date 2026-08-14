'''import os
import tensorflow_hub as hub

os.environ["TFHUB_CACHE_DIR"] = "models/tfhub_cache"

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
'''
import sounddevice as sd
print(sd.query_devices(9))
