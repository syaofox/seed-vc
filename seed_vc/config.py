import os


os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
