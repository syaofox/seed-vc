from seed_vc.config import *

from huggingface_hub import hf_hub_download, try_to_load_from_cache


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    cache_dir = os.environ.get("HF_HUB_CACHE", "./checkpoints/hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    repo_cache_name = repo_id.replace("/", "--")
    local_model_path = os.path.join(cache_dir, repo_cache_name, model_filename)
    local_config_path = os.path.join(cache_dir, repo_cache_name, config_filename) if config_filename else None

    if os.path.exists(local_model_path):
        print(f"Loading {model_filename} from local cache: {local_model_path}")
        if config_filename is None:
            return local_model_path
        if local_config_path and os.path.exists(local_config_path):
            return local_model_path, local_config_path

    cached_model_path = try_to_load_from_cache(repo_id=repo_id, filename=model_filename, cache_dir=cache_dir)
    if cached_model_path and os.path.exists(cached_model_path):
        print(f"Loading {model_filename} from HF cache: {cached_model_path}")
        if config_filename is None:
            return cached_model_path
        cached_config_path = try_to_load_from_cache(repo_id=repo_id, filename=config_filename, cache_dir=cache_dir)
        if cached_config_path and os.path.exists(cached_config_path):
            return cached_model_path, cached_config_path

    print(f"Downloading {model_filename} from HuggingFace Hub...")
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=cache_dir)
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=cache_dir)
    return model_path, config_path
