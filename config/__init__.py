import os
import yaml

CONFIG = None
cfg_path = os.path.join(os.path.dirname(__file__), "run_training_params.yaml")
if os.path.exists(cfg_path):
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            CONFIG = yaml.safe_load(f)
    except Exception:
        CONFIG = {}
else:
    CONFIG = {}
