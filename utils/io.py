import os
import gc
import torch


def get_output_dirs(runId=1, base_dir=None):
    if runId is None or base_dir is None:
        raise ValueError("runId and base_dir must be provided")

    google_drive = '/content/drive/MyDrive'
    base_path = os.path.join(google_drive, base_dir)
    print(f'Path to the base directory: {base_path}')

    log_dir = os.path.join(base_path, "run_" + str(runId))
    os.makedirs(log_dir, exist_ok=True)

    output_dir = os.path.join(log_dir, "latent_images")
    model_ckpnt_dir = os.path.join(log_dir, "checkpoint")
    tb_log_dir = os.path.join(log_dir, "tb_logs")
    predictions_dir = os.path.join(log_dir, "predictions")
    debug_log_dir = os.path.join(log_dir, "debug_logs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_ckpnt_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(debug_log_dir, exist_ok=True)

    return output_dir, model_ckpnt_dir, tb_log_dir, predictions_dir, debug_log_dir


def free_memory():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
