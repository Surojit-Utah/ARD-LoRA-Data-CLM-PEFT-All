"""
ARD-LoRA Classification Checkpoint Evaluation
==============================================

This script evaluates a trained ARD-LoRA classification model checkpoint by:
1. Loading the checkpoint and validation data
2. Evaluating uncertainty metrics (ACC, ECE, NLL)
3. Plotting estimated variances for ARD analysis

Usage:
    python evaluate/evaluate_classification_checkpoint.py \
        --checkpoint_path path/to/checkpoint \
        --config_path config/run_training_params.yaml \
        --output_dir results/evaluation
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CONFIG
from model.model_llama import ProbLoRALayer, inject_problora_llama
from dataset.S2ClassDataset import S2ClassDataset
from evaluate.uncertainty_metrics import UncertaintyEvaluator
from utils.plot import plot_mean_encodings
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ARD-LoRA classification checkpoint")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to run evaluation on (e.g., 'cuda' or 'cpu')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--plot_batch_size",
        type=int,
        required=True,
        help="Batch size for plotting (uses more samples)"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        required=True,
        help="Number of bins for ECE calculation"
    )
    return parser.parse_args()


def load_config():
    """Load configuration from YAML file."""
    cfg = CONFIG or {}
    merged = {}
    
    # Apply top-level defaults
    if cfg.get("defaults"):
        merged.update(cfg.get("defaults"))
    
    # Apply model-specific defaults
    model_name = merged.get("model_name")
    if model_name and "models" in cfg and model_name in cfg["models"]:
        model_cfg = cfg["models"][model_name]
        if model_cfg.get("defaults"):
            merged.update(model_cfg.get("defaults"))
    
    # Apply dataset-specific config
    dataset_name = merged.get("dataset_name")
    if dataset_name and "datasets" in cfg and dataset_name in cfg["datasets"]:
        dataset_cfg = cfg["datasets"][dataset_name]
        if dataset_cfg:
            merged.update(dataset_cfg)
    
    return merged


def load_checkpoint(checkpoint_path, config, device):
    """
    Load model checkpoint with ProbLoRA layers.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        config: Configuration dictionary
        device: Device to load model on
    
    Returns:
        model: Loaded model with ProbLoRA layers
        tokenizer: Associated tokenizer
    """
    print(f"\n[LOAD] Loading checkpoint from: {checkpoint_path}")
    
    # Load tokenizer
    model_name_or_path = config["model_name_or_path"]
    tokenizer_name = config.get("tokenizer_name") or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Configure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[TOKENIZER] Set pad_token to eos_token: {repr(tokenizer.pad_token)}")
    
    # Load base model
    model_kwargs = {}
    if config.get("bf16"):
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif config.get("fp16"):
        model_kwargs["torch_dtype"] = torch.float16
    
    print(f"[LOAD] Loading base model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    
    # Inject ProbLoRA layers (must match training configuration)
    print(f"[LOAD] Injecting ProbLoRA layers with rank={config['rank']}")
    model = inject_problora_llama(
        model,
        rank=config["rank"],
        num_tokens=config["max_len"],
        ard_prior_samples=config["ard_prior_samples"],
        logvar_clamp_min=config.get("logvar_clamp_min", -5.0),
        logvar_clamp_max=config.get("logvar_clamp_max", 5.0),
        beta_logvar_clamp_min=config.get("beta_logvar_clamp_min", -3.0),
        beta_logvar_clamp_max=config.get("beta_logvar_clamp_max", 3.0),
        sample_clamp_min=config.get("sample_clamp_min", -3.0),
        sample_clamp_max=config.get("sample_clamp_max", 3.0),
        attn_implementation=config.get("attn_implementation"),
        target_attention_layers=config.get("target_attention_layers"),
        deterministic=config.get("deterministic_lora", False),
        enable_clamps=config.get("enable_clamps", True),
        lora_alpha=config.get("lora_alpha"),
        scaling=config.get("scaling"),
    )
    
    # Load checkpoint weights
    checkpoint_file = Path(checkpoint_path) / "pytorch_model.bin"
    if not checkpoint_file.exists():
        # Try alternative checkpoint formats
        checkpoint_file = Path(checkpoint_path) / "model.safetensors"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    print(f"[LOAD] Loading checkpoint weights from: {checkpoint_file}")
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # Move to device and set eval mode
    model = model.to(device)
    model.eval()
    
    print(f"[LOAD] Model loaded successfully")
    print(f"[LOAD]   Device: {device}")
    print(f"[LOAD]   Dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer


def load_validation_data(config, tokenizer):
    """
    Load validation dataset.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer instance
    
    Returns:
        val_dataset: Validation dataset
        target_ids: Tensor of answer token IDs
    """
    print(f"\n[DATA] Loading validation dataset...")
    dataset_name = config["dataset_name_specific"]
    
    # Create dataset args
    class DatasetArgs:
        def __init__(self, config):
            dataset_name = config["dataset_name_specific"]
            if dataset_name.lower() == "arc_easy" or dataset_name.lower() == "arc-easy":
                self.dataset = "ARC-Easy"
            else:
                self.dataset = dataset_name
            
            self.model = config["model_name_or_path"]
            self.max_seq_len = config.get("max_len", 512)
            self.pad_to_max_length = False
            self.use_slow_tokenizer = False
            self.batch_size = config["batch_size"]
            self.num_workers = 0
            self.testing_set = "val"
            self.seed = config.get("seed", 42)
    
    args = DatasetArgs(config)
    print(f"[DATA] Dataset: {args.dataset}")
    
    # Add to task_info if needed
    if args.dataset not in S2ClassDataset.task_info:
        S2ClassDataset.task_info[args.dataset] = {
            "num_labels": 4,
            "tokenize_keys": ("question", "choices")
        }
    
    # Create dataset
    dataset = S2ClassDataset(accelerator=None, args=args)
    dataset.get_loaders()
    
    val_dataset = dataset.test_dataloader.dataset
    
    # Get target token IDs for answer choices
    def last_token_id(tok, s: str) -> int:
        ids = tok.encode(s, add_special_tokens=False)
        return ids[-1]
    
    target_ids = torch.tensor([
        last_token_id(tokenizer, " A"),
        last_token_id(tokenizer, " B"),
        last_token_id(tokenizer, " C"),
        last_token_id(tokenizer, " D"),
    ])
    
    print(f"[DATA] Validation samples: {len(val_dataset)}")
    print(f"[DATA] Target token IDs: {target_ids.tolist()}")
    
    return val_dataset, target_ids


def evaluate_uncertainty(model, val_dataset, target_ids, device, batch_size, n_bins):
    """
    Evaluate uncertainty metrics on validation set.
    
    Args:
        model: Trained model
        val_dataset: Validation dataset
        target_ids: Tensor of answer token IDs
        device: Device to run on
        batch_size: Batch size for evaluation
        n_bins: Number of bins for ECE
    
    Returns:
        metrics: Dictionary with uncertainty metrics
    """
    print(f"\n[UNCERTAINTY] Evaluating uncertainty metrics...")
    print(f"[UNCERTAINTY]   Dataset size: {len(val_dataset)}")
    print(f"[UNCERTAINTY]   Batch size: {batch_size}")
    print(f"[UNCERTAINTY]   ECE bins: {n_bins}")
    
    # Create dataloader
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=False
    )
    
    # Initialize evaluator
    evaluator = UncertaintyEvaluator(n_bins=n_bins)
    
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Extract classes
            if "classes" in inputs:
                classes = inputs.pop("classes")
            elif "labels" in inputs:
                classes = inputs.pop("labels")
            else:
                continue
            
            # Forward pass
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
            logits = outputs.logits
            
            # Extract last token logits
            attn = inputs["attention_mask"]
            last_idx = attn.long().sum(dim=1) - 1
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            last_token_logits = logits[batch_indices, last_idx, :]
            
            # Filter to answer tokens
            target_ids_device = target_ids.to(last_token_logits.device)
            filtered_logits = last_token_logits[:, target_ids_device.squeeze()]
            
            # Convert to probabilities
            probs = torch.softmax(filtered_logits, dim=-1)
            
            # Collect predictions
            all_labels.extend(classes.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"[UNCERTAINTY]   Processed {batch_idx + 1} batches, {len(all_labels)} samples")
    
    print(f"[UNCERTAINTY] Evaluation completed on {len(all_labels)} samples")
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = evaluator.evaluate_predictions(y_true, y_probs)
    
    print(f"\n[UNCERTAINTY] Results:")
    print(f"   Accuracy (ACC): {metrics['accuracy']:.4f}")
    print(f"   Expected Calibration Error (ECE): {metrics['ece']:.4f}")
    print(f"   Negative Log-Likelihood (NLL): {metrics['nll']:.4f}")
    
    return metrics


def plot_variances(model, val_dataset, device, output_dir, plot_batch_size):
    """
    Plot estimated variances for ARD analysis.
    
    Args:
        model: Trained model
        val_dataset: Validation dataset
        device: Device to run on
        output_dir: Directory to save plots
        plot_batch_size: Batch size for plotting
    """
    print(f"\n[PLOTTING] Generating variance plots...")
    print(f"[PLOTTING]   Plot batch size: {plot_batch_size}")
    
    # Create plot directory
    plot_dir = Path(output_dir) / "variance_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloader for plotting
    plot_dataloader = DataLoader(
        val_dataset,
        batch_size=plot_batch_size,
        collate_fn=default_data_collator,
        shuffle=False
    )
    
    # Generate plots
    plot_mean_encodings(model, plot_dataloader, device, str(plot_dir), epoch=0)
    
    print(f"[PLOTTING] Plots saved to: {plot_dir}")
    
    # List generated plots
    plot_files = sorted(plot_dir.glob("*.jpg"))
    print(f"[PLOTTING] Generated {len(plot_files)} plot files:")
    for plot_file in plot_files[:10]:  # Show first 10
        print(f"   - {plot_file.name}")
    if len(plot_files) > 10:
        print(f"   ... and {len(plot_files) - 10} more")


def save_results(metrics, output_dir, checkpoint_path):
    """
    Save evaluation results to JSON file.
    
    Args:
        metrics: Uncertainty metrics dictionary
        output_dir: Directory to save results
        checkpoint_path: Path to checkpoint (for metadata)
    """
    results_file = Path(output_dir) / "evaluation_results.json"
    
    results = {
        "checkpoint_path": str(checkpoint_path),
        "uncertainty_metrics": metrics,
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVE] Results saved to: {results_file}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 80)
    print("ARD-LoRA Classification Checkpoint Evaluation")
    print("=" * 80)
    
    # Determine output directory based on checkpoint path
    checkpoint_path = Path(args.checkpoint_path)
    
    # Navigate up from checkpoint to run directory (e.g., checkpoint-1000 -> run_2)
    # Checkpoint structure: .../run_X/model_ckpt/checkpoint-XXXX
    if checkpoint_path.parent.name == "model_ckpt":
        run_dir = checkpoint_path.parent.parent
        output_dir = run_dir / "eval_results"
    else:
        # Fallback to user-specified output_dir if structure doesn't match
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] Saving results to: {output_dir}")
    
    # Load configuration
    print(f"\n[CONFIG] Loading configuration from: {args.config_path}")
    config = load_config()
    
    # Override batch size if specified
    if args.batch_size:
        config["batch_size"] = args.batch_size
    
    print(f"[CONFIG] Model: {config.get('model_name')}")
    print(f"[CONFIG] Dataset: {config.get('dataset_name_specific')}")
    print(f"[CONFIG] Device: {args.device}")
    
    # Load checkpoint
    model, tokenizer = load_checkpoint(args.checkpoint_path, config, args.device)
    
    # Load validation data
    val_dataset, target_ids = load_validation_data(config, tokenizer)
    
    # Evaluate uncertainty
    metrics = evaluate_uncertainty(
        model=model,
        val_dataset=val_dataset,
        target_ids=target_ids,
        device=args.device,
        batch_size=args.batch_size,
        n_bins=args.n_bins
    )
    
    # Plot variances
    plot_variances(
        model=model,
        val_dataset=val_dataset,
        device=args.device,
        output_dir=str(output_dir),
        plot_batch_size=args.plot_batch_size
    )
    
    # Save results
    save_results(metrics, str(output_dir), args.checkpoint_path)
    
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()