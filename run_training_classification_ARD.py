"""
ARD-LoRA Classification Training on ARC-Easy
============================================

This script trains ARD-LoRA on classification tasks using the ARC-Easy dataset.
Follows the structure of run_training_cached.py but adapted for classification.

Key Features:
1. Last-token prediction with K-class CE loss (not full-vocab)
2. Uses ARCEasyDataset for prompt formatting and answer token filtering
3. ARDClassificationTrainer for classification-specific loss computation
4. Evaluates after every epoch with validation split
"""

import os
from pathlib import Path
import torch
from config import CONFIG
from model.model_llama import ProbLoRALayer, inject_problora_llama
from trainer.trainer_classification_ARD import ARDClassificationTrainer, build_classification_trainer
from dataset.S2ClassDataset import S2ClassDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils.io import get_output_dirs, free_memory


def _merge_config(defaults: dict):
    """Merge configuration with hierarchy: defaults -> top-level -> model -> dataset"""
    cfg = CONFIG or {}
    merged = dict(defaults)

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
    
    # Validate and fix data types for critical parameters
    _validate_config_types(merged)
    
    return merged


def _validate_config_types(config):
    """Validate and fix data types for critical configuration parameters"""
    # Ensure numeric parameters are correct types
    float_params = ["learning_rate", "kl_loss_beta", "warmup_ratio", "weight_decay", "scaling"]
    int_params = ["rank", "batch_size", "train_epochs", "max_len", "ard_prior_samples", 
                  "uncertainty_eval_samples", "uncertainty_n_bins", "gradient_accumulation_steps",
                  "runId", "num_classes", "plot_start_epoch", "plot_interval"]
    bool_params = ["fp16", "bf16", "load_in_4bit", "use_cache", "gradient_checkpointing", 
                   "enable_callbacks", "enable_plotting", "enable_resampling"]
    
    # Convert string numbers to floats
    for param in float_params:
        if param in config and isinstance(config[param], str):
            try:
                config[param] = float(config[param])
                print(f"[CONFIG] Converted {param} from string to float: {config[param]}")
            except ValueError:
                print(f"[WARNING] Could not convert {param} to float: {config[param]}")
    
    # Convert string numbers to ints
    for param in int_params:
        if param in config and isinstance(config[param], str):
            try:
                config[param] = int(config[param])
                print(f"[CONFIG] Converted {param} from string to int: {config[param]}")
            except ValueError:
                print(f"[WARNING] Could not convert {param} to int: {config[param]}")
    
    # Validate classification-specific settings
    if "task_mode" in config and config["task_mode"] != "classification":
        print(f"[WARNING] Classification training requires task_mode='classification', but got '{config['task_mode']}'")
        print(f"[WARNING] Please update your config file")
    
    if "num_classes" not in config or config["num_classes"] <= 0:
        print(f"[WARNING] Classification training requires num_classes > 0, got {config.get('num_classes')}")
        print(f"[WARNING] Setting default num_classes=4 for ARC-Easy (A, B, C, D)")
        config["num_classes"] = 4
    
    # Validate memory optimization settings for A100
    if not config["use_cache"]:
        print(f"[CONFIG] KV caching disabled for memory optimization")
    if config["gradient_checkpointing"]:
        print(f"[CONFIG] Gradient checkpointing enabled for memory optimization")
    if config["bf16"]:
        print(f"[CONFIG] BF16 precision enabled for A100 GPU optimization")


def _validate_tokenizer_alignment(tokenizer):
    """Validate tokenizer configuration for classification training"""
    print(f"[TOKENIZER] Validation for classification training:")
    print(f"[TOKENIZER]   Model: {tokenizer.name_or_path}")
    print(f"[TOKENIZER]   Vocab size: {tokenizer.vocab_size}")
    print(f"[TOKENIZER]   BOS token: {repr(tokenizer.bos_token)} (id: {tokenizer.bos_token_id})")
    print(f"[TOKENIZER]   EOS token: {repr(tokenizer.eos_token)} (id: {tokenizer.eos_token_id})")
    print(f"[TOKENIZER]   PAD token: {repr(tokenizer.pad_token)} (id: {tokenizer.pad_token_id})")
    print(f"[TOKENIZER]   UNK token: {repr(tokenizer.unk_token)} (id: {tokenizer.unk_token_id})")
    
    # Validate LLaMA-2 specific expectations
    if "llama" in tokenizer.name_or_path.lower():
        # Check PAD token alignment
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            print(f"[TOKENIZER] PAD token aligned with EOS: {tokenizer.pad_token_id}")
        else:
            print(f"[TOKENIZER] WARNING: PAD token not aligned with EOS: pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}")
    
    # Check for potential issues
    if tokenizer.pad_token_id is None:
        print(f"[TOKENIZER] ERROR: pad_token_id is None - this will cause training issues!")
        raise ValueError("pad_token_id cannot be None for training")
    
    print(f"[TOKENIZER] Tokenizer validation complete")


def load_model_with_problora(config, verbose=False):
    """Load LLaMA2 model and inject ProbLoRA layers"""
    model_name_or_path = config["model_name_or_path"]
    tokenizer_name = config.get("tokenizer_name") or model_name_or_path
    
    print(f"[INFO] Loading model: {model_name_or_path}")
    
    # Model loading arguments
    model_kwargs = {}
    if config["load_in_4bit"]:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    
    # Set torch dtype based on precision preference (bf16 preferred on A100)
    import torch
    if config["bf16"]:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif config["fp16"]:
        model_kwargs["torch_dtype"] = torch.float16
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Disable dropout for deterministic training
    print(f"[DROPOUT] Checking and disabling dropout in model configuration...")
    dropout_params = ['attention_dropout', 'hidden_dropout', 'dropout']
    dropout_disabled = []
    for param in dropout_params:
        if hasattr(model.config, param):
            original_value = getattr(model.config, param)
            if original_value > 0:
                setattr(model.config, param, 0.0)
                print(f"[DROPOUT] Disabled {param}: {original_value} -> 0.0")
                dropout_disabled.append(f"{param}={original_value}")
            else:
                print(f"[DROPOUT] {param} already disabled: {original_value}")
        else:
            print(f"[DROPOUT] {param} not found in model config (not applicable)")
    
    if dropout_disabled:
        print(f"[DROPOUT] Summary: Disabled dropout parameters: {', '.join(dropout_disabled)}")
    else:
        print(f"[DROPOUT] Summary: All dropout parameters already disabled or not present")
    
    # Configure pad token for CLM training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[TOKENIZER] Set pad_token to eos_token: {repr(tokenizer.pad_token)}")
    
    # Validate tokenizer configuration for CLM
    _validate_tokenizer_alignment(tokenizer)
    
    # Configure use_cache for memory optimization
    original_use_cache = getattr(model.config, 'use_cache', None)
    print(f"[MEMORY] Original model.config.use_cache: {original_use_cache}")
    
    # Set use_cache based on configuration (disable for training to save memory)
    use_cache_setting = config["use_cache"]
    model.config.use_cache = use_cache_setting
    print(f"[MEMORY] Set model.config.use_cache to: {model.config.use_cache}")
    if not use_cache_setting:
        print(f"[MEMORY] KV caching disabled during training to reduce memory usage on 40GB A100")
    
    # Inject ProbLoRA with numerical stability parameters
    model = inject_problora_llama(
        model,
        rank=config["rank"],
        num_tokens=config["max_len"],
        ard_prior_samples=config["ard_prior_samples"],
        logvar_clamp_min=config["logvar_clamp_min"],
        logvar_clamp_max=config["logvar_clamp_max"],
        beta_logvar_clamp_min=config["beta_logvar_clamp_min"],
        beta_logvar_clamp_max=config["beta_logvar_clamp_max"],
        sample_clamp_min=config["sample_clamp_min"],
        sample_clamp_max=config["sample_clamp_max"],
        attn_implementation=config["attn_implementation"],
        target_attention_layers=config["target_attention_layers"],  # Read from YAML config
        deterministic=config.get("deterministic_lora"),  # Enable deterministic mode if configured
        enable_clamps=config.get("enable_clamps"),  # Enable/disable numerical stability clamps
        lora_alpha=config.get("lora_alpha"),  # Standard LoRA alpha parameter
        scaling=config.get("scaling"),
    )

    if verbose:
        print("[DEBUG] Using ProbLoRALayer type-based parameter detection...")

    # Freeze base parameters and unfreeze LoRA parameters
    trainable_count = 0
    all_param_names = []
    quantized_params_skipped = 0
    
    # ProbLoRALayer detection
    for mod_name, mod in model.named_modules():
        if isinstance(mod, ProbLoRALayer):
            if verbose:
                print(f"[DEBUG] Found ProbLoRALayer module: {mod_name}")
            
            # Only parameters directly on this module (no recursion)
            for p_name, p in mod.named_parameters(recurse=False):
                full_param_name = f"{mod_name}.{p_name}" if mod_name else p_name
                all_param_names.append(full_param_name)
                
                # CRITICAL: Debug parameter details before setting gradients
                if verbose:
                    print(f"[DEBUG] Found ProbLoRA parameter: {full_param_name}")
                    print(f"[DEBUG]   Local name: {p_name}")
                    print(f"[DEBUG]   Shape: {p.shape}")
                    print(f"[DEBUG]   Dtype: {p.dtype}")
                    print(f"[DEBUG]   Is floating point: {p.is_floating_point()}")
                
                # CRITICAL: Only set gradients on floating-point parameters
                if p.is_floating_point():
                    p.requires_grad_(True)
                    trainable_count += 1
                    if verbose:
                        print(f"[DEBUG] Trainable ProbLoRA param: {full_param_name} (shape: {p.shape}, dtype: {p.dtype})")
                else:
                    quantized_params_skipped += 1
                    print(f"[WARNING] Skipping quantized ProbLoRA param: {full_param_name} (dtype: {p.dtype})")
                    print(f"[WARNING] This ProbLoRA parameter is quantized and cannot have gradients!")
    
    # Freeze all non-ProbLoRA parameters
    for mod_name, mod in model.named_modules():
        if not isinstance(mod, ProbLoRALayer):
            for p_name, p in mod.named_parameters(recurse=False):
                if p.is_floating_point():
                    p.requires_grad_(False)

    print("\n[CHECK] ProbLoRA params and requires_grad:")
    for mod_name, mod in model.named_modules():
        if isinstance(mod, ProbLoRALayer):
            for p_name, p in mod.named_parameters(recurse=False):
                if "mu_A" in p_name or "A" in p_name or "B" in p_name:
                    full_param_name = f"{mod_name}.{p_name}" if mod_name else p_name
                    print(f"  {full_param_name:80s} requires_grad={p.requires_grad}")

    # LEGACY APPROACH (commented out - kept for reference)
    # lora_patterns = ['lora_a', 'lora_b', '.a.', '.b.', 'lora', 'adapter']
    # 
    # for name, param in model.named_parameters():
    #     all_param_names.append(name)
    #     # More comprehensive LoRA parameter detection
    #     is_lora = any(pattern in name.lower() for pattern in lora_patterns)
    #     
    #     if is_lora:
    #         # CRITICAL: Debug parameter details before setting gradients
    #         if verbose:
    #             print(f"[DEBUG] Found LoRA parameter: {name}")
    #             print(f"[DEBUG]   Shape: {param.shape}")
    #             print(f"[DEBUG]   Dtype: {param.dtype}")
    #             print(f"[DEBUG]   Is floating point: {param.dtype.is_floating_point}")
    #         
    #         # CRITICAL: Only set gradients on floating-point parameters
    #         if param.dtype.is_floating_point:
    #             param.requires_grad = True
    #             trainable_count += 1
    #             if verbose:
    #                 print(f"[DEBUG] ✅ Trainable LoRA param: {name} (shape: {param.shape}, dtype: {param.dtype})")
    #         else:
    #             quantized_params_skipped += 1
    #             print(f"[WARNING] Skipping quantized LoRA param: {name} (dtype: {param.dtype})")
    #             print(f"[WARNING] This LoRA parameter is quantized and cannot have gradients!")
    #     else:
    #         # Never touch quantized base weights - only set requires_grad on floating point
    #         if param.dtype.is_floating_point:
    #             param.requires_grad = False
    
    # If no ProbLoRA parameters found, provide detailed debugging
    if trainable_count == 0:
        print("[ERROR] No ProbLoRA parameters found! Debugging information:")
        
        # Count total ProbLoRALayer modules
        problora_modules = []
        for mod_name, mod in model.named_modules():
            if isinstance(mod, ProbLoRALayer):
                problora_modules.append(mod_name)
        
        print(f"[DEBUG] ProbLoRALayer modules found: {len(problora_modules)}")
        if problora_modules and verbose:
            for i, mod_name in enumerate(problora_modules[:10]):  # Show first 10
                print(f"[DEBUG]   Module {i+1}: {mod_name}")
            if len(problora_modules) > 10:
                print(f"[DEBUG]   ... and {len(problora_modules) - 10} more modules")
        
        if len(problora_modules) == 0:
            print("[ERROR] No ProbLoRALayer modules found in the model!")
            print("[SOLUTION] Check that inject_problora_llama() was called successfully")
        else:
            print(f"[ERROR] Found {len(problora_modules)} ProbLoRALayer modules but no trainable parameters!")
            print("[SOLUTION] Check for quantization issues or parameter dtype problems")
        
        if verbose and all_param_names:
            print(f"\n[DEBUG] Sample of all model parameters (first 20):")
            for i, name in enumerate(all_param_names[:20]):
                print(f"[DEBUG]   Parameter {i+1}: {name}")
            if len(all_param_names) > 20:
                print(f"[DEBUG]   ... and {len(all_param_names) - 20} more parameters")
        
        print(f"[INFO] Total parameters analyzed: {len(all_param_names)}")
        print(f"[INFO] Quantized parameters skipped: {quantized_params_skipped}")
        
        if quantized_params_skipped > 0:
            print(f"[DIAGNOSIS] All ProbLoRA parameters appear to be quantized.")
            print(f"[SOLUTION] Consider one of the following:")
            print(f"           1. Set load_in_4bit: false in config to disable quantization")
            print(f"           2. Use different LoRA injection that preserves floating-point parameters")
            print(f"           3. Check if ProbLoRA injection is compatible with quantization")
            print(f"[CONFIG] Current quantization setting: load_in_4bit = {config['load_in_4bit']}")
        
        raise RuntimeError("No trainable ProbLoRA parameters found! Check ProbLoRA injection and parameter types.")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Final parameter analysis and safety check
    print(f"\n[PARAMETER ANALYSIS] Final Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable parameter groups: {trainable_count}")
    print(f"  Quantized parameters skipped: {quantized_params_skipped}")
    print(f"  Trainable percentage: {100*trainable_params/total_params:.1f}%")
    
    print(f"[INFO] Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"[INFO] Found {trainable_count} trainable LoRA parameter groups")
    if quantized_params_skipped > 0:
        print(f"[INFO] Skipped {quantized_params_skipped} quantized parameters (cannot require gradients)")
    
    # Ensure model is in training mode
    model.train()
    print(f"[INFO] Model set to training mode: {model.training}")
    
    return model, tokenizer


def create_trainer(model, tokenizer, train_ds, val_ds, config, output_dir, target_ids, tb_log_dir=None, predictions_dir=None, latent_plot_dir=None, debug_log_dir=None):
    """Create enhanced ARD-LoRA trainer with uncertainty evaluation and callbacks"""
    
    # Get ARD prior samples directly from config
    ard_prior_samples = config["ard_prior_samples"]
    print(f"[INFO] ARD Prior: Using {ard_prior_samples} samples for ARD prior estimation")
    
    # Ensure tokenizer consistency validation
    print(f"[TOKENIZER] Trainer Creation - Tokenizer Consistency Check:")
    print(f"[TOKENIZER]   Tokenizer name: {tokenizer.name_or_path}")
    print(f"[TOKENIZER]   PAD token ID: {tokenizer.pad_token_id}")
    print(f"[TOKENIZER]   EOS token ID: {tokenizer.eos_token_id}")
    print(f"[TOKENIZER]   PAD = EOS alignment: {tokenizer.pad_token_id == tokenizer.eos_token_id}")
    
    # Create classification trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config.get("max_grad_norm", 1.0),
        bf16=config["bf16"],
        fp16=config.get("fp16", False),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=config.get("logging_steps", 10),
        save_total_limit=3,
        load_best_model_at_end=False,  # Disabled: eval doesn't return loss yet
        logging_dir=tb_log_dir,
        report_to=config.get("report_to", ["tensorboard"]),
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    
    # Use target_ids passed from main function
    print(f"[TRAINER] Target token IDs: {target_ids.tolist()}")
    
    # Check if callbacks are enabled in config
    enable_uncertainty = config.get("enable_callbacks")
    enable_pred_tracker = config.get("enable_prediction_tracking")
    
    # Prepare prediction tracker parameters
    pred_tracker_params = None
    if enable_pred_tracker and predictions_dir:
        pred_tracker_params = {
            'predictions_dir': predictions_dir,
            'prediction_n_examples': config.get('prediction_n_examples'),
            'dataset_name': config.get('dataset_name')
        }
    
    # Prepare plotting parameters from config
    enable_plotting = config.get('enable_plotting')
    plot_params = {
        'start_epoch': config.get('plot_start_epoch'),
        'interval': config.get('plot_interval'),
        'plot_batch_size': config.get('plot_batch_size'),
        'latent_plot_dir': latent_plot_dir  # Pass the latent plot directory
    } if enable_plotting else None
    
    # Use build_classification_trainer for proper callback integration
    trainer = build_classification_trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=None,  # Will use default
        tokenizer=tokenizer,
        config=config,
        target_ids=target_ids,
        num_classes=config.get('num_classes'),
        enable_uncertainty_eval=enable_uncertainty,
        enable_prediction_tracker=enable_pred_tracker,
        prediction_tracker_params=pred_tracker_params,
        enable_plotting=enable_plotting,
        plot_params=plot_params,
        debug_log_dir=debug_log_dir,
    )
    
    # Post-creation validation - ensure trainer uses the same tokenizer
    if hasattr(trainer, 'tokenizer') and trainer.tokenizer is not tokenizer:
        print(f"[TOKENIZER] WARNING: Trainer tokenizer differs from input tokenizer!")
        print(f"[TOKENIZER]   Input tokenizer PAD ID: {tokenizer.pad_token_id}")
        print(f"[TOKENIZER]   Trainer tokenizer PAD ID: {trainer.tokenizer.pad_token_id}")
    else:
        print(f"[TOKENIZER] Trainer tokenizer consistency verified")
    
    return trainer

def main():
    """Main training function for classification on ARC-Easy"""
    print("=" * 80)
    print("ARD-LoRA Classification Training on ARC-Easy")
    print("=" * 80)
    
    free_memory()
    
    # Load configuration from YAML file
    config = _merge_config({})
    
    # Validate required configuration
    if not config:
        raise ValueError("No configuration found! Please ensure config.py imports a valid YAML configuration.")
    
    print(f"[CONFIG] Model: {config.get('model_name')}")
    print(f"[CONFIG] Dataset: {config.get('dataset_name')}")
    print(f"[CONFIG] Dataset Name: {config['dataset_name_specific']}")
    print(f"[CONFIG] Task Mode: {config.get('task_mode', 'classification')}")
    print(f"[CONFIG] Num Classes: {config.get('num_classes')}")
    print(f"[CONFIG] KL Beta: {config.get('kl_loss_beta')}")
    print(f"[CONFIG] Rank: {config.get('rank')}")
    print(f"[CONFIG] Train Epochs: {config.get('train_epochs')}")
    
    # Memory optimization settings
    print(f"[CONFIG] Memory Optimizations for 40GB A100:")
    print(f"[CONFIG]   Use Cache: {config.get('use_cache')}")
    print(f"[CONFIG]   Gradient Checkpointing: {config.get('gradient_checkpointing')}")
    print(f"[CONFIG]   BF16: {config.get('bf16')}")
    print(f"[CONFIG]   Batch Size: {config.get('batch_size')}")
    print(f"[CONFIG]   Gradient Accumulation: {config.get('gradient_accumulation_steps')}")
    
    # ARD and uncertainty settings
    print(f"[CONFIG] ARD Prior Samples: {config.get('ard_prior_samples')}")
    print(f"[CONFIG] Uncertainty Eval Samples: {config.get('uncertainty_eval_samples')}")
    print(f"[CONFIG] Uncertainty ECE Bins: {config.get('uncertainty_n_bins')}")
    print(f"[CONFIG] Enable Callbacks: {config.get('enable_callbacks')}")
    print(f"[CONFIG] Enable Plotting: {config.get('enable_plotting')}")
    
    # Load model with ProbLoRA
    print("\n[STEP 1] Loading model and injecting ProbLoRA...")
    model, tokenizer = load_model_with_problora(config, verbose=False)
    
    # Load ARC-Easy dataset using BayesianPEFT's S2ClassDataset
    print(f"\n[STEP 2] Loading ARC-Easy dataset (BayesianPEFT format)...")
    dataset_name = config["dataset_name_specific"]
    
    try:
        # Create a simple args object for S2ClassDataset
        class DatasetArgs:
            def __init__(self, config):
                # BayesianPEFT expects "ARC-Easy" not "arc_easy"
                dataset_name = config["dataset_name_specific"]
                if dataset_name.lower() == "arc_easy" or dataset_name.lower() == "arc-easy":
                    self.dataset = "ARC-Easy"
                else:
                    self.dataset = dataset_name
                
                self.model = config["model_name_or_path"]
                self.max_seq_len = config.get("max_len", 512)
                self.pad_to_max_length = False  # Use dynamic padding
                self.use_slow_tokenizer = False
                self.batch_size = config["batch_size"]
                self.num_workers = 0
                self.testing_set = "val"  # Use validation set for evaluation
                self.seed = config.get("seed", 42)
        
        args = DatasetArgs(config)
        print(f"[INFO] Dataset name for S2ClassDataset: {args.dataset}")
        
        # Monkey-patch task_info to include ARC-Easy if not present
        # S2ClassDataset.task_info doesn't have ARC datasets since they're handled specially
        # But it still tries to access task_info[args.dataset] in __init__
        if args.dataset not in S2ClassDataset.task_info:
            # Add ARC-Easy with 4 labels (A, B, C, D) - though some have 5 (E)
            # The actual labels are handled in _tokenize method
            S2ClassDataset.task_info[args.dataset] = {
                "num_labels": 4,  # Typical for ARC (can be 3-5)
                "tokenize_keys": ("question", "choices")  # Will be handled specially
            }
            print(f"[INFO] Added {args.dataset} to task_info")
        
        # Create S2ClassDataset instance (following BayesianPEFT pattern)
        dataset = S2ClassDataset(accelerator=None, args=args)
        
        # Call get_loaders to process and tokenize the dataset
        dataset.get_loaders()
        
        # Get the processed datasets from dataloaders
        train_ds = dataset.train_dataloader.dataset
        val_ds = dataset.test_dataloader.dataset  # They call it test_dataloader
        
        # Get target_ids for answer tokens (A, B, C, D)
        # For ARC-Easy: map_dict = {"A": 0, "B": 1, "C": 2, "D": 3}
        # We need the token IDs for " A", " B", " C", " D"
        # CRITICAL: Use [-1] to get LAST token (the letter), not [0] (the space)
        def last_token_id(tok, s: str) -> int:
            """Extract last token ID from a string (safe for multi-piece tokens)."""
            ids = tok.encode(s, add_special_tokens=False)
            return ids[-1]  # safe even if multi-piece
        
        target_ids = torch.tensor([
            last_token_id(tokenizer, " A"),
            last_token_id(tokenizer, " B"),
            last_token_id(tokenizer, " C"),
            last_token_id(tokenizer, " D"),
        ])
        
        # Validate tokenizer consistency after dataset loading
        print(f"[TOKENIZER] Post-dataset loading validation:")
        print(f"[TOKENIZER]   PAD token ID: {tokenizer.pad_token_id}")
        print(f"[TOKENIZER]   EOS token ID: {tokenizer.eos_token_id}")
        if tokenizer.pad_token_id != tokenizer.eos_token_id:
            print(f"[TOKENIZER] WARNING: PAD and EOS token IDs don't match after dataset loading!")
        else:
            print(f"[TOKENIZER] PAD and EOS alignment maintained: {tokenizer.pad_token_id}")
        
        print(f"[INFO] Training samples: {len(train_ds)}")
        print(f"[INFO] Validation samples: {len(val_ds)}")
        print(f"[INFO] Target IDs for answer tokens: {target_ids.tolist()}")
        
        # Sanity check: print label mappings with token representations
        print(f"[TOKENIZER] Label token mappings:")
        for label, tid in zip([" A", " B", " C", " D"], target_ids.tolist()):
            token_str = tokenizer.convert_ids_to_tokens(tid)
            print(f"[TOKENIZER]   {repr(label)} -> id={tid}, tok={repr(token_str)}")
        
        # Check validation dataset size for ARD prior estimation
        val_size = len(val_ds)
        ard_samples = config["ard_prior_samples"]
        if val_size > 0 and val_size < ard_samples:
            print(f"\n[WARNING] Validation dataset size issue:")
            print(f"          Validation samples: {val_size}")
            print(f"          ARD prior samples requested: {ard_samples}")
            print(f"          This may affect ARD prior estimation quality")
            print(f"          Consider increasing validation data or reducing ard_prior_samples in config")
            print(f"          ARD training will use all {val_size} validation samples for prior estimation")
        
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Setup training
    print(f"\n[STEP 3] Setting up ARD-LoRA training...")
    
    # Create run-specific base directory with deterministic/probabilistic mode
    model_mode = "deterministic" if config.get('deterministic_lora', True) else "probabilistic"
    model_name_clean = config.get('model_name').replace('/', '_')
    base_output_dir = f"{model_name_clean}_ARD_LoRA_Classification_{dataset_name}_{model_mode}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    print(f"[INFO] Training mode: {model_mode.upper()}")
    print(f"[INFO] Base output directory: {base_output_dir}")
    
    # Get run-specific directories
    output_dir, model_ckpt_dir, tb_log_dir, predictions_dir, debug_log_dir = get_output_dirs(
        config["runId"],
        base_output_dir
    )
    
    print(f"[INFO] Directory structure created:")
    print(f"       Base: {base_output_dir}")
    print(f"       Latent images: {output_dir}")
    print(f"       Model checkpoints: {model_ckpt_dir}")
    print(f"       TensorBoard logs: {tb_log_dir}")
    print(f"       Predictions: {predictions_dir}")
    print(f"       Debug logs: {debug_log_dir}")

    trainer = create_trainer(model, tokenizer, train_ds, val_ds, config, model_ckpt_dir, target_ids, tb_log_dir, predictions_dir, output_dir, debug_log_dir)
    # Dropout and grad norm settings of Llama2
    print("TrainingArguments weight_decay:", trainer.args.weight_decay)
    print("max_grad_norm:", trainer.args.max_grad_norm)

    # Final tokenizer consistency validation before training
    print(f"\n[TOKENIZER] Final Pre-Training Validation:")
    print(f"[TOKENIZER]   Model tokenizer available: {hasattr(model, 'config') and hasattr(model.config, 'vocab_size')}")
    print(f"[TOKENIZER]   Main tokenizer PAD ID: {tokenizer.pad_token_id}")
    print(f"[TOKENIZER]   Trainer tokenizer PAD ID: {trainer.tokenizer.pad_token_id}")
    
    # Check if all tokenizers have the same PAD token ID
    all_pad_ids = [
        tokenizer.pad_token_id,
        trainer.tokenizer.pad_token_id,
    ]
    if len(set(all_pad_ids)) == 1:
        print(f"[TOKENIZER] All components use consistent PAD token ID: {all_pad_ids[0]}")
    else:
        print(f"[TOKENIZER] ERROR: Inconsistent PAD token IDs found: {all_pad_ids}")
        raise ValueError("Tokenizer PAD token ID mismatch detected - this will cause training issues!")
    
    # Training with ARD
    print(f"\n[STEP 4] Starting training...")
    print(f"[INFO] Beta (KL strength): {config.get('kl_loss_beta')}")
    print(f"[INFO] Output directory: {model_ckpt_dir}")
    print(f"[INFO] TensorBoard logs: {tb_log_dir}")
    print(f"[INFO] TensorBoard logging enabled: {'tensorboard' in config['report_to']}")
    
    # Ensure model is in training mode before training starts
    model.train()
    print(f"[INFO] Model training mode verified: {model.training}")
    
    # CRITICAL: Verify trainable parameters one more time before training
    trainable_params_check = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[CRITICAL CHECK] Pre-training parameter verification:")
    print(f"  Trainable parameters: {trainable_params_check:,}")
    print(f"  Model in training mode: {model.training}")
    
    # Sample a few trainable parameters to verify grad status
    trainable_sample = [(n, p.requires_grad, p.dtype) for n, p in model.named_parameters() if p.requires_grad][:5]
    if trainable_sample:
        print(f"  Sample trainable parameters:")
        for name, req_grad, dtype in trainable_sample:
            print(f"    - {name}: requires_grad={req_grad}, dtype={dtype}")
    else:
        print(f"  [ERROR] NO TRAINABLE PARAMETERS FOUND!")
        raise RuntimeError("No trainable parameters - training will fail!")
    
    try:
        # Skip initial evaluation to avoid eval_dataset requirement
        print("\n[EVAL] Skipping initial evaluation - proceeding directly to training")
        
        last_checkpoint = None
        if os.path.isdir(model_ckpt_dir):
            checkpoints = [os.path.join(model_ckpt_dir, d) for d in os.listdir(model_ckpt_dir) if "checkpoint" in d]
            if checkpoints:
                last_checkpoint = max(checkpoints, key=os.path.getmtime)

        if last_checkpoint:
            print(f"Resuming from checkpoint: {last_checkpoint}")
            print("\n[TRAIN] Starting ARD-LoRA classification training from checkpoint...")
            train_results = trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            print("Starting training from scratch...")
            print("\n[TRAIN] Starting ARD-LoRA classification training...")
            train_results = trainer.train()

        print(f"\n[SUCCESS] Training completed!")
        print(f"[INFO] Final loss: {train_results.training_loss:.4f}")
        
        # Print summary of uncertainty evolution
        if hasattr(trainer, 'uncertainty_results') and trainer.uncertainty_results:
            print(f"\n[UNCERTAINTY] Evolution across {len(trainer.uncertainty_results)} epochs:")
            for i, result in enumerate(trainer.uncertainty_results):
                epoch = result.get('epoch', i+1)
                acc = result.get('accuracy', 0)
                ece = result.get('ece', 0)
                nll = result.get('nll', 0)
                print(f"   Epoch {epoch}: ACC={acc:.4f}, ECE={ece:.4f}, NLL={nll:.4f}")
        
        # Save final model
        trainer.save_model()
        print(f"[SAVE] Model saved to {model_ckpt_dir}")
        
        # Print TensorBoard information
        if 'tensorboard' in config['report_to']:
            print(f"\n[TENSORBOARD] Training metrics logged to: {tb_log_dir}")
            print(f"[INFO] To view metrics, run: tensorboard --logdir {tb_log_dir}")
            print(f"[INFO] Metrics include:")
            print(f"       - train/ce_loss: Cross-entropy loss during training")
            print(f"       - train/kl_loss: KL divergence loss during training")
            print(f"       - train/total_loss: Total loss (CE + KL) during training")
            print(f"       - eval/ce_loss: Cross-entropy loss during evaluation")
            print(f"       - eval/kl_loss: KL divergence loss during evaluation")
            print(f"       - eval/total_loss: Total loss during evaluation")
        
        print("\n" + "=" * 80)
        print("ARD-LoRA classification training completed successfully!")
        print(f"Model and logs saved to: {base_output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()