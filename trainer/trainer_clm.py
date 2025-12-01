from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.trainer_callback import TrainerCallback
import torch
from torch import nn
import numpy as np
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, Subset
import json
import os
import gc
from pathlib import Path
from evaluate.uncertainty_metrics import UncertaintyEvaluator, print_evaluation_results
from utils.plot import plot_mean_encodings, diag_axis_splom
from utils.dataset_debug import (
    debug_dataset_filtering,
    analyze_data_collator_filtering,
    get_effective_dataset_size,
    apply_dataset_filtering_fixes,
    run_complete_dataset_analysis
)
from evaluate.prediction_tracker import PredictionTracker


class ARDCLMTrainer(Trainer):
    """Enhanced ARD Trainer following DeBERTa pattern, adapted for LLaMA."""
    
    def __init__(self, *args, beta=0.01, ard_heldout_loader=None, 
                 n_bins=15, output_dir=None, ard_prior_samples=100, target_attention_layers=None, verbose=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.ard_heldout_loader = ard_heldout_loader  # Pre-configured DataLoader for ARD prior estimation
        self.ard_prior_samples = ard_prior_samples  # Store ARD prior samples count
        self.n_bins = n_bins
        self.uncertainty_evaluator = UncertaintyEvaluator(n_bins=n_bins)
        self.uncertainty_results = []  # Store results across epochs
        self.output_dir = output_dir or self.args.output_dir
        self.verbose = verbose  # Control debug message verbosity
        
        # Store layer configuration from YAML - must be provided explicitly
        if target_attention_layers is None:
            raise ValueError("target_attention_layers must be provided from YAML configuration")
        self.target_attention_layers = target_attention_layers
        
        # Set trainer reference on model for callbacks
        self.model.trainer = self
        
        # Track loss components for logging
        self.last_ce_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_total_loss = 0.0
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss following DeBERTa pattern, adapted for LLaMA architecture."""
        # Extract labels for causal LM
        labels = inputs.get("labels")
        
        # Single forward pass with hidden states for both CE loss and KL computation
        outputs = model(
            **inputs,
            output_hidden_states=True,  # Get hidden states in single forward pass
            use_cache=False
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states  # Get hidden states from first forward pass (with gradients!)
        
        # Compute CE loss (causal LM loss)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ce_loss = loss_fct(shift_logits, shift_labels)
        else:
            ce_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Compute KL divergence following DeBERTa pattern
        kl = 0.0
        kl_debug_info = {}
        total_kl_layers = 0
        
        # Debug: Check if hidden states have gradients (only once per epoch)
        if hasattr(self, '_debug_step_count'):
            self._debug_step_count += 1
        else:
            self._debug_step_count = 1
            
        # Track current epoch for debug printing - use integer epoch from trainer state
        current_epoch = int(getattr(self.state, 'epoch', 0)) if hasattr(self, 'state') else 0
        
        # Initialize or check if we need to print debug for this epoch
        if not hasattr(self, '_last_gradient_debug_epoch'):
            self._last_gradient_debug_epoch = -1
            
        # Only print once per epoch when epoch changes
        if current_epoch > self._last_gradient_debug_epoch:
            self._last_gradient_debug_epoch = current_epoch
            if hidden_states is not None:
                print(f"\n[GRADIENT DEBUG] Epoch {current_epoch} - Hidden States Analysis:")
                print(f"[GRADIENT DEBUG]   Number of hidden state layers: {len(hidden_states)}")
                print(f"[GRADIENT DEBUG]   Hidden states[0] requires_grad: {hidden_states[0].requires_grad}")
                print(f"[GRADIENT DEBUG]   Hidden states[0] has grad_fn: {hidden_states[0].grad_fn is not None}")
                print(f"[GRADIENT DEBUG]   ✅ Hidden states obtained from SINGLE forward pass with gradients!")
            else:
                print(f"\n[GRADIENT DEBUG] Epoch {current_epoch} - ❌ WARNING: No hidden states found!")
        
        if hidden_states is not None and hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer_idx, layer in enumerate(model.model.layers):
                # Get the input to this layer (previous layer's output)
                layer_input = hidden_states[layer_idx] if layer_idx < len(hidden_states) else None
                
                if layer_input is not None:
                    # Check attention projections (configurable from YAML)
                    if hasattr(layer, 'self_attn') and self.target_attention_layers:
                        attn = layer.self_attn
                        layer_kl_total = 0.0
                        layer_proj_count = 0
                        
                        # Process attention projections based on YAML configuration
                        for proj_name in self.target_attention_layers:
                            if hasattr(attn, proj_name):
                                proj = getattr(attn, proj_name)
                                # Check if this is a ProbLoRA layer
                                if hasattr(proj, 'kl_divergence_latent'):
                                    try:
                                        proj_kl = proj.kl_divergence_latent(layer_input)
                                        kl += proj_kl
                                        layer_kl_total += proj_kl.item() if torch.is_tensor(proj_kl) else float(proj_kl)
                                        layer_proj_count += 1
                                    except Exception:
                                        continue
                        
                        # Store debug info for this layer if it contributed to KL
                        if layer_proj_count > 0:
                            kl_debug_info[f"layer_{layer_idx}"] = {
                                "projections_processed": layer_proj_count,
                                "target_projections": list(self.target_attention_layers),
                                "layer_kl_total": layer_kl_total
                            }
                            total_kl_layers += 1
        
        # If no KL components found, create zero tensor with gradient connection
        if not torch.is_tensor(kl) or kl == 0.0:
            kl = torch.tensor(0.0, device=ce_loss.device, requires_grad=True)
        
        # Debug: Log KL computation details once per epoch
        if not hasattr(self, '_last_kl_debug_epoch'):
            self._last_kl_debug_epoch = -1
            
        # Only print once per epoch when epoch changes (use same integer epoch)
        if current_epoch > self._last_kl_debug_epoch:
            self._last_kl_debug_epoch = current_epoch
            if self.verbose:
                print(f"\n[KL DEBUG] Epoch {current_epoch} - KL Computation Details:")
                print(f"[KL DEBUG]   Target attention layers: {self.target_attention_layers}")
                print(f"[KL DEBUG]   Total layers with KL contribution: {total_kl_layers}")
                print(f"[KL DEBUG]   Total KL value: {kl.item() if torch.is_tensor(kl) else float(kl):.6f}")
                print(f"[KL DEBUG]   KL requires_grad: {kl.requires_grad if torch.is_tensor(kl) else 'N/A'}")
                print(f"[KL DEBUG]   KL grad_fn: {kl.grad_fn is not None if torch.is_tensor(kl) else 'N/A'}")
            
                if kl_debug_info:
                    print(f"[KL DEBUG]   Layer-wise breakdown:")
                    for layer_name, info in kl_debug_info.items():
                        print(f"[KL DEBUG]     {layer_name}: {info['projections_processed']}/{len(info['target_projections'])} projections, "
                              f"KL={info['layer_kl_total']:.6f}")
                else:
                    print(f"[KL DEBUG]   ⚠️ WARNING: No KL contributions found from any layer!")
                    print(f"[KL DEBUG]   This may indicate ProbLoRA layers are not properly injected.")
        
        # Total loss with KL regularization (following DeBERTa pattern)
        loss = ce_loss + self.beta * kl
        
        # Store loss components for logging
        self.last_ce_loss = ce_loss.item() if torch.is_tensor(ce_loss) else float(ce_loss)
        self.last_kl_loss = kl.item() if torch.is_tensor(kl) else float(kl)
        self.last_total_loss = loss.item() if torch.is_tensor(loss) else float(loss)
        
        return (loss, outputs) if return_outputs else loss
        
    def validate_model_gradients(self, model):
        """Validate that model parameters requiring gradients are properly set up."""
        trainable_params = 0
        total_params = 0
        
        problematic_params = []
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                # Check if parameter has gradient function but data is detached
                if param.grad_fn is None and param.is_leaf and param.requires_grad:
                    # This is normal for leaf parameters
                    pass
                elif not param.is_leaf and param.grad_fn is None:
                    problematic_params.append(f"{name}: non-leaf parameter without grad_fn")
            
        print(f"[GRADIENT VALIDATION] Model parameters:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params:.4f}")
        
        if problematic_params:
            print(f"[GRADIENT WARNING] Found {len(problematic_params)} potentially problematic parameters:")
            for param_info in problematic_params[:5]:  # Show first 5
                print(f"    {param_info}")
        else:
            print(f"[GRADIENT VALIDATION] ✅ All parameters appear correctly configured")
        
        return trainable_params > 0


    def evaluate_uncertainty(self) -> Optional[Dict[str, float]]:
        """Evaluate model uncertainty using ACC, ECE, and NLL metrics.
        
        Always uses the full self.eval_dataset for evaluation.
        """
        if self.uncertainty_evaluator is None:
            print("[WARNING] Uncertainty evaluator not available")
            return None
            
        if self.eval_dataset is None:
            print("[WARNING] No evaluation dataset available for uncertainty evaluation")
            return None
            
        eval_dataset = self.eval_dataset
        dataset_size = len(eval_dataset)
        
        print(f"\n🔄 Starting uncertainty evaluation on full dataset ({dataset_size} samples)...")
        
        # Memory optimization: Reduce batch size for evaluation
        original_eval_batch_size = self.args.per_device_eval_batch_size
        if torch.cuda.is_available():
            # Reduce eval batch size to save memory
            memory_optimized_batch_size = max(1, original_eval_batch_size // 2)
            self.args.per_device_eval_batch_size = memory_optimized_batch_size
            print(f"[MEMORY] Reducing eval batch size from {original_eval_batch_size} to {memory_optimized_batch_size}")
        
        # Create evaluation dataloader with smaller batch size
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        self.model.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            sample_count = 0
            
            for batch_idx, batch in enumerate(eval_dataloader):
                if sample_count >= dataset_size:
                    break
                
                # Move batch to device
                inputs = self._prepare_inputs(batch)
                
                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits
                labels = inputs.get('labels')
                
                if labels is None:
                    continue
                
                # For causal LM: shift predictions and labels
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten tokens
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                # Keep only non-masked tokens (labels != -100)
                valid_mask = shift_labels != -100
                valid_logits = shift_logits[valid_mask]
                valid_labels = shift_labels[valid_mask]
                
                if len(valid_labels) > 0:
                    # Convert logits to probabilities
                    probs = torch.softmax(valid_logits, dim=-1)
                    
                    # Sample subset if too many tokens
                    max_tokens_per_batch = min(200, dataset_size - sample_count)
                    if len(valid_labels) > max_tokens_per_batch:
                        indices = torch.randperm(len(valid_labels))[:max_tokens_per_batch]
                        valid_labels = valid_labels[indices]
                        probs = probs[indices]
                    
                    # Collect predictions
                    all_labels.extend(valid_labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                    sample_count += len(valid_labels)
                
                if batch_idx % 10 == 0:
                    print(f"   Processed {batch_idx + 1} batches, {sample_count} predictions")
        
        if len(all_labels) == 0:
            print("[WARNING] No valid predictions found for uncertainty evaluation")
            return None
            
        print(f"✅ Uncertainty evaluation completed on {len(all_labels)} predictions")
        
        # Restore original batch size
        if torch.cuda.is_available():
            self.args.per_device_eval_batch_size = original_eval_batch_size
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_probs = np.array(all_probs)
        
        # Compute uncertainty metrics
        metrics = self.uncertainty_evaluator.evaluate_predictions(y_true, y_probs)
        
        return metrics
    
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of each epoch. Uncertainty evaluation now handled by UncertaintyEvaluationCallback."""
        print(f"\n🔍 [DEBUG] ARDCLMTrainer.on_epoch_begin called for epoch {state.epoch}")
        print(f"[INFO] Uncertainty evaluation is now handled by UncertaintyEvaluationCallback")
        super().on_epoch_begin(args, state, control, model=model, **kwargs)
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch. Eval loss component logging now handled by EvalLossComponentsCallback."""
        print(f"\n🔍 [DEBUG] ARDCLMTrainer.on_epoch_end called for epoch {state.epoch}")
        print(f"[INFO] Evaluation loss component logging is now handled by EvalLossComponentsCallback")
        super().on_epoch_end(args, state, control, model=model, **kwargs)
    
    def _compute_eval_loss_components(self, model):
        """Compute CE and KL loss components on evaluation dataset.
        Always uses self.eval_dataset for evaluation.
        Matches training logic: uses output_hidden_states=True and use_cache=False for KL computation."""
        was_training = model.training
        try:
            model.eval()
            eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
            batch = next(iter(eval_dataloader))
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                # Forward pass with hidden states for KL computation
                outputs = model(
                    **batch,
                    output_hidden_states=True,
                    use_cache=False
                )
                logits = outputs.logits
                hidden_states = outputs.hidden_states
                labels = batch.get('labels')
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    ce_loss = loss_fct(shift_logits, shift_labels)
                else:
                    ce_loss = torch.tensor(0.0, device=logits.device)

                # KL computation: match training logic using kl_divergence_latent
                kl = 0.0
                if hidden_states is not None and hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    for layer_idx, layer in enumerate(model.model.layers):
                        layer_input = hidden_states[layer_idx] if layer_idx < len(hidden_states) else None
                        if layer_input is not None:
                            if hasattr(layer, 'self_attn') and self.target_attention_layers:
                                attn = layer.self_attn
                                for proj_name in self.target_attention_layers:
                                    if hasattr(attn, proj_name):
                                        proj = getattr(attn, proj_name)
                                        if hasattr(proj, 'kl_divergence_latent'):
                                            try:
                                                kl_val = proj.kl_divergence_latent(layer_input)
                                                if torch.is_tensor(kl_val):
                                                    kl = kl + kl_val
                                            except Exception:
                                                pass
                if not torch.is_tensor(kl) or kl == 0.0:
                    kl = torch.tensor(0.0, device=ce_loss.device)

                ce_loss_val = ce_loss.item() if torch.is_tensor(ce_loss) else float(ce_loss)
                kl_loss_val = kl.item() if torch.is_tensor(kl) else float(kl)
                return ce_loss_val, kl_loss_val
        except Exception as e:
            print(f"⚠️ Failed to compute eval loss components: {e}")
            return 0.0, 0.0
        finally:
            if was_training:
                model.train()
            else:
                model.eval()

    def _save_uncertainty_results(self):
        """Save uncertainty evaluation results to JSON file."""
        if not self.uncertainty_results:
            return
            
        results_path = Path(self.output_dir) / "uncertainty_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()  # Convert numpy scalar to Python scalar
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy array to Python list
            else:
                return obj
        
        # Convert the results to JSON-serializable format
        json_safe_results = convert_numpy_types(self.uncertainty_results)
        
        with open(results_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
            
        print(f"💾 Uncertainty results saved to: {results_path}")


class PriorEstimationCallback(TrainerCallback):
    """Callback to estimate ARD priors following DeBERTa pattern."""
    
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Estimate ARD priors using held-out data at the beginning of each epoch."""
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None:
            print("[PriorEstimationCallback] No trainer reference found on model")
            return
            
        # Use ard_heldout_loader for ARD prior estimation - fail if not configured
        if not hasattr(trainer, 'ard_heldout_loader'):
            raise AttributeError("[PriorEstimationCallback] trainer.ard_heldout_loader is required but not found. Check trainer configuration.")
        
        eval_data = trainer.ard_heldout_loader
        if eval_data is None:
            raise ValueError("[PriorEstimationCallback] trainer.ard_heldout_loader is None. ARD prior estimation requires a configured DataLoader.")
        
        print(f"[PriorEstimationCallback] Estimating ARD priors at epoch {int(state.epoch)}...")
        
        try:
            # Get relevance thresholds from trainer
            high_threshold = getattr(trainer, 'high_relevance_threshold', 0.1)
            medium_threshold = getattr(trainer, 'medium_relevance_threshold', 0.01)
            # Get layer configuration from trainer
            target_attention_layers = getattr(trainer, 'target_attention_layers', None)
            if target_attention_layers is None:
                raise ValueError("target_attention_layers must be configured in trainer")
            estimate_ard_priors_clm(model, eval_data, self.device, 
                                  high_relevance_threshold=high_threshold,
                                  medium_relevance_threshold=medium_threshold,
                                  target_attention_layers=target_attention_layers,
                                  verbose=False)
            print("[PriorEstimationCallback] ARD prior estimation completed")
        except Exception as e:
            print(f"[PriorEstimationCallback] ARD prior estimation failed: {e}")
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class UncertaintyEvaluationCallback(TrainerCallback):
    """Callback to run uncertainty evaluation at the beginning of each epoch."""
    
    def __init__(self):
        super().__init__()
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Run uncertainty evaluation at the beginning of each epoch."""
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None:
            print("[UncertaintyEvaluationCallback] No trainer reference found")
            return
        
        print(f"\n🔍 [DEBUG] UncertaintyEvaluationCallback.on_epoch_begin called for epoch {state.epoch}")
        print(f"🔍 [DEBUG] trainer.eval_dataset type: {type(trainer.eval_dataset)}")
        print(f"🔍 [DEBUG] trainer.eval_dataset is None: {trainer.eval_dataset is None}")
        if trainer.eval_dataset is not None:
            try:
                print(f"🔍 [DEBUG] trainer.eval_dataset length: {len(trainer.eval_dataset)}")
            except Exception as e:
                print(f"🔍 [DEBUG] Error getting eval_dataset length: {e}")
        
        # Run uncertainty evaluation at the beginning of each epoch
        print(f"🔍 [DEBUG] About to check if eval_dataset is not None...")
        if trainer.eval_dataset is not None:
            print(f"🔍 [DEBUG] ✅ eval_dataset exists - proceeding with uncertainty evaluation")
            print(f"\n📊 Running uncertainty evaluation at beginning of epoch {state.epoch}...")
            metrics = trainer.evaluate_uncertainty()
            
            if metrics is not None:
                # Add epoch information
                metrics['epoch'] = state.epoch
                metrics['global_step'] = state.global_step
                
                # Store results
                trainer.uncertainty_results.append(metrics)
                
                # Print formatted results
                print(f"\n📈 Epoch {state.epoch} Uncertainty Results (Pre-Training):")
                print(f"   Accuracy (ACC): {metrics['accuracy']:.4f}")
                print(f"   Expected Calibration Error (ECE): {metrics['ece']:.4f}")
                print(f"   Negative Log-Likelihood (NLL): {metrics['nll']:.4f}")
                
                # Log uncertainty metrics to tensorboard
                if trainer.args.report_to and 'tensorboard' in trainer.args.report_to:
                    uncertainty_metrics = {}
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            uncertainty_metrics[f"uncertainty_pre_epoch/{key}"] = value
                    trainer.log(uncertainty_metrics)
                
                # Save results to file
                trainer._save_uncertainty_results()
            else:
                print(f"[WARNING] No uncertainty metrics available for epoch {state.epoch}")
        else:
            print(f"🔍 [DEBUG] ❌ eval_dataset is None - skipping uncertainty evaluation")
            print(f"🔍 [DEBUG] This is why you don't see uncertainty evaluation messages!")
            print(f"[INFO] No evaluation dataset available for uncertainty evaluation at epoch {state.epoch}")


class EvalLossComponentsCallback(TrainerCallback):
    """Callback to compute and log evaluation loss components at the end of each epoch."""
    
    def __init__(self):
        super().__init__()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Compute and log evaluation loss components at the end of each epoch."""
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None:
            print("[EvalLossComponentsCallback] No trainer reference found")
            return
        
        print(f"\n🔍 [DEBUG] EvalLossComponentsCallback.on_epoch_end called for epoch {state.epoch}")
        print(f"🔍 [DEBUG] Training args eval_strategy: {trainer.args.eval_strategy}")
        print(f"🔍 [DEBUG] Training args eval_steps: {getattr(trainer.args, 'eval_steps', 'None')}")
        
        # Log training loss components to TensorBoard
        if trainer.args.report_to and 'tensorboard' in trainer.args.report_to:
            training_metrics = {
                'train/ce_loss': trainer.last_ce_loss,
                'train/kl_loss': trainer.last_kl_loss, 
                'train/total_loss': trainer.last_total_loss,
                'train/kl_beta': trainer.beta
            }
            trainer.log(training_metrics)
            print(f"\n📊 Training Loss Components (Epoch {state.epoch}):")
            print(f"   CE Loss: {trainer.last_ce_loss:.4f}")
            print(f"   KL Loss: {trainer.last_kl_loss:.4f}")
            print(f"   Total Loss: {trainer.last_total_loss:.4f}")
            print(f"   KL Beta: {trainer.beta:.4f}")
        
        # Run evaluation and log eval losses
        print(f"\n🔍 [DEBUG] Checking eval_dataset: {trainer.eval_dataset is not None}")
        if trainer.eval_dataset is not None:
            print(f"🔍 [DEBUG] eval_dataset length: {len(trainer.eval_dataset)}")
            print(f"\n📊 Running evaluation after epoch {state.epoch}...")
            
            # Get evaluation metrics including loss
            eval_results = trainer.evaluate()
            
            # Extract eval loss components if available
            eval_loss = eval_results.get('eval_loss', 0.0)
            
            # Run one forward pass on eval set to get loss components
            eval_ce_loss, eval_kl_loss = trainer._compute_eval_loss_components(model)
            
            # Log evaluation loss components to TensorBoard
            if trainer.args.report_to and 'tensorboard' in trainer.args.report_to:
                eval_metrics = {
                    'eval/ce_loss': eval_ce_loss,
                    'eval/kl_loss': eval_kl_loss,
                    'eval/total_loss': eval_loss
                }
                trainer.log(eval_metrics)
                
            print(f"\n📊 Evaluation Loss Components (Epoch {state.epoch}):")
            print(f"   CE Loss: {eval_ce_loss:.4f}")
            print(f"   KL Loss: {eval_kl_loss:.4f}")
            print(f"   Total Loss: {eval_loss:.4f}")
        else:
            print(f"\n🔍 [DEBUG] eval_dataset is None - skipping custom evaluation block")
            print(f"🔍 [DEBUG] This means the evaluation output you see is from HuggingFace's automatic evaluation")


class PredictionTrackerCallback(TrainerCallback):
    """Callback to track predictions on fixed examples across epochs for interpretability."""
    
    def __init__(self, train_dataset, eval_dataset, output_dir, predictions_dir, tokenizer, 
                 n_examples=10, dataset_name="arc_easy"):
        super().__init__()
        # Use the provided predictions directory directly
        self.prediction_tracker = PredictionTracker(
            output_dir=predictions_dir,  # Use standardized predictions directory
            tokenizer=tokenizer,
            n_examples=n_examples,
            dataset_name=dataset_name
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.examples_selected = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Select fixed examples at the start of training."""
        print(f"\n [PredictionTracker] on_train_begin called - examples_selected: {self.examples_selected}")
        print(f" [PredictionTracker] train_dataset is not None: {self.train_dataset is not None}")
        
        if not self.examples_selected and self.train_dataset is not None:
            print(f"\n [PredictionTracker] Selecting {self.prediction_tracker.n_examples} examples for tracking...")
            
            # Select examples from training and validation sets
            if self.eval_dataset is not None:
                print(f" [PredictionTracker] Using both train and eval datasets for example selection")
                self.prediction_tracker.select_examples(self.train_dataset, self.eval_dataset)
            else:
                # If no eval dataset, just use training set
                print(f" [PredictionTracker] Using only train dataset for example selection")
                self.prediction_tracker.select_examples(self.train_dataset, None)
            
            self.examples_selected = True
            print(f" [PredictionTracker] Examples selected and will be tracked every epoch")
        else:
            if self.examples_selected:
                print(f" [PredictionTracker] Examples already selected, skipping selection")
            if self.train_dataset is None:
                print(f" [PredictionTracker] WARNING: train_dataset is None, cannot select examples")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Track predictions at the end of each epoch."""
        if not self.examples_selected:
            print(f"[PredictionTracker] No examples selected, skipping prediction tracking")
            return
        
        model = kwargs.get("model")
        if model is None:
            print(f"[PredictionTracker] No model found, skipping prediction tracking")
            return
        
        epoch = int(state.epoch)
        print(f"\n📝 [PredictionTracker] Saving predictions for epoch {epoch}...")
        
        try:
            # Generate and save predictions for this epoch
            self.prediction_tracker.track_predictions(model, epoch)
            print(f"📝 [PredictionTracker] Predictions saved for epoch {epoch}")
        except Exception as e:
            print(f"[PredictionTracker] Failed to save predictions for epoch {epoch}: {e}")


class LatentPlotCallback(TrainerCallback):
    """Callback to plot latent encodings following DeBERTa pattern."""
    
    def __init__(self, device, output_dir, start_epoch, interval, plot_batch_size):
        super().__init__()
        self.device = device
        self.output_dir = Path(output_dir)
        self.start_epoch = start_epoch
        self.interval = interval
        self.plot_batch_size = plot_batch_size
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Plot latent encodings at specified intervals."""
        current_epoch = int(state.epoch)
        
        # Check if we should plot this epoch
        if current_epoch < self.start_epoch or (current_epoch - self.start_epoch) % self.interval != 0:
            return
        
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None:
            print("[LatentPlotCallback] No trainer reference found")
            return
        
        # Use ard_heldout_loader for plotting - fail if not configured
        if not hasattr(trainer, 'ard_heldout_loader'):
            raise AttributeError("[LatentPlotCallback] trainer.ard_heldout_loader is required but not found. Check trainer configuration.")
        
        eval_data = trainer.ard_heldout_loader
        if eval_data is None:
            raise ValueError("[LatentPlotCallback] trainer.ard_heldout_loader is None. Plotting requires a configured DataLoader.")
        
        if plot_mean_encodings is None:
            print("[LatentPlotCallback] Plotting utilities not available")
            return
        
        print(f"[LatentPlotCallback] Plotting latent encodings at epoch {current_epoch}...")
        
        try:
            # Create plots directory
            plot_dir = self.output_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the existing ARD DataLoader directly since it's already properly configured
            # with the correct batch_size, collate_fn, etc.
            plot_dataloader = eval_data
            
            # Generate plots
            plot_mean_encodings(model, plot_dataloader, self.device, str(plot_dir), epoch=current_epoch)
            print(f"[LatentPlotCallback] Plots saved to {plot_dir}")
        except Exception as e:
            print(f"[LatentPlotCallback] Failed to generate plots: {e}")
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HeldoutResampleCallback(TrainerCallback):
    """Callback to resample ARD samples from training data each epoch for dynamic prior estimation."""
    
    def __init__(self, train_ds, val_ds, ard_prior_samples, batch_size, tokenizer=None, data_collator=None):
        super().__init__()
        self.train_ds = train_ds  # FULL training data - used for ARD sampling
        self.val_ds = val_ds      # FIXED validation data - never changes
        self.ard_prior_samples = ard_prior_samples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data_collator = data_collator  # For creating fresh ARD DataLoaders
        
        # Validate that we have enough training data for ARD sampling
        if self.train_ds is not None and len(self.train_ds) < self.ard_prior_samples:
            print(f"[HeldoutResampleCallback] WARNING: Training dataset has only {len(self.train_ds)} samples, "
                  f"but {self.ard_prior_samples} requested for ARD. Will use all training data for ARD.")
            self.ard_prior_samples = len(self.train_ds)
    
    def _resample_ard_samples(self):
        """Dynamically resample ARD samples from training data each epoch."""
        if self.train_ds is None or len(self.train_ds) == 0:
            print("[HeldoutResampleCallback] No training data available for ARD sampling")
            return None
        
        total_train = len(self.train_ds)
        
        # Ensure we don't exceed training dataset size
        ard_samples = min(self.ard_prior_samples, total_train)
        
        # Randomly shuffle training indices for dynamic sampling
        perm = np.random.permutation(total_train)
        
        # Sample ARD indices from training data
        # Convert numpy indices to Python integers to avoid type errors
        ard_indices = [int(idx) for idx in perm[:ard_samples]]
        
        # Create ARD dataset from training data
        if hasattr(self.train_ds, 'select'):
            # HuggingFace dataset
            ard_dataset = self.train_ds.select(ard_indices)
        else:
            # PyTorch dataset
            ard_dataset = Subset(self.train_ds, ard_indices)

        return ard_dataset
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Resample ARD samples from training data at the beginning of each epoch."""
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None:
            print("[HeldoutResampleCallback] No trainer reference found")
            return
        
        if self.train_ds is None:
            print("[HeldoutResampleCallback] No training dataset available for ARD resampling")
            return
        
        try:
            # Dynamically resample ARD samples from training data
            ard_dataset = self._resample_ard_samples()
            
            if ard_dataset is not None:
                # ✅ KEEP VALIDATION FIXED - Do NOT modify trainer.eval_dataset
                # The validation dataset (self.val_ds) remains unchanged throughout training
                
                # Update trainer's ARD DataLoader with fresh samples from training data
                if self.data_collator is not None:
                    trainer.ard_heldout_loader = DataLoader(
                        ard_dataset,
                        batch_size=self.batch_size,
                        collate_fn=self.data_collator,
                        shuffle=False
                    )
                    print(f"[HeldoutResampleCallback] ✅ Updated ARD DataLoader with {len(ard_dataset)} fresh samples from training data")
                else:
                    print(f"[HeldoutResampleCallback] ⚠️ No data_collator provided - ARD DataLoader not updated")
                
                print(f"[HeldoutResampleCallback] Epoch {int(state.epoch)} → "
                      f"ARD: {len(ard_dataset)} samples (from {len(self.train_ds)} training samples), "
                      f"Fixed Eval: {len(self.val_ds) if self.val_ds else 0} samples (unchanged)")
            
        except Exception as e:
            print(f"[HeldoutResampleCallback] Failed to resample ARD data from training: {e}")


def prepare_validation_dataset(val_dataset, train_dataset=None, train_split_ratio=0.1, seed=42):
    """
    Prepare validation dataset for new 3-way data architecture.
    
    In the NEW architecture:
    - ARD samples come from training data (handled by HeldoutResampleCallback)
    - Validation data stays FIXED for evaluation only
    - No validation splitting needed for ARD
    
    Args:
        val_dataset: Original validation dataset from new data loading (can be None)
        train_dataset: Training dataset (used if val_dataset is None)
        train_split_ratio: Fraction of training data to use for validation when val_dataset is None (typically 10%)
        seed: Random seed for reproducible splits
    
    Returns:
        validation_dataset: Fixed validation dataset for evaluation (no ARD splitting)
    """
    # If validation dataset provided, use it as-is (NEW architecture provides fixed validation)
    if val_dataset is not None and len(val_dataset) > 0:
        print(f"[INFO] Using provided validation dataset: {len(val_dataset)} samples (fixed for evaluation)")
        return val_dataset
    
    # If no validation dataset, create one from training data (fallback case)
    if train_dataset is None or len(train_dataset) == 0:
        print("[WARNING] No validation or training dataset available")
        return None
    
    # Ensure validation split is at least 10% of training data
    min_split_ratio = max(0.1, train_split_ratio)
    
    print(f"[INFO] No validation dataset found. Creating validation split from training data ({min_split_ratio:.1%})")
    
    # Create validation split from training data
    total_train = len(train_dataset)
    # Calculate validation size: use the specified split ratio (typically 10%)
    val_size = int(total_train * min_split_ratio)
    val_size = min(val_size, total_train // 2)  # Don't use more than half of training data
    
    np.random.seed(seed)
    val_indices = np.random.choice(total_train, val_size, replace=False)
    # Convert numpy indices to Python integers to avoid type errors
    val_indices = [int(idx) for idx in val_indices]
    validation_dataset = Subset(train_dataset, val_indices)
    
    print(f"[INFO] Created validation split with {len(validation_dataset)} samples from training data")
    print(f"      → {len(validation_dataset)/total_train:.1%} of training data reserved for validation")
    print(f"      → Validation data will remain FIXED (no ARD splitting)")
    
    return validation_dataset


@torch.no_grad()
def estimate_ard_priors_clm(model, ard_heldout_loader, device, 
                            high_relevance_threshold, medium_relevance_threshold,
                            target_attention_layers, verbose=False):
    """
    Estimate ARD priors following DeBERTa pattern, adapted for LLaMA architecture.
    
    Args:
        model: ARD-LoRA model with ProbLoRA layers
        ard_heldout_loader: Pre-configured DataLoader for ARD prior estimation
        device: Device to run computation on
        high_relevance_threshold: Threshold for high relevance classification (REQUIRED)
        medium_relevance_threshold: Threshold for medium relevance classification (REQUIRED)
        target_attention_layers: List of attention projection names to target (from YAML config - REQUIRED)
    """
    # Calculate total samples from DataLoader
    total_samples = len(ard_heldout_loader.dataset) if hasattr(ard_heldout_loader, 'dataset') else 0
    print(f"[ARD] Estimating priors using all {total_samples} samples from DataLoader...")
    
    # Validate required parameters
    if target_attention_layers is None:
        raise ValueError("target_attention_layers must be provided from YAML configuration")
    
    # Set model to eval mode for prior estimation
    was_training = model.training
    model.eval()
    
    # Use the pre-configured DataLoader directly
    eval_dataloader = ard_heldout_loader
    
    # Collect ProbLoRA layers and initialize beta accumulators (following DeBERTa pattern)
    prob_lora_layers = {}
    
    # Find all ProbLoRA layers in the model (adapted from DeBERTa approach)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer_idx, layer in enumerate(model.model.layers):
            layer_projections = {}
            
            # Check attention projections (configurable from YAML)
            if hasattr(layer, 'self_attn') and target_attention_layers:
                attn = layer.self_attn
                for proj_name in target_attention_layers:
                    if hasattr(attn, proj_name):
                        proj = getattr(attn, proj_name)
                        # Check if this is a ProbLoRA layer (has beta_get_sample method)
                        if hasattr(proj, 'beta_get_sample') and hasattr(proj, 'rank'):
                            layer_projections[f'attn_{proj_name}'] = proj
            
            if layer_projections:
                prob_lora_layers[layer_idx] = layer_projections
    
    if not prob_lora_layers:
        print("[ARD] No ProbLoRA layers found for prior estimation")
        # Restore original training mode
        if was_training:
            model.train()
        return []
    
    # Initialize beta accumulators for each layer (following DeBERTa pattern)
    beta_accumulators = {}
    for layer_idx, projections in prob_lora_layers.items():
        beta_accumulators[layer_idx] = {}
        for proj_name, proj in projections.items():
            beta_accumulators[layer_idx][proj_name] = np.zeros(proj.rank, dtype=np.float32)
    
    sample_count = 0
    
    for batch_idx, batch in enumerate(eval_dataloader):
        
        # Move to device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Get hidden states for each layer (following DeBERTa approach)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        # Get hidden states from model forward pass (similar to DeBERTa encoder outputs)
        if hasattr(model, 'model'):
            hidden_states_outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            hidden_states = hidden_states_outputs.hidden_states
            
            # Process each layer that has ProbLoRA projections (following DeBERTa pattern)
            for layer_idx, projections in prob_lora_layers.items():
                if layer_idx < len(hidden_states):
                    layer_input = hidden_states[layer_idx]
                    
                    # Accumulate beta samples for each projection (following DeBERTa pattern)
                    for proj_name, proj in projections.items():
                        try:
                            beta_sample = proj.beta_get_sample(layer_input)
                            
                            # NUMERICAL STABILITY: Check for inf/nan before accumulation
                            if np.isnan(beta_sample).any() or np.isinf(beta_sample).any():
                                print(f"[ARD] Warning: NaN/Inf in beta sample for layer {layer_idx} {proj_name}, skipping")
                                continue
                            
                            # Safely compute squared samples
                            try:
                                beta_squared = beta_sample ** 2
                                # Check for overflow after squaring
                                if np.isnan(beta_squared).any() or np.isinf(beta_squared).any():
                                    print(f"[ARD] Warning: Overflow in beta^2 for layer {layer_idx} {proj_name}, skipping")
                                    continue
                                
                                beta_accumulators[layer_idx][proj_name] += np.sum(beta_squared, axis=0) / 2.0
                            except (FloatingPointError, OverflowError):
                                print(f"[ARD] Warning: Numerical overflow for layer {layer_idx} {proj_name}, using fallback")
                                # Fallback: use much smaller values to maintain numerical stability
                                beta_accumulators[layer_idx][proj_name] += np.ones(proj.rank, dtype=np.float32) * 0.1
                                
                        except Exception as e:
                            print(f"[ARD] Warning: Failed to get beta sample for layer {layer_idx} {proj_name}: {e}")
                            continue
        
        sample_count += inputs['input_ids'].size(0)
        
        if batch_idx % 10 == 0:
            print(f"   Processed {batch_idx + 1} batches, ~{sample_count} samples")
    
    # Update est_var for each layer (following DeBERTa pattern: est_var = beta / alpha + 1e-6)
    if verbose:
        print(f"\n[ARD] Updating estimated variances for {len(prob_lora_layers)} layers:")
    
    for layer_idx, projections in prob_lora_layers.items():
        for proj_name, proj in projections.items():
            beta_accumulated = beta_accumulators[layer_idx][proj_name]
            
            # NUMERICAL STABILITY: Check beta_accumulated for inf/nan values
            if np.isnan(beta_accumulated).any() or np.isinf(beta_accumulated).any():
                print(f"[ARD] Warning: Invalid beta_accumulated for layer {layer_idx} {proj_name}, using fallback")
                beta_accumulated = np.ones(proj.rank, dtype=np.float32) * 0.1  # Small positive values
            
            # Calculate est_var: beta / alpha + 1e-6 (same as DeBERTa)
            # NUMERICAL STABILITY: Ensure alpha is reasonable and prevent division by zero
            alpha_safe = max(proj.alpha, 1e-6)
            est_var_values = beta_accumulated / alpha_safe + 1e-6
            
            # Check final values before creating tensor
            if np.isnan(est_var_values).any() or np.isinf(est_var_values).any():
                print(f"[ARD] Warning: Invalid est_var for layer {layer_idx} {proj_name}, using default")
                est_var_values = np.ones(proj.rank, dtype=np.float32)  # Default to 1.0
            
            # Create a float32 tensor on the target device for est_var
            est_var_tensor = torch.tensor(est_var_values, dtype=torch.float32, device=device)

            # If the module already has an est_var tensor buffer, try to update it in-place
            try:
                if hasattr(proj, 'est_var') and isinstance(proj.est_var, torch.Tensor):
                    # Attempt an in-place copy while respecting existing dtype/device
                    try:
                        proj.est_var.data = est_var_tensor.to(proj.est_var.device, dtype=proj.est_var.dtype).data
                    except Exception:
                        # Fallback to re-registering the buffer if in-place update fails
                        try:
                            proj.register_buffer('est_var', est_var_tensor)
                        except Exception:
                            # As a last resort, set attribute (works but won't move with .to())
                            setattr(proj, 'est_var', est_var_tensor)
                else:
                    # Register as a buffer so it moves with the module's device/state_dict
                    proj.register_buffer('est_var', est_var_tensor)
            except Exception:
                # Be defensive: ensure there's at least an attribute even if register_buffer fails
                try:
                    proj.est_var = est_var_tensor
                except Exception:
                    # Give up quietly but report
                    print(f"[ARD] Warning: Could not set est_var for projection {proj_name} in layer {layer_idx}")
            
            # Print statistics (only if verbose)
            if verbose:
                avg_est_var = np.mean(est_var_values)
                print(f"   Layer {layer_idx} {proj_name}: avg_est_var={avg_est_var:.6f}")
                
                # ARD relevance determination (using est_var)
                relevance = "High" if avg_est_var > high_relevance_threshold else "Medium" if avg_est_var > medium_relevance_threshold else "Low"
                print(f"     → Estimated relevance: {relevance}")
    
    # Restore original training mode
    if was_training:
        model.train()
    else:
        model.eval()
    
    print(f"[ARD] ✅ Prior estimation completed - updated est_var for all ProbLoRA layers")
    return list(prob_lora_layers.keys())


def optimize_memory_settings(gpu_memory_fraction=0.9):
    """
    Apply memory optimization settings to reduce CUDA OOM errors.
    Call this before training starts.
    
    Args:
        gpu_memory_fraction: Fraction of GPU memory to use (default: 0.9)
    """
    print("[MEMORY] Applying memory optimization settings...")
    
    # Set CUDA memory allocation configuration for fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Disable tokenizer parallelism to prevent memory issues
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Reduce memory fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # Set memory fraction to leave more memory for system and avoid OOM
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            print(f"[MEMORY] Set GPU memory fraction to {gpu_memory_fraction*100:.0f}%")
        except:
            print("[MEMORY] Could not set memory fraction")
        
        # Print current memory status
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        cached_memory = torch.cuda.memory_reserved() / 1024**3
        
        print(f"[MEMORY] GPU Total: {total_memory:.2f} GB")
        print(f"[MEMORY] GPU Allocated: {allocated_memory:.2f} GB")
        print(f"[MEMORY] GPU Cached: {cached_memory:.2f} GB")
        print(f"[MEMORY] GPU Free: {total_memory - cached_memory:.2f} GB")
    
    # Set environment variables for memory optimization
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Reduce tokenizer memory usage
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Reduce fragmentation
    
    print("[MEMORY] Memory optimization settings applied")


def emergency_memory_cleanup():
    """
    Emergency memory cleanup function to call when OOM occurs.
    """
    print("[MEMORY] Emergency memory cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[MEMORY] Emergency cleanup completed")


def create_ard_callbacks(device, output_dir, train_ds=None, val_ds=None, 
                        ard_prior_samples=1000, batch_size=4, tokenizer=None, data_collator=None,
                        enable_plotting=True, enable_resampling=True, enable_prediction_tracking=True,
                        plot_start_epoch=2, plot_interval=2, plot_batch_size=16,
                        prediction_n_examples=10, dataset_name="arc_easy", predictions_dir=None):
    """Create standard ARD training callbacks for NEW 3-way data architecture.
    
    NEW DATA ARCHITECTURE:
    - train_ds: FULL training dataset used for both SGD and dynamic ARD sampling
    - val_ds: FIXED validation dataset used only for evaluation (never changes)
    - ARD samples: Dynamically sampled from train_ds each epoch by HeldoutResampleCallback
    
    Args:
        device: Device for computation
        output_dir: Output directory for plots and logs
        train_ds: Full training dataset (for ARD dynamic sampling via HeldoutResampleCallback)
        val_ds: Fixed validation dataset (for evaluation only - never modified)
        ard_prior_samples: Number of samples for ARD prior estimation (sampled from train_ds)
        batch_size: Batch size for data loaders
        tokenizer: Tokenizer for data collation
        data_collator: Data collator for dynamic ARD DataLoader creation
        enable_plotting: Whether to enable latent plotting
        enable_resampling: Whether to enable held-out resampling from training data
        enable_prediction_tracking: Whether to enable prediction tracking on fixed examples
        plot_start_epoch: Epoch to start plotting
        plot_interval: Interval between plots
        prediction_n_examples: Number of examples to track for predictions
        dataset_name: Name of dataset for prediction formatting
    
    Returns:
        List of callback instances configured for new data architecture
    """
    callbacks = []
    
    # Add resampling callback if enabled and datasets available
    if enable_resampling and train_ds is not None:
        callbacks.append(HeldoutResampleCallback(
            train_ds=train_ds,        # ✅ Full training data for ARD sampling
            val_ds=val_ds,            # ✅ Fixed validation data (never changes)
            ard_prior_samples=ard_prior_samples,
            batch_size=batch_size,
            tokenizer=tokenizer,
            data_collator=data_collator  # For dynamic ARD DataLoader creation
        ))
        
        print(f"[INFO] HeldoutResampleCallback configured:")
        print(f"   • Will sample {ard_prior_samples} ARD samples from {len(train_ds)} training samples each epoch")
        print(f"   • Validation data remains fixed: {len(val_ds) if val_ds else 0} samples")

    # Always add uncertainty evaluation callback
    callbacks.append(UncertaintyEvaluationCallback())
    
    # Always add eval loss components callback
    callbacks.append(EvalLossComponentsCallback())

    # Always add prior estimation callback
    callbacks.append(PriorEstimationCallback(device))
    
    # Add prediction tracking callback if enabled and datasets available
    if enable_prediction_tracking and train_ds is not None and tokenizer is not None:
        # Use predictions_dir if provided, otherwise create subdirectory under output_dir
        pred_dir = predictions_dir if predictions_dir is not None else str(Path(output_dir) / "predictions")
        
        callbacks.append(PredictionTrackerCallback(
            train_dataset=train_ds,
            eval_dataset=val_ds,
            output_dir=output_dir,
            predictions_dir=pred_dir,  # Pass standardized predictions directory
            tokenizer=tokenizer,
            n_examples=prediction_n_examples,
            dataset_name=dataset_name
        ))
        print(f"[INFO] PredictionTrackerCallback configured:")
        print(f"   • Will track {prediction_n_examples} examples from train/val sets")
        print(f"   • Predictions saved to: {pred_dir}")
    
    # Add plotting callback if enabled and utilities available
    if enable_plotting and plot_mean_encodings is not None:
        callbacks.append(LatentPlotCallback(
            device=device,
            output_dir=output_dir,
            start_epoch=plot_start_epoch,
            interval=plot_interval,
            plot_batch_size=plot_batch_size
        ))

    return callbacks


def build_clm_trainer(model, tokenizer, train_dataset, eval_dataset, cfg, output_dir, 
                     ard_prior_samples, enable_callbacks, tb_log_dir, predictions_dir):
    """Build enhanced CLM trainer with uncertainty evaluation, ARD callbacks, and prior estimation.
    
    NEW DATA ARCHITECTURE:
    - train_dataset: Full training data (used for both SGD and ARD dynamic sampling)
    - eval_dataset: Fixed validation data from new 3-tuple loading (evaluation only)
    - ARD samples: Dynamically created from train_dataset by HeldoutResampleCallback each epoch
    """
    
    # NEW: Prepare validation dataset (no ARD splitting needed)
    # eval_dataset should be the fixed validation data from new 3-tuple loading
    validation_dataset = prepare_validation_dataset(
        eval_dataset, 
        train_dataset=train_dataset,
        train_split_ratio=cfg["validation_split_ratio"],
        seed=cfg["random_seed"]
    )
    
    # Use TensorBoard log directory from get_output_dirs() - required parameter
    if tb_log_dir is None:
        raise ValueError("tb_log_dir must be provided from get_output_dirs()")
    
    # Training arguments - ALL VALUES FROM CONFIG FILE (NO DEFAULTS)
    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=tb_log_dir,  # TensorBoard logging directory from get_output_dirs()
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["train_epochs"],
        bf16=cfg["bf16"],
        fp16=cfg["fp16"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        eval_strategy=cfg["eval_strategy"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg.get("eval_steps"),  # Optional parameter
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg.get("metric_for_best_model"),  # Optional parameter
        report_to=cfg.get("report_to"),  # Optional parameter
        remove_unused_columns=cfg["remove_unused_columns"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        dataloader_pin_memory=cfg["dataloader_pin_memory"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs=cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        group_by_length=cfg["group_by_length"],  # Group samples by length for efficiency
    )

    # Data collator for causal language modeling with padding optimization
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, 
        mlm=False,
        pad_to_multiple_of=cfg.get("pad_to_multiple_of")  # Optimize for A100 tensor cores
    )

    # NEW: ARD DataLoader will be created dynamically by HeldoutResampleCallback
    # No static ARD DataLoader from validation splits
    ard_heldout_loader = None  # Will be updated by HeldoutResampleCallback with training data
    
    print(f"[INFO] NEW Data Architecture:")
    print(f"   • Training dataset: {len(train_dataset)} samples (for SGD + dynamic ARD sampling)")
    print(f"   • Validation dataset: {len(validation_dataset) if validation_dataset else 0} samples (fixed for evaluation)")
    print(f"   • ARD samples: {ard_prior_samples} (will be dynamically sampled from training data each epoch)")

    # DEBUG: Analyze dataset filtering issues if enabled
    if cfg.get("debug_dataset_filtering", False):
        print("\n🔍 DATASET FILTERING DEBUG ANALYSIS ENABLED")
        print("=" * 60)
        
        # Debug training dataset
        if train_dataset is not None:
            print("\n📊 TRAINING DATASET ANALYSIS:")
            debug_stats = debug_dataset_filtering(
                train_dataset, 
                tokenizer, 
                max_len=cfg["max_len"],
                sample_size=min(1000, len(train_dataset))
            )
            
            # Test data collator
            collator_stats = analyze_data_collator_filtering(
                train_dataset,
                tokenizer,
                batch_size=cfg["batch_size"],
                max_len=cfg["max_len"],
                num_batches=10
            )
            
        # Debug validation dataset
        if validation_dataset is not None:
            print("\n📊 VALIDATION DATASET ANALYSIS:")
            debug_dataset_filtering(
                validation_dataset, 
                tokenizer, 
                max_len=cfg["max_len"],
                sample_size=min(250, len(validation_dataset))
            )

    # Create the enhanced trainer - ALL VALUES FROM CONFIG FILE
    trainer = ARDCLMTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,  # Use fixed validation dataset for standard evaluation
        data_collator=data_collator,
        tokenizer=tokenizer,
        beta=cfg["kl_loss_beta"],
        ard_heldout_loader=ard_heldout_loader,  # Will be updated by callbacks
        n_bins=cfg["uncertainty_n_bins"],
        output_dir=output_dir,
        ard_prior_samples=ard_prior_samples,
        target_attention_layers=cfg["target_attention_layers"],  # REQUIRED from YAML config
        verbose=False  # Set verbose mode to False to suppress debug messages
    )
    
    # Set relevance thresholds on trainer for ARD prior estimation
    trainer.high_relevance_threshold = cfg["ard_high_relevance_threshold"]
    trainer.medium_relevance_threshold = cfg["ard_medium_relevance_threshold"]
    
    # DEBUG: Get effective dataset size after trainer creation
    if cfg.get("debug_dataset_filtering", False):
        effective_stats = get_effective_dataset_size(trainer)
        
        # Print summary of filtering analysis
        print("\n📋 FILTERING ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"🔍 Issue: {effective_stats.get('filtering_percentage', 0):.1f}% of training data is being filtered out")
        print(f"📊 Effective training samples: {effective_stats.get('effective_dataset_size', 0):,} / {effective_stats.get('reported_dataset_size', 0):,}")
        print(f"⏱️  Training steps: {effective_stats.get('dataloader_batch_count', 0):,}")
        print()
        print("🔧 RECOMMENDED SOLUTIONS:")
        print("   1. Add 'debug_dataset_filtering: true' to your config to see detailed analysis")
        print("   2. Check sequence length distribution in your dataset")
        print("   3. Consider reducing max_len if most sequences are shorter")
        print("   4. Verify tokenization is working correctly")
        print("   5. Check for empty or corrupted samples in your dataset")
        print("   6. Use apply_dataset_filtering_fixes() function to clean your dataset")
        print()
    # GRADIENT-CHECKPOINTING FIX: ensure at least one checkpointed input has requires_grad=True
    # Register a small embed forward-hook and per-layer forward_pre_hooks. Track how many
    # hooks succeeded so we can print a coverage summary. Optionally enable a debug wrapper
    # for torch.utils.checkpoint.checkpoint via cfg['checkpoint_debug'] to log offending calls.
    try:
        hook_embed_attached = 0
        hook_layer_attached = 0

        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            def _embed_forward_hook(module, inp, out):
                try:
                    if isinstance(out, torch.Tensor):
                        out.requires_grad_(True)
                except Exception:
                    pass
            try:
                model.model.embed_tokens.register_forward_hook(_embed_forward_hook)
                hook_embed_attached = 1
            except Exception:
                hook_embed_attached = 0

        if hasattr(model.model, 'layers'):
            def _make_pre_hook():
                def _pre_hook(module, inputs):
                    try:
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            if not inputs[0].requires_grad:
                                inputs[0].requires_grad_(True)
                    except Exception:
                        pass
                return _pre_hook

            for lyr in model.model.layers:
                try:
                    lyr.register_forward_pre_hook(_make_pre_hook())
                    hook_layer_attached += 1
                except Exception:
                    continue

        print(f"[GRADIENT FIX] Hook coverage: embed_hook={hook_embed_attached}, layer_pre_hooks={hook_layer_attached}/{len(getattr(model.model, 'layers', []))}")

        # Optional debug wrapper for checkpoint to log inputs when no requires_grad present
        try:
            debug_flag = False
            if isinstance(cfg, dict):
                debug_flag = bool(cfg.get('checkpoint_debug', False))
            # Fall back to environment variable CHECKPOINT_DEBUG=1 (useful for full training runs)
            if not debug_flag:
                try:
                    import os as _os
                    if _os.environ.get('CHECKPOINT_DEBUG', '').lower() in ('1', 'true', 'yes'):
                        debug_flag = True
                except Exception:
                    pass

            if debug_flag:
                import traceback as _traceback
                _orig_checkpoint = torch.utils.checkpoint.checkpoint

                if not getattr(torch.utils.checkpoint, '_checkpoint_debug_wrapped', False):
                    def _debug_checkpoint(func, *args, **kwargs):
                        # gather tensors
                        tensors = []
                        def _g(x):
                            if isinstance(x, torch.Tensor):
                                tensors.append(x)
                            elif isinstance(x, (list, tuple)):
                                for e in x: _g(e)
                            elif isinstance(x, dict):
                                for e in x.values(): _g(e)
                        _g(args)
                        _g(kwargs)

                        any_req = False
                        for t in tensors:
                            try:
                                if t is not None and t.is_floating_point() and t.requires_grad:
                                    any_req = True
                                    break
                            except Exception:
                                continue

                        if not any_req:
                            print('[CHECKPOINT_DEBUG] checkpoint called with NO inputs requiring grad')
                            for i, t in enumerate(tensors[:12]):
                                try:
                                    print(f'  tensor[{i}]: shape={tuple(t.shape) if t is not None else None}, dtype={t.dtype if t is not None else None}, requires_grad={t.requires_grad if t is not None else None}')
                                except Exception:
                                    pass
                            _traceback.print_stack(limit=6)

                        return _orig_checkpoint(func, *args, **kwargs)

                    torch.utils.checkpoint.checkpoint = _debug_checkpoint
                    torch.utils.checkpoint._checkpoint_debug_wrapped = True
                    print('[CHECKPOINT_DEBUG] Wrapped torch.utils.checkpoint.checkpoint with debug logger')

        except Exception as e:
            print(f"[CHECKPOINT_DEBUG] Failed to install debug wrapper: {e}")

    except Exception as e:
        print(f"[GRADIENT FIX] Failed to register checkpoint hooks: {e}")
    
    # Enable gradient checkpointing for memory efficiency based on configuration
    if cfg.get("gradient_checkpointing", True) and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("[MEMORY] Enabled gradient checkpointing based on configuration")
    elif hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        print("[MEMORY] Disabled gradient checkpointing based on configuration")
    else:
        print(f"[MEMORY] Gradient checkpointing setting: {cfg.get('gradient_checkpointing', True)}")
    
    # Memory optimization: Clear cache and show memory status
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[MEMORY] GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
        print(f"[MEMORY] GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Add ARD callbacks if enabled
    if enable_callbacks:
        device = next(model.parameters()).device
        
        # NEW: Pass correct datasets for new architecture
        callbacks = create_ard_callbacks(
            device=device,
            output_dir=output_dir,
            train_ds=train_dataset,        # ✅ Full training data for ARD dynamic sampling
            val_ds=validation_dataset,     # ✅ Fixed validation data for evaluation
            ard_prior_samples=ard_prior_samples,  # Use parameter passed to function
            batch_size=cfg["batch_size"],
            tokenizer=tokenizer,
            data_collator=data_collator,  # Pass data_collator for dynamic ARD DataLoader creation
            enable_plotting=cfg["enable_plotting"],
            enable_resampling=cfg["enable_resampling"],
            enable_prediction_tracking=cfg.get("enable_prediction_tracking"),  # Default to True
            plot_start_epoch=cfg["plot_start_epoch"],
            plot_interval=cfg["plot_interval"],
            plot_batch_size=cfg["plot_batch_size"],
            prediction_n_examples=cfg.get("prediction_n_examples"),  # Default to 10
            dataset_name=cfg.get("dataset_name", "arc_easy"),  # Default dataset name
            predictions_dir=predictions_dir  # Pass predictions directory from get_output_dirs()
        )
        
        # Add callbacks to trainer
        for callback in callbacks:
            trainer.add_callback(callback)
            
        print(f"[INFO] Added {len(callbacks)} ARD callbacks to trainer")
        for callback in callbacks:
            print(f"   - {callback.__class__.__name__}")
        
        print(f"\n[INFO] Callback Configuration:")
        print(f"   • HeldoutResampleCallback: ARD samples from {len(train_dataset)} training samples")
        print(f"   • UncertaintyEvaluationCallback: Uses {len(validation_dataset) if validation_dataset else 0} fixed validation samples")
        print(f"   • PriorEstimationCallback: Uses dynamic ARD samples from training data")
        print(f"   • LatentPlotCallback: Uses dynamic ARD samples from training data")
        print(f"   • PredictionTrackerCallback: Tracks {cfg.get('prediction_n_examples')} examples every epoch")
    
    return trainer
