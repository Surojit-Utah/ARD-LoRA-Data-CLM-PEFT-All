
"""
ARD-LoRA Classification Trainer for Multiple Choice QA
======================================================

This module provides a specialized trainer for classification tasks like ARC-Easy,
where we predict a single answer from K choices using last-token logits.

Key Differences from Standard CLM:
1. Extract ONLY last token logits (not full sequence)
2. Filter logits to K answer tokens (e.g., A, B, C, D, E)
3. Compute CE loss over K classes (not full vocabulary)
4. KL divergence computed same way (over hidden states)
"""

import torch
from torch import nn
from typing import Dict, Any, Optional
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback
from evaluate.uncertainty_metrics import UncertaintyEvaluator
from evaluate.prediction_tracker import PredictionTracker
import numpy as np
from pathlib import Path


class ARDClassificationTrainer(Trainer):
    """
    Enhanced ARD Trainer for classification tasks with last-token prediction.
    
    Adapted from ARDCLMTrainer but with classification-specific loss computation.
    """
    
    def __init__(
        self,
        *args,
        config=None,  # NEW: Accept config dict
        beta=None,
        ard_heldout_loader=None,
        n_bins=None,
        output_dir=None,
        ard_prior_samples=None,
        target_attention_layers=None,
        target_ids=None,  # NEW: Token IDs for valid answers
        num_classes=None,    # NEW: Number of classes (K)
        train_dataset_for_weights=None,  # NEW: Training dataset for class weight computation
        verbose=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Instance variable to control KL loss computation
        self.use_kl = False  # Set to True to enable KL divergence loss
        
        # Load parameters from config with fallbacks
        if config is not None:
            self.beta = beta if beta is not None else config.get('kl_loss_beta')
            self.n_bins = n_bins if n_bins is not None else config.get('uncertainty_n_bins')
            self.ard_prior_samples = ard_prior_samples if ard_prior_samples is not None else config.get('ard_prior_samples')
            self.num_classes = num_classes if num_classes is not None else config.get('num_classes')
            self.verbose = verbose if verbose is not None else config.get('verbose')
            
            # Target attention layers
            if target_attention_layers is None:
                target_attention_layers = config.get('target_attention_layers')
        else:
            self.beta = beta if beta is not None else 0.01
            self.n_bins = n_bins if n_bins is not None else 15
            self.ard_prior_samples = ard_prior_samples if ard_prior_samples is not None else 100
            self.num_classes = num_classes if num_classes is not None else 4  # Default to 4 (A, B, C, D)
            self.verbose = verbose
            
            if target_attention_layers is None:
                raise ValueError("target_attention_layers must be provided")
        
        self.ard_heldout_loader = ard_heldout_loader
        self.uncertainty_evaluator = UncertaintyEvaluator(n_bins=self.n_bins)
        self.uncertainty_results = []
        self.output_dir = output_dir or self.args.output_dir
        
        # Classification-specific parameters
        if target_ids is None:
            raise ValueError("target_ids must be provided for classification mode")
        self.target_ids = target_ids  # Tensor of shape [num_classes]
        
        # Compute class weights for imbalanced datasets
        if train_dataset_for_weights is not None and self.num_classes is not None:
            self.class_weights = self._compute_class_weights(train_dataset_for_weights, self.num_classes)
            print(f"\n[CLASS WEIGHTS] Computed from {len(train_dataset_for_weights)} training samples:")
            for i, w in enumerate(self.class_weights):
                print(f"  Class {i}: weight={w:.4f}")
        else:
            self.class_weights = None
            print(f"\n[CLASS WEIGHTS] No class weighting applied (balanced CE loss)")
        
        # Store layer configuration
        self.target_attention_layers = target_attention_layers
        
        # Set trainer reference on model for callbacks
        self.model.trainer = self
        
        # Track loss components for logging
        self.last_ce_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_total_loss = 0.0
        
        print(f"[CLASSIFICATION] ARDClassificationTrainer initialized:")
        print(f"[CLASSIFICATION]   Num classes: {self.num_classes}")
        print(f"[CLASSIFICATION]   Target IDs: {self.target_ids.tolist()}")
        print(f"[CLASSIFICATION]   KL beta: {self.beta}")
        print(f"[CLASSIFICATION]   Use KL loss: {self.use_kl}")
    
    def _compute_class_weights(self, dataset, num_classes):
        """
        Compute inverse frequency class weights for imbalanced datasets.
        
        Formula: weight[i] = total_samples / (num_classes * count[i])
        
        Args:
            dataset: Training dataset with 'classes' or 'labels' field
            num_classes: Number of classes
        
        Returns:
            torch.Tensor: Class weights of shape [num_classes]
        """
        import torch
        from collections import Counter
        
        # Extract all class labels from dataset
        all_classes = []
        for i in range(len(dataset)):
            sample = dataset[i]
            # Support both 'classes' and 'labels' fields
            if 'classes' in sample:
                all_classes.append(sample['classes'])
            elif 'labels' in sample:
                all_classes.append(sample['labels'])
            else:
                raise ValueError(f"Dataset must have 'classes' or 'labels' field, got: {list(sample.keys())}")
        
        # Count class frequencies
        class_counts = Counter(all_classes)
        total_samples = len(all_classes)
        
        # Compute inverse frequency weights
        weights = torch.zeros(num_classes)
        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 0)
            if count > 0:
                weights[class_idx] = total_samples / (num_classes * count)
            else:
                weights[class_idx] = 0.0  # No samples for this class
        
        return weights
    
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        """
        Compute classification loss with last-token prediction.
        
        Key Steps:
        1. Forward pass to get full logits [batch, seq_len, vocab_size]
        2. Extract ONLY last token logits [batch, vocab_size]
        3. Filter to K answer tokens [batch, num_classes]
        4. Compute CE loss over K classes
        5. Compute KL divergence over hidden states (same as CLM)
        """
        # Extract gold classes (class indices)
        # S2ClassDataset uses 'labels', our custom datasets might use 'classes'
        # Support both for compatibility
        if "classes" in inputs:
            classes = inputs.pop("classes")
        elif "labels" in inputs:
            classes = inputs.pop("labels")
        else:
            raise ValueError("Classification trainer requires 'classes' or 'labels' in inputs")
        
        # CRITICAL: Verify model is in training mode
        if not hasattr(self, '_training_mode_checked'):
            self._training_mode_checked = True
            print(f"\n[CRITICAL] Model training mode in compute_loss: {model.training}")
            # Check if any parameters require gradients
            trainable = sum(1 for p in model.parameters() if p.requires_grad)
            print(f"[CRITICAL] Trainable parameters in compute_loss: {trainable}")
        
        # Forward pass with hidden states for both CE loss and KL computation
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False
        )
        
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        hidden_states = outputs.hidden_states
        
        # ===== CLASSIFICATION-SPECIFIC LOGIC =====
        
        # Step 1: Extract ONLY the last token logits
        # CRITICAL FIX: Use attention mask to find last non-pad token per example
        attn = inputs["attention_mask"]  # [batch_size, seq_len]
        last_idx = attn.long().sum(dim=1) - 1  # [batch_size]
        batch_indices = torch.arange(logits.size(0), device=logits.device)  # [batch_size]
        last_token_logits = logits[batch_indices, last_idx, :]  # [batch_size, vocab_size]
        
        # Step 2: Filter to ONLY valid answer tokens
        # target_ids shape: [num_classes] (e.g., [319, 350, 315, 360, 382] for A,B,C,D,E)
        target_ids_device = self.target_ids.to(last_token_logits.device)
        filtered_logits = last_token_logits[:, target_ids_device.squeeze()]
        # Shape: [batch_size, num_classes]
        
        # Step 3: Compute cross-entropy loss over K classes (with optional class weighting)
        if self.class_weights is not None:
            weights = self.class_weights.to(filtered_logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
        ce_loss = loss_fct(filtered_logits, classes)
        
        # Debug info (print once per epoch)
        if not hasattr(self, '_debug_classification_printed'):
            self._debug_classification_printed = True
            print(f"\n[CLASSIFICATION LOSS DEBUG]:")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Last token logits shape: {last_token_logits.shape}")
            print(f"  Filtered logits shape: {filtered_logits.shape}")
            print(f"  Classes shape: {classes.shape}")
            print(f"  Classes: {classes.tolist()}")
            print(f"  CE Loss: {ce_loss.item():.4f}")
        
        # ===== KL DIVERGENCE COMPUTATION (SAME AS CLM) =====
        
        kl = 0.0
        
        if self.use_kl:
            # KL computation only when enabled
            kl_debug_info = {}
            total_kl_layers = 0
            
            # Track current epoch for debug printing
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
            
            # Compute KL divergence over ProbLoRA layers
            if hidden_states is not None and hasattr(model, 'model') and hasattr(model.model, 'layers'):
                for layer_idx, layer in enumerate(model.model.layers):
                    layer_input = hidden_states[layer_idx] if layer_idx < len(hidden_states) else None
                    
                    if layer_input is not None:
                        if hasattr(layer, 'self_attn') and self.target_attention_layers:
                            attn = layer.self_attn
                            layer_kl_total = 0.0
                            layer_proj_count = 0
                            
                            for proj_name in self.target_attention_layers:
                                if hasattr(attn, proj_name):
                                    proj = getattr(attn, proj_name)
                                    if hasattr(proj, 'kl_divergence_latent'):
                                        try:
                                            proj_kl = proj.kl_divergence_latent(layer_input)
                                            kl += proj_kl
                                            layer_kl_total += proj_kl.item() if torch.is_tensor(proj_kl) else float(proj_kl)
                                            layer_proj_count += 1
                                        except Exception:
                                            continue
                            
                            if layer_proj_count > 0:
                                kl_debug_info[f"layer_{layer_idx}"] = {
                                    "projections_processed": layer_proj_count,
                                    "target_projections": list(self.target_attention_layers),
                                    "layer_kl_total": layer_kl_total
                                }
                                total_kl_layers += 1
            
            # If no KL components found, create zero tensor with gradient connection
            # Safe check to avoid "Boolean value of Tensor is ambiguous" error
            if not torch.is_tensor(kl) or (torch.is_tensor(kl) and float(kl.detach().item()) == 0.0):
                kl = torch.tensor(0.0, device=ce_loss.device, requires_grad=True)
        
        # Combine losses
        if self.use_kl:
            total_loss = ce_loss + self.beta * kl
        else:
            total_loss = ce_loss  # Keep tensor, don't add scalar 0.0
        
        # Store for logging
        self.last_ce_loss = ce_loss.item()
        self.last_kl_loss = kl.item() if torch.is_tensor(kl) else float(kl) if self.use_kl else 0.0
        self.last_total_loss = total_loss.item()
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to add gradient sanity check after first backward.
        
        Args:
            model: The model being trained
            inputs: The inputs and targets for the model
            num_items_in_batch: Number of items in the batch (for gradient accumulation)
        """
        # Call parent training_step (this does forward, backward, optimizer step)
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Gradient sanity check: run once after first backward pass
        if not hasattr(self, '_gradient_sanity_checked'):
            self._gradient_sanity_checked = True
            print("\n" + "="*60)
            print("[GRAD SANITY CHECK] After first backward pass")
            print("="*60)
            
            nz, tot = 0, 0
            for n, p in model.named_parameters():
                if p.requires_grad:
                    tot += 1
                    if p.grad is not None and torch.count_nonzero(p.grad).item() > 0:
                        nz += 1
            
            print(f"[GRAD] Trainable parameters with nonzero gradients: {nz}/{tot}")
            
            if nz == 0:
                print("[GRAD] ⚠️  WARNING: NO gradients computed! Check loss computation.")
            elif nz < tot:
                print(f"[GRAD] ⚠️  WARNING: Only {nz}/{tot} parameters have gradients!")
            else:
                print(f"[GRAD] ✅ All trainable parameters have gradients.")
            
            # Optional: Show a few gradient samples
            print(f"[GRAD] Sample gradients (first 3 LoRA parameters):")
            count = 0
            for n, p in model.named_parameters():
                if p.requires_grad and 'lora' in n.lower() and p.grad is not None:
                    grad_norm = p.grad.norm().item()
                    print(f"[GRAD]   {n}: grad_norm={grad_norm:.6f}")
                    count += 1
                    if count >= 3:
                        break
            print("="*60 + "\n")
        
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step for classification tasks.
        
        Returns predictions as class probabilities.
        """
        # Debug: Print what fields are in inputs (once)
        if not hasattr(self, '_debug_eval_inputs_printed'):
            self._debug_eval_inputs_printed = True
            print(f"\n[EVAL DEBUG] Input keys during evaluation: {list(inputs.keys())}")
            if "classes" in inputs:
                print(f"[EVAL DEBUG] Found 'classes' field with shape: {inputs['classes'].shape}")
            if "labels" in inputs:
                print(f"[EVAL DEBUG] Found 'labels' field with shape: {inputs['labels'].shape}")
        
        # Extract classes
        classes = inputs.pop("classes") if "classes" in inputs else None
        if classes is None:
            classes = inputs.pop("labels") if "labels" in inputs else None
        
        has_labels = classes is not None
        
        if not has_labels:
            print(f"[EVAL WARNING] No classes or labels found in evaluation batch!")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False, use_cache=False)
            logits = outputs.logits
            
            # Extract last token and filter to answer tokens
            # CRITICAL FIX: Use attention mask to find last non-pad token per example
            attn = inputs["attention_mask"]  # [batch_size, seq_len]
            last_idx = attn.long().sum(dim=1) - 1  # [batch_size]
            batch_indices = torch.arange(logits.size(0), device=logits.device)  # [batch_size]
            last_token_logits = logits[batch_indices, last_idx, :]  # [batch_size, vocab_size]
            target_ids_device = self.target_ids.to(last_token_logits.device)
            filtered_logits = last_token_logits[:, target_ids_device.squeeze()]
            
            # Compute loss if we have labels
            if has_labels:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(filtered_logits, classes)
            else:
                loss = None
        
        # Convert logits to predictions (class indices)
        preds = torch.argmax(filtered_logits, dim=-1)
        
        # Return in format expected by Trainer
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, preds, classes)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Enhanced evaluation with classification metrics.
        """
        # Call parent evaluate
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Add classification-specific metrics if needed
        # (accuracy, F1, etc. can be added here)
        
        return metrics
    
    def evaluate_uncertainty(self) -> Optional[Dict[str, float]]:
        """
        Evaluate model uncertainty using ACC, ECE, and NLL metrics.
        Classification-specific version using last-token prediction.
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
        
        # Create evaluation dataloader
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
                
                # Extract classes
                if "classes" in inputs:
                    classes = inputs.pop("classes")
                elif "labels" in inputs:
                    classes = inputs.pop("labels")
                else:
                    continue
                
                # Forward pass
                outputs = self.model(**inputs, output_hidden_states=False, use_cache=False)
                logits = outputs.logits
                
                # Extract last token logits and filter to answer tokens (classification-specific)
                attn = inputs["attention_mask"]
                last_idx = attn.long().sum(dim=1) - 1
                batch_indices = torch.arange(logits.size(0), device=logits.device)
                last_token_logits = logits[batch_indices, last_idx, :]
                
                # Filter to answer tokens only
                target_ids_device = self.target_ids.to(last_token_logits.device)
                filtered_logits = last_token_logits[:, target_ids_device.squeeze()]
                
                # Convert logits to probabilities
                probs = torch.softmax(filtered_logits, dim=-1)
                
                # Collect predictions
                all_labels.extend(classes.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                sample_count += len(classes)
                
                # Log progress every 100 batches instead of every 10
                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(f"   Processed {batch_idx + 1} batches, {sample_count} samples")
        
        if len(all_labels) == 0:
            print("[WARNING] No valid predictions found for uncertainty evaluation")
            return None
            
        print(f"✅ Uncertainty evaluation completed on {len(all_labels)} samples")
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_probs = np.array(all_probs)
        
        # Compute uncertainty metrics
        metrics = self.uncertainty_evaluator.evaluate_predictions(y_true, y_probs)
        
        return metrics
    
    def _save_uncertainty_results(self):
        """Save uncertainty evaluation results to JSON file."""
        import json
        
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
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert the results to JSON-serializable format
        json_safe_results = convert_numpy_types(self.uncertainty_results)
        
        with open(results_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        print(f"💾 Saved uncertainty results to {results_path}")
    
    def _compute_eval_loss_components(self, model):
        """
        Compute evaluation loss components (CE and KL) on eval dataset.
        Similar to training loss computation but in eval mode.
        """
        if self.eval_dataset is None:
            return 0.0, 0.0
        
        was_training = model.training
        model.eval()
        
        try:
            # Get a batch from eval dataset
            eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
            batch = next(iter(eval_dataloader))
            
            # Move batch to device
            batch = {k: v.to(self.args.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with torch.no_grad():
                # Extract classes
                if "classes" in batch:
                    classes = batch.pop("classes")
                elif "labels" in batch:
                    classes = batch.pop("labels")
                else:
                    return 0.0, 0.0
                
                # Forward pass
                outputs = model(
                    **batch,
                    output_hidden_states=True,
                    use_cache=False
                )
                logits = outputs.logits
                hidden_states = outputs.hidden_states
                
                # Extract last token logits
                attn = batch["attention_mask"]
                last_idx = attn.long().sum(dim=1) - 1
                batch_indices = torch.arange(logits.size(0), device=logits.device)
                last_token_logits = logits[batch_indices, last_idx, :]
                
                # Filter to answer tokens
                target_ids_device = self.target_ids.to(last_token_logits.device)
                filtered_logits = last_token_logits[:, target_ids_device.squeeze()]
                
                # Compute CE loss
                loss_fct = nn.CrossEntropyLoss()
                ce_loss = loss_fct(filtered_logits, classes)
                
                # Compute KL loss (if enabled)
                kl = 0.0
                if self.use_kl:
                    if hidden_states is not None and hasattr(model, 'model') and hasattr(model.model, 'layers'):
                        for layer_idx, layer in enumerate(model.model.layers):
                            layer_input = hidden_states[layer_idx] if layer_idx < len(hidden_states) else None
                            if layer_input is not None:
                                if hasattr(layer, 'self_attn') and self.target_attention_layers:
                                    attn_layer = layer.self_attn
                                    for proj_name in self.target_attention_layers:
                                        if hasattr(attn_layer, proj_name):
                                            proj = getattr(attn_layer, proj_name)
                                            if hasattr(proj, 'kl_divergence_latent'):
                                                try:
                                                    kl_val = proj.kl_divergence_latent(layer_input)
                                                    if torch.is_tensor(kl_val):
                                                        kl = kl + kl_val
                                                except Exception:
                                                    pass
                
                if not torch.is_tensor(kl):
                    kl = torch.tensor(0.0, device=ce_loss.device)
                
                ce_loss_val = ce_loss.item()
                kl_loss_val = kl.item() if torch.is_tensor(kl) else 0.0
                
                return ce_loss_val, kl_loss_val
        
        except Exception as e:
            print(f"⚠️ Failed to compute eval loss components: {e}")
            return 0.0, 0.0
        finally:
            if was_training:
                model.train()
            else:
                model.eval()


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
        
        print(f"\n📊 [EVAL] Logging metrics for epoch {state.epoch}")
        
        # Log training loss components to TensorBoard
        if trainer.args.report_to and 'tensorboard' in trainer.args.report_to:
            training_metrics = {
                'train/ce_loss': trainer.last_ce_loss,
                'train/total_loss': trainer.last_total_loss,
            }
            
            # Only log KL loss if use_kl is enabled
            if trainer.use_kl:
                training_metrics['train/kl_loss'] = trainer.last_kl_loss
                training_metrics['train/kl_beta'] = trainer.beta
            
            trainer.log(training_metrics)
            print(f"📊 Training Loss Components (Epoch {state.epoch}):")
            print(f"   CE Loss: {trainer.last_ce_loss:.4f}")
            if trainer.use_kl:
                print(f"   KL Loss: {trainer.last_kl_loss:.4f}")
                print(f"   Total Loss: {trainer.last_total_loss:.4f}")
                print(f"   KL Beta: {trainer.beta:.4f}")
            else:
                print(f"   Total Loss: {trainer.last_total_loss:.4f}")
                print(f"   (KL loss disabled)")
        
        # Run evaluation and log eval losses
        if trainer.eval_dataset is not None:
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
                    'eval/total_loss': eval_loss
                }
                
                # Only log KL loss if use_kl is enabled
                if trainer.use_kl:
                    eval_metrics['eval/kl_loss'] = eval_kl_loss
                
                trainer.log(eval_metrics)
                
            print(f"📊 Evaluation Loss Components (Epoch {state.epoch}):")
            print(f"   CE Loss: {eval_ce_loss:.4f}")
            if trainer.use_kl:
                print(f"   KL Loss: {eval_kl_loss:.4f}")
                print(f"   Total Loss: {eval_loss:.4f}")
            else:
                print(f"   Total Loss: {eval_loss:.4f}")
                print(f"   (KL loss disabled)")
            
            # Also log accuracy if available
            if 'eval_accuracy' in eval_results:
                print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")


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
        
        # Run uncertainty evaluation at the beginning of each epoch
        if trainer.eval_dataset is not None:
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
            print(f"[INFO] No evaluation dataset available for uncertainty evaluation at epoch {state.epoch}")


class PredictionTrackerCallback(TrainerCallback):
    """Callback to track predictions on fixed examples across epochs for interpretability."""
    
    def __init__(self, train_dataset, eval_dataset, output_dir, predictions_dir, tokenizer, 
                 n_examples=10, dataset_name="arc_easy", target_ids=None, labels=None):
        super().__init__()
        # Use the provided predictions directory directly
        self.prediction_tracker = PredictionTracker(
            output_dir=predictions_dir,  # Use standardized predictions directory
            tokenizer=tokenizer,
            n_examples=n_examples,
            dataset_name=dataset_name,
            target_ids=target_ids,
            labels=labels
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.examples_selected = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Select fixed examples at the start of training."""
        print(f"\n📝 [PredictionTracker] on_train_begin called - examples_selected: {self.examples_selected}")
        print(f"📝 [PredictionTracker] train_dataset is not None: {self.train_dataset is not None}")
        
        if not self.examples_selected and self.train_dataset is not None:
            print(f"\n📝 [PredictionTracker] Selecting {self.prediction_tracker.n_examples} examples for tracking...")
            
            # Select examples from training and validation sets
            if self.eval_dataset is not None:
                print(f"📝 [PredictionTracker] Using both train and eval datasets for example selection")
                self.prediction_tracker.select_examples(self.train_dataset, self.eval_dataset)
            else:
                # If no eval dataset, just use training set
                print(f"📝 [PredictionTracker] Using only train dataset for example selection")
                self.prediction_tracker.select_examples(self.train_dataset, None)
            
            self.examples_selected = True
            print(f"📝 [PredictionTracker] Examples selected and will be tracked every epoch")
        else:
            if self.examples_selected:
                print(f"📝 [PredictionTracker] Examples already selected, skipping selection")
            if self.train_dataset is None:
                print(f"📝 [PredictionTracker] WARNING: train_dataset is None, cannot select examples")
    
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


def build_classification_trainer(
    model,
    args,
    train_dataset,
    eval_dataset,
    data_collator,
    tokenizer,
    config,  # NEW: Configuration dict
    ard_heldout_loader=None,
    train_dataset_for_weights=None,  # NEW: Optional separate dataset for weight computation
    target_ids=None,
    num_classes=None,
    enable_uncertainty_eval=False,
    enable_prediction_tracker=False,
    prediction_tracker_params=None,
    **kwargs
):
    """
    Convenience function to build ARDClassificationTrainer with proper configuration.
    
    Args:
        model: The model to train
        args: TrainingArguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator function
        tokenizer: Tokenizer
        config: Configuration dictionary (from YAML)
        ard_heldout_loader: DataLoader for ARD prior estimation
        target_ids: Tensor of target token IDs for answer classes
        num_classes: Number of classes (K)
        enable_uncertainty_eval: Whether to enable UncertaintyEvaluationCallback
        enable_prediction_tracker: Whether to enable PredictionTrackerCallback
        prediction_tracker_params: Dict with params for PredictionTrackerCallback
            - predictions_dir: Directory to save predictions
            - n_examples: Number of examples to track (default: 10)
            - dataset_name: Name of dataset (default: "arc_easy")
        **kwargs: Additional trainer arguments
    
    Returns:
        ARDClassificationTrainer instance
    """
    # Get num_classes from config if not provided
    if num_classes is None:
        num_classes = config.get('num_classes')
    
    # Use provided dataset for weights, or fall back to train_dataset
    weight_dataset = train_dataset_for_weights if train_dataset_for_weights is not None else train_dataset
    
    trainer = ARDClassificationTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        config=config,  # Pass config
        ard_heldout_loader=ard_heldout_loader,
        target_ids=target_ids,
        num_classes=num_classes,
        train_dataset_for_weights=weight_dataset,  # Pass dataset for class weight computation
        **kwargs
    )
    
    # Add evaluation callback to track metrics after each epoch
    trainer.add_callback(EvalLossComponentsCallback())
    print("[CLASSIFICATION] Added EvalLossComponentsCallback for epoch-end metric tracking")
    
    # Add uncertainty evaluation callback if requested
    if enable_uncertainty_eval:
        trainer.add_callback(UncertaintyEvaluationCallback())
        print("[CLASSIFICATION] Added UncertaintyEvaluationCallback for uncertainty metrics")
    
    # Add prediction tracker callback if requested
    if enable_prediction_tracker:
        if prediction_tracker_params is None:
            prediction_tracker_params = {}
        
        predictions_dir = prediction_tracker_params.get('predictions_dir', 
                                                        Path(args.output_dir) / 'predictions')
        n_examples = prediction_tracker_params.get('prediction_n_examples')
        dataset_name = prediction_tracker_params.get('dataset_name')
        
        # Extract target_ids and labels from prediction_tracker_params
        tracker_target_ids = prediction_tracker_params.get('target_ids')
        tracker_labels = prediction_tracker_params.get('labels')
        
        trainer.add_callback(PredictionTrackerCallback(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir,
            predictions_dir=predictions_dir,
            tokenizer=tokenizer,
            n_examples=n_examples,
            dataset_name=dataset_name,
            target_ids=tracker_target_ids,
            labels=tracker_labels
        ))
        print(f"[CLASSIFICATION] Added PredictionTrackerCallback (tracking {n_examples} examples)")
    
    return trainer