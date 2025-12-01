
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
from torch.utils.data import Subset, DataLoader
from typing import Dict, Any, Optional
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback
from evaluate.uncertainty_metrics import UncertaintyEvaluator
from evaluate.prediction_tracker import PredictionTracker
from model.model_llama import ProbLoRALayer
from utils.plot import plot_mean_encodings
import numpy as np
from pathlib import Path
import traceback
import gc
import os

class ResamplingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Flag the callback can flip
        self._force_dataloader_rebuild = False
        # Track identity of the dataset used to build the current train dataloader
        self._last_train_dataset_id = None
        # Optional cache to avoid repeated rebuilds within an epoch
        self._cached_train_dataloader = None

    def request_train_dataloader_rebuild(self):
        self._force_dataloader_rebuild = True

    def get_train_dataloader(self) -> DataLoader:
        # Rebuild if:
        #  - a callback asked for it, or
        #  - the dataset object identity changed (new split)
        dataset_id = id(self.train_dataset)
        need_rebuild = (
            self._force_dataloader_rebuild
            or (self._last_train_dataset_id is None)
            or (dataset_id != self._last_train_dataset_id)
        )

        if need_rebuild:
            dl = super().get_train_dataloader()  # this sets up sampler, collator, etc.
            self._cached_train_dataloader = dl
            self._last_train_dataset_id = dataset_id
            self._force_dataloader_rebuild = False
            return dl

        # Return the cached one for the rest of the epoch
        if self._cached_train_dataloader is not None:
            return self._cached_train_dataloader

        # Fallback (shouldn’t happen often)
        dl = super().get_train_dataloader()
        self._cached_train_dataloader = dl
        self._last_train_dataset_id = dataset_id
        return dl


class ARDClassificationTrainer(ResamplingTrainer):
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
        verbose=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Instance variable to control KL loss computation
        # KL loss should be enabled for probabilistic mode, disabled for deterministic mode
        # Determine from config: if deterministic_lora is False, use KL loss
        self.use_kl = not config.get('deterministic_lora') if config is not None else False
        self.verbose = verbose
        self._deterministic = config.get('deterministic_lora') if config is not None else False

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
        
        # Store layer configuration
        self.target_attention_layers = target_attention_layers
        
        # KL Warmup Configuration
        self.kl_loss_beta_max = self.beta  # Store target β_max
        self.enable_kl_warmup = config.get('enable_kl_warmup', False) if config is not None else False
        self.kl_warmup_epochs = config.get('kl_warmup_epochs', 2) if config is not None else 2
        self.kl_warmup_steps = config.get('kl_warmup_steps', None) if config is not None else None
        
        # Total warmup steps will be calculated after first dataloader is created
        self.total_warmup_steps = None
        self._warmup_steps_calculated = False
        
        # Set trainer reference on model for callbacks
        self.model.trainer = self
        
        # Track loss components for logging
        self.last_ce_loss = 0.0
        self.last_kl_loss = 0.0
        self.last_total_loss = 0.0
        
        # Running averages for epoch-level metrics
        self.epoch_ce_loss_sum = 0.0
        self.epoch_kl_loss_sum = 0.0
        self.epoch_total_loss_sum = 0.0
        self.epoch_step_count = 0
        
        # Debug logging directory (will be set by build_classification_trainer)
        self.debug_log_dir = None
        
        print(f"[CLASSIFICATION] ARDClassificationTrainer initialized:")
        print(f"[CLASSIFICATION]   Num classes: {self.num_classes}")
        print(f"[CLASSIFICATION]   Target IDs: {self.target_ids.tolist()}")
        print(f"[CLASSIFICATION]   KL beta (max): {self.beta}")
        print(f"[CLASSIFICATION]   Use KL loss: {self.use_kl}")
        print(f"[CLASSIFICATION]   KL warmup enabled: {self.enable_kl_warmup}")
        if self.enable_kl_warmup:
            if self.kl_warmup_steps is not None:
                print(f"[CLASSIFICATION]   KL warmup steps: {self.kl_warmup_steps}")
            else:
                print(f"[CLASSIFICATION]   KL warmup epochs: {self.kl_warmup_epochs}")
        print(f"[CLASSIFICATION]   Data collator type: {type(self.data_collator).__name__ if self.data_collator else 'None'}")
        print(f"[CLASSIFICATION]   Data collator: {self.data_collator}")
    
    def _calculate_total_warmup_steps(self):
        """Calculate total warmup steps based on dataset size and training config.
        
        This is called lazily on first compute_loss call, after dataloader is created.
        """
        if self._warmup_steps_calculated:
            return
        
        if not self.enable_kl_warmup:
            self._warmup_steps_calculated = True
            return
        
        if self.kl_warmup_steps is not None:
            # Use explicit step count
            self.total_warmup_steps = self.kl_warmup_steps
        else:
            # Calculate from epochs
            try:
                steps_per_epoch = len(self.get_train_dataloader())
                self.total_warmup_steps = steps_per_epoch * self.kl_warmup_epochs
            except Exception as e:
                print(f"[WARNING] Could not calculate warmup steps from epochs: {e}")
                print(f"[WARNING] Falling back to 1000 warmup steps")
                self.total_warmup_steps = 1000
        
        self._warmup_steps_calculated = True
        print(f"[KL WARMUP] Total warmup steps calculated: {self.total_warmup_steps}")
        print(f"[KL WARMUP] Beta schedule: 0.0 -> {self.kl_loss_beta_max:.4f} over {self.total_warmup_steps} steps")
    
    def get_current_kl_beta(self, current_step):
        """
        Compute current KL loss weight using linear warmup schedule.
        β(t) = min(1, t/T_warmup) * β_max
        
        Args:
            current_step: Current training step (0-indexed)
        
        Returns:
            Current β value
        """
        # Lazy calculation of total warmup steps
        if not self._warmup_steps_calculated:
            self._calculate_total_warmup_steps()
        
        if not self.enable_kl_warmup:
            return self.kl_loss_beta_max
        
        if current_step >= self.total_warmup_steps:
            return self.kl_loss_beta_max
        
        # Linear warmup: β(t) = (t / T_warmup) * β_max
        warmup_progress = current_step / self.total_warmup_steps
        current_beta = warmup_progress * self.kl_loss_beta_max
        
        return current_beta

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
        # Get current KL beta based on warmup schedule
        current_step = self.state.global_step if hasattr(self, 'state') else 0
        current_beta = self.get_current_kl_beta(current_step)
        
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
        
        # Step 3: Compute cross-entropy loss over K classes
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
        
        # kl = 0.0
        
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
                    print(f"[GRADIENT DEBUG]   Hidden states[0] (embeddings) requires_grad: {hidden_states[0].requires_grad}")
                    print(f"[GRADIENT DEBUG]   Hidden states[0] (embeddings) has grad_fn: {hidden_states[0].grad_fn is not None}")
                    
                    # Check middle and last hidden states (these SHOULD have gradients if LoRA is working)
                    if len(hidden_states) > 1:
                        mid_idx = len(hidden_states) // 2
                        print(f"[GRADIENT DEBUG]   Hidden states[{mid_idx}] (mid-layer) requires_grad: {hidden_states[mid_idx].requires_grad}")
                        print(f"[GRADIENT DEBUG]   Hidden states[{mid_idx}] (mid-layer) has grad_fn: {hidden_states[mid_idx].grad_fn is not None}")
                        print(f"[GRADIENT DEBUG]   Hidden states[-1] (final) requires_grad: {hidden_states[-1].requires_grad}")
                        print(f"[GRADIENT DEBUG]   Hidden states[-1] (final) has grad_fn: {hidden_states[-1].grad_fn is not None}")
                        
                    # Most important: Check if logits have grad_fn (proves gradient path from loss to model)
                    print(f"[GRADIENT DEBUG]   Logits requires_grad: {logits.requires_grad}")
                    print(f"[GRADIENT DEBUG]   Logits has grad_fn: {logits.grad_fn is not None}")
                    print(f"[GRADIENT DEBUG]   CE Loss requires_grad: {ce_loss.requires_grad}")
                    print(f"[GRADIENT DEBUG]   CE Loss has grad_fn: {ce_loss.grad_fn is not None}")
            

            # ---- NEW: collect KL terms instead of += on a Python float
            kl_terms = []

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
                                            # Detach layer_input to prevent KL gradients from flowing to B matrices
                                            # KL loss should only affect A (distribution params), not B (output projection)
                                            proj_kl = proj.kl_divergence_latent(layer_input.detach())

                                            # ---- NEW: append the *Tensor* (no .item(), no .detach())
                                            kl_terms.append(proj_kl)

                                            # kl += proj_kl

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
            
            # ---- NEW: build the final KL tensor once
            if kl_terms:
                kl = torch.stack(kl_terms).sum()             # tensor with grad_fn
            else:
                kl = ce_loss.new_zeros(())                   # scalar 0.0 tensor (no grad_fn)

            # # If no KL components found, create zero tensor with gradient connection
            # # Safe check to avoid "Boolean value of Tensor is ambiguous" error
            # if not torch.is_tensor(kl) or (torch.is_tensor(kl) and float(kl.detach().item()) == 0.0):
            #     kl = torch.tensor(0.0, device=ce_loss.device, requires_grad=True)
        
        # 🔎 DEBUG: print batch KL once it's finalized, before combining losses
        if self.use_kl and isinstance(kl, torch.Tensor) and kl.requires_grad:
            # (optional) only print on main rank to avoid DDP spam
            # Only log every 100 steps to reduce verbosity
            step_for_log = self.state.global_step if hasattr(self, 'state') else 0
            if getattr(self.args, "local_rank", -1) in (-1, 0) and step_for_log % 100 == 0:
                warmup_status = f" (warmup: {current_beta:.6f} / {self.kl_loss_beta_max:.4f})" if self.enable_kl_warmup and current_beta < self.kl_loss_beta_max else ""
                print(f"[KL] batch KL={kl.detach().item():.6f} (beta={current_beta:.6f}){warmup_status}")

        # Prove CE vs KL gradients separately (probes to check gradient flow)
        # Run at start of each epoch after epoch 0 to track gradient evolution
        current_epoch = int(self.state.epoch) if hasattr(self, 'state') else 0
        current_step_in_epoch = self.state.global_step - (self.state.global_step // len(self.get_train_dataloader())) * len(self.get_train_dataloader()) if hasattr(self, 'state') else 0
        
        # Run GRAD-PROBE once per epoch: at first step of each epoch after epoch 0
        if current_epoch >= 0 and current_step_in_epoch == 0:

            # Collect ALL adapter params, then take FIRST 12 and LAST 12 to validate gradient flow theory
            # (Last layer B should have zero KL gradient, earlier layers should have non-zero)
            all_probe_params = []
            for mod_name, mod in model.named_modules():
                if not isinstance(mod, ProbLoRALayer):
                    continue
                
                # Only parameters directly on this module (no recursion)
                for p_name, p in mod.named_parameters(recurse=False):
                    if not p.requires_grad:
                        continue
                    
                    # ProbLoRA/LoRA adapters to monitor (mode-specific)
                    full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                    if self.use_kl:
                        # Probabilistic mode: monitor A (contains mu and logvar) and B
                        if p_name in ["A", "B"]:
                            all_probe_params.append((full_name, p))
                    else:
                        # Deterministic mode: monitor mu_A and B
                        if p_name in ["mu_A", "B"]:
                            all_probe_params.append((full_name, p))
            
            # Take FIRST 12 and LAST 12 parameters (lowest and highest layers)
            if len(all_probe_params) >= 24:
                probe_params = all_probe_params[:12] + all_probe_params[-12:]
            else:
                probe_params = all_probe_params
            
            mode_str = "probabilistic" if self.use_kl else "deterministic"
            print(f"[GRAD-PROBE] monitoring {len(probe_params)} adapter params ({mode_str} mode)")
            print(model.model.layers[0].self_attn.q_proj)
            print(type(model.model.layers[0].self_attn.q_proj))

            # # CE-only
            # model.zero_grad(set_to_none=True)
            # ce_loss.backward(retain_graph=True)
            # ce_g = [(None if p.grad is None else p.grad.detach().norm().item()) for p in probe_params]
            # print(f"[GRAD-PROBE] CE-only grad norms: {ce_g}")

            # # KL-only (only if KL is enabled and has grad path)
            # if self.use_kl and isinstance(kl, torch.Tensor) and kl.requires_grad:
            #     model.zero_grad(set_to_none=True)
            #     (self.beta * kl).backward(retain_graph=True)
            #     kl_g = [(None if p.grad is None else p.grad.detach().norm().item()) for p in probe_params]
            #     print(f"[GRAD-PROBE] KL-only grad norms (beta={self.beta}): {kl_g}")

            print(f"[GRAD-PROBE] monitoring {len(probe_params)} adapter params:")
            for i, (name, p) in enumerate(probe_params):
                print(f"  [{i}] {name}  shape={tuple(p.shape)}")

            # ---------- CE-only gradients ----------
            model.zero_grad(set_to_none=True)
            ce_loss.backward(retain_graph=True)

            ce_grad_norms = []
            for name, p in probe_params:
                if p.grad is None:
                    ce_grad_norms.append(None)
                else:
                    ce_grad_norms.append(p.grad.detach().norm().item())

            print("[GRAD-PROBE] CE-only grad norms (by param):")
            for i, ((name, _), g) in enumerate(zip(probe_params, ce_grad_norms)):
                print(f"  [{i}] {name}: CE_grad_norm = {g}")

            # ---------- KL-only gradients ----------
            kl_grad_norms = None
            if getattr(self, "use_kl", False) and isinstance(kl, torch.Tensor) and kl.requires_grad:
                model.zero_grad(set_to_none=True)
                (self.beta * kl).backward(retain_graph=True)

                kl_grad_norms = []
                for name, p in probe_params:
                    if p.grad is None:
                        kl_grad_norms.append(None)
                    else:
                        kl_grad_norms.append(p.grad.detach().norm().item())

                print(f"[GRAD-PROBE] KL-only grad norms (beta={self.beta}):")
                for i, ((name, _), g) in enumerate(zip(probe_params, kl_grad_norms)):
                    print(f"  [{i}] {name}: KL_grad_norm = {g}")

            # ---------- Total loss gradients ----------
            model.zero_grad(set_to_none=True)
            if self.use_kl:
                total_loss_probe = ce_loss + self.beta * kl
            else:
                total_loss_probe = ce_loss
            total_loss_probe.backward(retain_graph=True)

            total_grad_norms = []
            for name, p in probe_params:
                if p.grad is None:
                    total_grad_norms.append(None)
                else:
                    total_grad_norms.append(p.grad.detach().norm().item())

            print(f"[GRAD-PROBE] Total loss grad norms (CE + beta*KL):")
            for i, ((name, _), g) in enumerate(zip(probe_params, total_grad_norms)):
                print(f"  [{i}] {name}: Total_grad_norm = {g}")

            model.zero_grad(set_to_none=True)  # clean for Trainer's backward


        # Combine losses (use current_beta for warmup)
        if self.use_kl:
            total_loss = ce_loss + current_beta * kl
        else:
            total_loss = ce_loss  # Keep tensor, don't add scalar 0.0
        
        # Store for logging
        ce_loss_val = ce_loss.item()
        if self.use_kl:
            kl_loss_val = kl.item() if torch.is_tensor(kl) else float(kl) if self.use_kl else 0.0
        total_loss_val = total_loss.item()
        
        self.last_ce_loss = ce_loss_val
        if self.use_kl:
            self.last_kl_loss = kl_loss_val
        self.last_total_loss = total_loss_val
        
        # Accumulate for epoch average
        self.epoch_ce_loss_sum += ce_loss_val
        if self.use_kl:
            self.epoch_kl_loss_sum += kl_loss_val
        self.epoch_total_loss_sum += total_loss_val
        self.epoch_step_count += 1
        
        # Log current beta periodically (every 50 steps)
        if self.enable_kl_warmup and current_step % 50 == 0:
            if hasattr(self.args, 'report_to') and 'tensorboard' in self.args.report_to:
                self.log({"train/kl_beta_current": current_beta})
        
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
        
        # Monitor log_var values periodically to detect training instability
        if not hasattr(self, '_logvar_monitor_step'):
            self._logvar_monitor_step = 0
        
        self._logvar_monitor_step += 1
        
        # Monitor every 50 steps
        if self._logvar_monitor_step % 50 == 0:
            self._monitor_logvar_stability(model)
        
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
                print("[GRAD]   WARNING: NO gradients computed! Check loss computation.")
            elif nz < tot:
                print(f"[GRAD]   WARNING: Only {nz}/{tot} parameters have gradients!")
            else:
                print(f"[GRAD] ✅ All trainable parameters have gradients.")
            
            # Optional: Show a few gradient samples
            print(f"[GRAD] Sample gradients (first 3 ProbLoRA modules):")
            count = 0
            for mod_name, mod in model.named_modules():
                if not isinstance(mod, ProbLoRALayer):
                    continue
                # Check gradients for mu_A (always trainable in both modes)
                if hasattr(mod, 'mu_A') and mod.mu_A.grad is not None:
                    grad_norm = mod.mu_A.grad.norm().item()
                    print(f"[GRAD]   {mod_name}.mu_A: grad_norm={grad_norm:.6f}")
                    count += 1
                # Only check logvar_A gradients if not deterministic (i.e., if KL loss is used)
                if (not self._deterministic) and hasattr(mod, 'logvar_A') and mod.logvar_A is not None and mod.logvar_A.grad is not None:
                    grad_norm = mod.logvar_A.grad.norm().item()
                    print(f"[GRAD]   {mod_name}.logvar_A: grad_norm={grad_norm:.6f}")
                    count += 1
                if count >= 6:  # Show 3 modules (mu_A + logvar_A each)
                    break
            print("="*60 + "\n")
        
        return loss
    
    def _log_debug(self, message, console=True):
        """Log debug message to debug log file and optionally to console.
        
        Creates a separate log file for each epoch: logvar_monitoring_epoch_N.log
        """
        if console:
            print(message)
        
        if self.debug_log_dir is not None:
            # Get current epoch number
            current_epoch = int(self.state.epoch) if hasattr(self, 'state') else 0
            
            # Create epoch-specific log file
            log_file = os.path.join(self.debug_log_dir, f"logvar_monitoring_epoch_{current_epoch}.log")
            
            with open(log_file, 'a') as f:
                f.write(f"{message}\n")
    
    def _monitor_logvar_stability(self, model):
        """
        Monitor log_var values across all ProbLoRA layers to detect numerical instability.
        Saves warnings to debug log file when extreme values are detected.
        """
        if not self.use_kl:
            return  # Only relevant for probabilistic mode
        
        # INFO: Indicate monitoring is running
        current_step = self.state.global_step if hasattr(self, 'state') else self._logvar_monitor_step
        current_epoch = int(self.state.epoch) if hasattr(self, 'state') else 0
        step_in_epoch = current_step - getattr(self, '_epoch_start_step', current_step)
        
        log_dir_info = f" -> Log: {self.debug_log_dir}" if self.debug_log_dir else " (no log dir set)"
        self._log_debug(f"[INFO] Monitoring log_var stability - Epoch {current_epoch}, Step {step_in_epoch} (Global: {current_step}){log_dir_info}", console=True)
        
        logvar_stats = []
        extreme_layers = []
        nan_inf_layers = []
        
        # Debug: Track what we're finding
        total_layers = 0
        prob_lora_layers = 0
        det_lora_layers = 0
        
        # Collect log_var statistics from all ProbLoRA layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            total_layers = len(model.model.layers)
            for layer_idx, layer in enumerate(model.model.layers):
                if hasattr(layer, 'self_attn') and self.target_attention_layers:
                    attn = layer.self_attn
                    for proj_name in self.target_attention_layers:
                        if hasattr(attn, proj_name):
                            proj = getattr(attn, proj_name)
                            # Check if this is a ProbLoRA layer
                            if hasattr(proj, 'A'):  # Probabilistic mode: A contains both mu and logvar
                                prob_lora_layers += 1
                                # Extract logvar from A (second half of the tensor)
                                _, logvar = torch.split(proj.A, proj.rank, dim=0)
                                logvar = logvar.detach()
                                
                                # Convert log_var to var using exponential
                                var = torch.exp(logvar)
                                
                                # Compute statistics on variance (not log variance)
                                min_val = var.min().item()
                                max_val = var.max().item()
                                mean_val = var.mean().item()
                                std_val = var.std().item()
                                
                                # Also track log_var statistics for debugging
                                logvar_min = logvar.min().item()
                                logvar_max = logvar.max().item()
                                logvar_mean = logvar.mean().item()
                                logvar_std = logvar.std().item()                                

                                logvar_stats.append({
                                    'layer': layer_idx,
                                    'proj': proj_name,
                                    'var_min': min_val,
                                    'var_max': max_val,
                                    'var_mean': mean_val,
                                    'var_std': std_val,
                                    'logvar_min': logvar_min,
                                    'logvar_max': logvar_max,
                                    'logvar_mean': logvar_mean,
                                    'logvar_std': logvar_std,
                                })
                                
                                # Check for NaN or Inf first (most critical)
                                if torch.isnan(logvar).any() or torch.isinf(logvar).any():
                                    nan_inf_layers.append(f"L{layer_idx}.{proj_name}")
                                
                                # Check for extreme values that could cause instability
                                # log_var > 10 means var > exp(10) ≈ 22000 (too large)
                                # log_var < -10 means var < exp(-10) ≈ 0.000045 (too small)
                                elif logvar_max > 10 or logvar_min < -10:
                                    extreme_layers.append(f"L{layer_idx}.{proj_name}")
                            elif hasattr(proj, 'mu_A'):  # Deterministic mode: only has mu_A
                                det_lora_layers += 1
                                # Skip monitoring for deterministic layers (no variance to monitor)
                                pass
        
        # Log summary of what was found (only on first run)
        if not hasattr(self, '_logvar_monitor_initialized'):
            self._logvar_monitor_initialized = True
            self._log_debug(f"[INFO] LogVar Monitor initialized:")
            self._log_debug(f"[INFO]   Total transformer layers: {total_layers}")
            self._log_debug(f"[INFO]   ProbLoRA layers found: {prob_lora_layers}")
            self._log_debug(f"[INFO]   Deterministic LoRA layers found: {det_lora_layers}")
            self._log_debug(f"[INFO]   Target projections: {self.target_attention_layers}")

        
        # Print and log summary if we have statistics
        if logvar_stats:
            current_step = self.state.global_step if hasattr(self, 'state') else self._logvar_monitor_step
            current_epoch = int(self.state.epoch) if hasattr(self, 'state') else 0
            
            # Track epoch changes and add separator
            if not hasattr(self, '_last_logged_epoch'):
                self._last_logged_epoch = -1
            
            if current_epoch > self._last_logged_epoch:
                self._last_logged_epoch = current_epoch
                self._epoch_start_step = current_step
                # Add epoch separator in log file
                separator = "\n" + "="*80
                self._log_debug(separator, console=False)
                self._log_debug(f"EPOCH {current_epoch} - Log Var Monitoring", console=False)
                self._log_debug("="*80, console=False)
            
            # Calculate step within current epoch
            step_in_epoch = current_step - getattr(self, '_epoch_start_step', current_step)
            
            # Compute global statistics (on variance, not log_var)
            all_var_mins = [s['var_min'] for s in logvar_stats]
            all_var_maxs = [s['var_max'] for s in logvar_stats]
            all_var_means = [s['var_mean'] for s in logvar_stats]
            
            # Also compute log_var global stats for reference
            all_logvar_mins = [s['logvar_min'] for s in logvar_stats]
            all_logvar_maxs = [s['logvar_max'] for s in logvar_stats]
            all_logvar_means = [s['logvar_mean'] for s in logvar_stats]
            all_logvar_stds = [s['logvar_std'] for s in logvar_stats]
            
            self._log_debug(f"\nEpoch {current_epoch}, Step {step_in_epoch} (Global Step {current_step})")
            self._log_debug(f"  Variance (σ²) - Global range: [{min(all_var_mins):.6f}, {max(all_var_maxs):.6f}]")
            self._log_debug(f"  Variance (σ²) - Mean across layers: {sum(all_var_means)/len(all_var_means):.6f}")
            self._log_debug(f"  Log Variance - Global range: [{min(all_logvar_mins):.3f}, {max(all_logvar_maxs):.3f}]")
            self._log_debug(f"  Log Variance - Mean across layers: {sum(all_logvar_means)/len(all_logvar_means):.3f}")
            self._log_debug(f"  Log Variance - Std across layers: {sum(all_logvar_stds)/len(all_logvar_stds):.3f}")
            
            # Critical warning for NaN/Inf
            if nan_inf_layers:
                self._log_debug(f"  CRITICAL: NaN/Inf detected!")
                self._log_debug(f"  Affected layers: {', '.join(nan_inf_layers[:10])}")
                if len(nan_inf_layers) > 10:
                    self._log_debug(f"  ... and {len(nan_inf_layers) - 10} more layers")
                self._log_debug(f"  Training may become unstable - consider adding log_var clamping")
            
            # Warning for extreme values
            elif extreme_layers:
                self._log_debug(f"  WARNING: Extreme log_var values detected!")
                self._log_debug(f"  Affected layers: {', '.join(extreme_layers[:10])}")
                if len(extreme_layers) > 10:
                    self._log_debug(f"  ... and {len(extreme_layers) - 10} more layers")
                self._log_debug(f"  Consider adding log_var clamping: torch.clamp(log_var, min=-10, max=10)")
            
            # Show details for all layers (to file, selectively to console)
            if len(logvar_stats) > 0:
                self._log_debug(f"  Total projections monitored: {len(logvar_stats)}")
                details_msg = ["  Layer-by-layer details (variance and log_var):"]
                for stat in logvar_stats:  # Show ALL layers
                    details_msg.append(
                        f"    L{stat['layer']}.{stat['proj']}: "
                        f"var_mean={stat['var_mean']:.6f}, var_std={stat['var_std']:.6f}, "
                        f"var_range=[{stat['var_min']:.6f}, {stat['var_max']:.6f}]"
                    )
                    details_msg.append(
                        f"      (log_var: mean={stat['logvar_mean']:.3f}, std={stat['logvar_std']:.3f}, "
                        f"range=[{stat['logvar_min']:.3f}, {stat['logvar_max']:.3f}])"
                    )
                
                # Determine if we should print to console
                show_console = self.verbose or extreme_layers or nan_inf_layers
                
                # Log details (to file always, to console conditionally)
                for msg in details_msg:
                    self._log_debug(msg, console=show_console)
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step for classification tasks.
        
        Returns predictions as class probabilities.
        """
        # Extract classes
        classes = inputs.pop("classes") if "classes" in inputs else None
        inputs.pop("labels", None)
        
        has_labels = classes is not None
        
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
        # Save original args
        original_disable_tqdm = self.args.disable_tqdm
        original_logging_steps = self.args.logging_steps
        
        # Disable progress bar and logging
        self.args.disable_tqdm = True
        self.args.logging_steps = 999999
        
        # Remove ProgressCallback temporarily
        from transformers.trainer_callback import ProgressCallback
        progress_callbacks = [cb for cb in self.callback_handler.callbacks if isinstance(cb, ProgressCallback)]
        for cb in progress_callbacks:
            self.callback_handler.remove_callback(cb)
        
        try:
            # Call parent evaluate
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )
            
            return metrics
        finally:
            # Restore original settings
            self.args.disable_tqdm = original_disable_tqdm
            self.args.logging_steps = original_logging_steps
            
            # Re-add ProgressCallback
            for cb in progress_callbacks:
                self.callback_handler.add_callback(cb)
    
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
                
                if batch_idx % 100 == 0:
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
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Reset epoch-level loss accumulators at the start of each epoch."""
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is not None:
            trainer.epoch_ce_loss_sum = 0.0
            trainer.epoch_kl_loss_sum = 0.0
            trainer.epoch_total_loss_sum = 0.0
            trainer.epoch_step_count = 0
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Compute and log evaluation loss components at the end of each epoch."""
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None:
            print("[EvalLossComponentsCallback] No trainer reference found")
            return
        
        print(f"\n📊 [EVAL] Logging metrics for epoch {state.epoch}")
        
        # Compute epoch averages
        if trainer.epoch_step_count > 0:
            avg_ce_loss = trainer.epoch_ce_loss_sum / trainer.epoch_step_count
            avg_kl_loss = trainer.epoch_kl_loss_sum / trainer.epoch_step_count
            avg_total_loss = trainer.epoch_total_loss_sum / trainer.epoch_step_count
        else:
            avg_ce_loss = trainer.last_ce_loss
            avg_kl_loss = trainer.last_kl_loss
            avg_total_loss = trainer.last_total_loss
        
        # Log training loss components to TensorBoard
        if trainer.args.report_to and 'tensorboard' in trainer.args.report_to:
            training_metrics = {
                'train/ce_loss_epoch': avg_ce_loss,
                'train/total_loss_epoch': avg_total_loss,
            }
            
            # Only log KL loss if use_kl is enabled
            if trainer.use_kl:
                training_metrics['train/kl_loss_epoch'] = avg_kl_loss
                # Log the current (or max) beta depending on warmup status
                current_beta = trainer.get_current_kl_beta(state.global_step)
                training_metrics['train/kl_beta'] = current_beta
                if trainer.enable_kl_warmup:
                    training_metrics['train/kl_beta_max'] = trainer.kl_loss_beta_max
            
            trainer.log(training_metrics)
            print(f"📊 Training Loss Components - Epoch {state.epoch} Average (over {trainer.epoch_step_count} steps):")
            print(f"   CE Loss: {avg_ce_loss:.4f}")
            if trainer.use_kl:
                print(f"   KL Loss: {avg_kl_loss:.4f}")
                print(f"   Total Loss: {avg_total_loss:.4f}")
                current_beta = trainer.get_current_kl_beta(state.global_step)
                if trainer.enable_kl_warmup:
                    warmup_pct = min(100, 100 * current_beta / trainer.kl_loss_beta_max)
                    print(f"   [INFO] KL Beta (current): {current_beta:.6f} ({warmup_pct:.1f}% of max={trainer.kl_loss_beta_max:.4f})")
                else:
                    print(f"   [INFO] KL Beta: {current_beta:.4f}")
            else:
                print(f"   Total Loss: {avg_total_loss:.4f}")
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
    
    def __init__(self, train_dataset, eval_dataset, predictions_dir, tokenizer, 
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


class HeldoutResampleCallback(TrainerCallback):
    """Callback to resample ARD samples from training data each epoch for dynamic prior estimation.
    
    CRITICAL: Creates MUTUALLY EXCLUSIVE splits:
    - SGD samples: Used for gradient updates
    - ARD samples: Used ONLY for ARD prior estimation (NOT in SGD)
    """
    
    def __init__(self, train_ds, val_ds, ard_prior_samples, batch_size, tokenizer=None, data_collator=None):
        super().__init__()
        self.train_ds = train_ds  # FULL training data
        self.val_ds = val_ds
        self.ard_prior_samples = ard_prior_samples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        
        # Validate split feasibility
        if self.train_ds is not None:
            total_train = len(self.train_ds)
            min_sgd_samples = total_train - self.ard_prior_samples
            
            if min_sgd_samples < self.ard_prior_samples:
                print(f"[HeldoutResampleCallback] WARNING: Training dataset has only {total_train} samples.")
                print(f"[HeldoutResampleCallback] Adjusting ARD samples to {total_train // 2} to ensure equal SGD/ARD split")
                self.ard_prior_samples = total_train // 2
            
            if min_sgd_samples < 100:
                raise ValueError(f"Insufficient SGD samples ({min_sgd_samples}). Need more training data!")
    
    def _create_mutually_exclusive_splits(self):
        """Create mutually exclusive SGD and ARD splits from training data.
        
        Returns:
            sgd_dataset: Dataset for SGD training (no overlap with ARD)
            ard_dataset: Dataset for ARD estimation (no overlap with SGD)
        """
        if self.train_ds is None or len(self.train_ds) == 0:
            print("[HeldoutResampleCallback] No training data available")
            return None, None
        
        total_train = len(self.train_ds)
        ard_samples = min(self.ard_prior_samples, total_train // 2)  # At most 50%
        sgd_samples = total_train - ard_samples
        
        # Randomly shuffle ALL training indices
        perm = np.random.permutation(total_train)
        
        # MUTUALLY EXCLUSIVE SPLIT
        ard_indices = [int(idx) for idx in perm[:ard_samples]]          # First N for ARD
        sgd_indices = [int(idx) for idx in perm[ard_samples:]]          # Remaining for SGD
        
        # Verify no overlap (sanity check)
        assert set(ard_indices).isdisjoint(set(sgd_indices)), "ARD and SGD indices overlap!"
        
        # Create datasets
        if hasattr(self.train_ds, 'select'):
            # HuggingFace dataset
            sgd_dataset = self.train_ds.select(sgd_indices)
            ard_dataset = self.train_ds.select(ard_indices)
        else:
            # PyTorch dataset
            sgd_dataset = Subset(self.train_ds, sgd_indices)
            ard_dataset = Subset(self.train_ds, ard_indices)
        
        return sgd_dataset, ard_dataset
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Resample MUTUALLY EXCLUSIVE SGD and ARD splits at the beginning of each epoch."""
        print(f"\n[HeldoutResampleCallback] on_epoch_begin called for epoch {int(state.epoch)}")
        
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None:
            print("[HeldoutResampleCallback] No trainer reference found")
            return
        
        if self.train_ds is None:
            print("[HeldoutResampleCallback] No training dataset available")
            return
        
        print(f"[HeldoutResampleCallback] train_ds size: {len(self.train_ds)}")
        print(f"[HeldoutResampleCallback] data_collator is None: {self.data_collator is None}")
        
        try:
            # Create mutually exclusive splits
            print(f"[HeldoutResampleCallback] Creating mutually exclusive splits...")
            sgd_dataset, ard_dataset = self._create_mutually_exclusive_splits()
            
            print(f"[HeldoutResampleCallback] Splits created - sgd_dataset: {sgd_dataset is not None}, ard_dataset: {ard_dataset is not None}")
            
            if sgd_dataset is not None and ard_dataset is not None:
                # ✅ UPDATE TRAINER'S TRAINING DATASET (for SGD)
                trainer.train_dataset = sgd_dataset
                
                # Ask the trainer to rebuild
                if hasattr(trainer, "request_train_dataloader_rebuild"):
                    trainer.request_train_dataloader_rebuild()
                else:
                    trainer._force_dataloader_rebuild = True  # your existing flag

                # For newer transformers versions, also hint the core loop:
                if "control" in locals() and hasattr(control, "should_recompute_train_dataloader"):
                    control.should_recompute_train_dataloader = True

                # ✅ CREATE ARD DATALOADER (for ARD estimation only)
                # Use provided data_collator or fallback to trainer's data_collator
                collator = self.data_collator if self.data_collator is not None else trainer.data_collator
                
                if collator is not None:
                    trainer.ard_heldout_loader = DataLoader(
                        ard_dataset,
                        batch_size=self.batch_size,
                        collate_fn=collator,
                        shuffle=False
                    )

                    # 🔎 SANITY PRINT — add here
                    train_dl = trainer.get_train_dataloader()  # rebuilds now because of the flag
                    try:
                        num_batches = len(train_dl)
                    except TypeError:
                        num_batches = "unknown (iterable dataset)"
                    bs = getattr(train_dl, "batch_size")
                    num_samples = len(trainer.train_dataset)  # size of *SGD split*
                    print(f"[HeldoutResample] epoch {int(state.epoch)} "
                        f"SGD batches={num_batches}, batch_size={bs}, SGD samples≈{num_samples}")

                    print(f"[HeldoutResampleCallback] ✅ Epoch {int(state.epoch)} - Mutually Exclusive Split:")
                    print(f"   SGD samples: {len(sgd_dataset)} (for gradient updates)")
                    print(f"   ARD samples: {len(ard_dataset)} (for prior estimation ONLY)")
                    print(f"   Total: {len(sgd_dataset) + len(ard_dataset)} from {len(self.train_ds)} training samples")
                    print(f"   Overlap: 0 samples ✅")
                    print(f"   Eval (fixed): {len(self.val_ds) if self.val_ds else 0} samples")
                    print(f"   Data collator: {type(collator).__name__}")
                else:
                    print(f"[HeldoutResampleCallback] ⚠️ No data_collator available (callback: {self.data_collator is None}, trainer: {trainer.data_collator is None})")

        except Exception as e:
            print(f"[HeldoutResampleCallback] Failed to create splits: {e}")
            traceback.print_exc()


class PriorEstimationCallback(TrainerCallback):
    """Callback to estimate ARD priors following DeBERTa pattern.
    
    Uses on_step_begin to run AFTER HeldoutResampleCallback.on_epoch_begin
    completes, ensuring ard_heldout_loader is ready.
    """
    
    def __init__(self, device):
        super().__init__()
        self.device = device
        self._last_estimated_epoch = -1  # Track which epoch we last estimated for
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Estimate ARD priors at the first step of each epoch (after resampling completes)."""
        model = kwargs["model"]
        trainer = getattr(model, 'trainer', None)
        
        if trainer is None:
            return
        
        # Only run once per epoch (at the first step)
        current_epoch = int(state.epoch)
        if current_epoch == self._last_estimated_epoch:
            return  # Already estimated for this epoch
        
        self._last_estimated_epoch = current_epoch
        
        # Use ard_heldout_loader for ARD prior estimation
        if not hasattr(trainer, 'ard_heldout_loader'):
            print("[PriorEstimationCallback] trainer.ard_heldout_loader not found, skipping ARD estimation")
            return
        
        eval_data = trainer.ard_heldout_loader
        
        # Skip ARD estimation if loader hasn't been created yet by HeldoutResampleCallback
        if eval_data is None:
            print(f"[PriorEstimationCallback] Skipping ARD estimation at epoch {current_epoch} - DataLoader not yet created")
            return
        
        print(f"[PriorEstimationCallback] Estimating ARD priors at beginning of epoch {current_epoch}...")
        
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
            traceback.print_exc()
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_classification_trainer(
    model,
    args,
    train_dataset,
    eval_dataset,
    data_collator,
    tokenizer,
    config,  # NEW: Configuration dict
    ard_heldout_loader=None,
    target_ids=None,
    num_classes=None,
    enable_uncertainty_eval=False,
    enable_prediction_tracker=False,
    prediction_tracker_params=None,
    enable_plotting=False,
    plot_params=None,
    debug_log_dir=None,
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
        enable_plotting: Whether to enable LatentPlotCallback
        plot_params: Dict with params for LatentPlotCallback
            - start_epoch: Epoch to start plotting from (default: 0)
            - interval: Interval between plots (default: 1)
            - plot_batch_size: Batch size for plotting (default: 8)
            - latent_plot_dir: Directory to save latent plots
            Note: device is automatically extracted from args.device
        **kwargs: Additional trainer arguments
    
    Returns:
        ARDClassificationTrainer instance
    """
    # Get num_classes from config if not provided
    if num_classes is None:
        num_classes = config.get('num_classes')
    
    # Debug: Show data_collator details before creating trainer
    print(f"\n[BUILD_TRAINER] Creating ARDClassificationTrainer:")
    print(f"[BUILD_TRAINER]   data_collator provided: {data_collator is not None}")
    print(f"[BUILD_TRAINER]   data_collator type: {type(data_collator).__name__ if data_collator else 'None'}")
    print(f"[BUILD_TRAINER]   data_collator object: {data_collator}")
    
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
        **kwargs
    )
    
    # Set debug_log_dir for log_var monitoring
    if debug_log_dir is not None:
        trainer.debug_log_dir = debug_log_dir
        print(f"[CLASSIFICATION] Debug logging enabled: {debug_log_dir}")
    
    # AFTER trainer creation, extract the actual data_collator used by HF Trainer
    print(f"\n[BUILD_TRAINER] After trainer initialization:")
    print(f"[BUILD_TRAINER]   Trainer's data_collator type: {type(trainer.data_collator).__name__ if trainer.data_collator else 'None'}")
    print(f"[BUILD_TRAINER]   Trainer's data_collator object: {trainer.data_collator}")
    
    # Add evaluation callback to track metrics after each epoch
    trainer.add_callback(EvalLossComponentsCallback())
    print("[CLASSIFICATION] Added EvalLossComponentsCallback for epoch-end metric tracking")

    # Add HeldoutResampleCallback for dynamic ARD/SGD splits
    # FIXED: Use dict membership check instead of hasattr() for dict objects
    if train_dataset is not None and 'ard_prior_samples' in config and 'batch_size' in config:
        # Verify data_collator consistency
        print(f"[CLASSIFICATION] data_collator type: {type(data_collator).__name__ if data_collator else 'None'}")
        print(f"[CLASSIFICATION] Trainer's data_collator is same instance: {trainer.data_collator is data_collator}")
        
        trainer.add_callback(HeldoutResampleCallback(
                train_ds=train_dataset,
                val_ds=eval_dataset,
                ard_prior_samples=config['ard_prior_samples'],
                batch_size=config['batch_size'],
                tokenizer=tokenizer,
                data_collator=trainer.data_collator  # CRITICAL: Same instance as trainer
            ))
        print("[CLASSIFICATION] Added HeldoutResampleCallback for dynamic ARD/SGD splits")
        print("[CLASSIFICATION] ✅ All dataloaders (train/eval/ARD) use the same data_collator instance")
    else:
        print(f"[CLASSIFICATION] ⚠️ HeldoutResampleCallback NOT added:")
        print(f"   train_dataset is not None: {train_dataset is not None}")
        print(f"   'ard_prior_samples' in config: {'ard_prior_samples' in config if config else 'config is None'}")
        print(f"   'batch_size' in config: {'batch_size' in config if config else 'config is None'}")

    # Add PriorEstimationCallback for ARD prior estimation
    if hasattr(args, 'device'):
        trainer.add_callback(PriorEstimationCallback(device=args.device))
        print("[CLASSIFICATION] Added PriorEstimationCallback for ARD prior estimation")
    else:
        print("[CLASSIFICATION] ⚠️ PriorEstimationCallback NOT added - args.device not found")
    
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
        
        trainer.add_callback(PredictionTrackerCallback(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            predictions_dir=predictions_dir,
            tokenizer=tokenizer,
            n_examples=n_examples,
            dataset_name=dataset_name
        ))
        print(f"[CLASSIFICATION] Added PredictionTrackerCallback (tracking {n_examples} examples)")
    
    # Add plotting callback if enabled and utilities available
    if enable_plotting:
        # Extract plot parameters with defaults
        start_epoch = plot_params.get('start_epoch')
        interval = plot_params.get('interval')
        plot_batch_size = plot_params.get('plot_batch_size')
        latent_plot_dir = plot_params.get('latent_plot_dir')
        
        # Check if plot utilities are available
        if plot_mean_encodings is not None:
            # Use latent_plot_dir if provided, otherwise fallback to args.output_dir
            plot_output_dir = latent_plot_dir
            trainer.add_callback(LatentPlotCallback(
                device=args.device,
                output_dir=plot_output_dir,
                start_epoch=start_epoch,
                interval=interval,
                plot_batch_size=plot_batch_size
            ))
            print(f"[CLASSIFICATION] Added LatentPlotCallback (start_epoch={start_epoch}, interval={interval})")
            print(f"[CLASSIFICATION]   Plot output directory: {plot_output_dir}")
        else:
            print("[CLASSIFICATION] ⚠️ LatentPlotCallback NOT added - plot_mean_encodings utility not available")

    return trainer

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
        was_training = model.training  # <--- save
        
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
            model.eval()  # <--- enter eval explicitly
            with torch.no_grad():
                # Create plots directory with epoch number
                plot_dir = Path(self.output_dir) / f"epoch_{current_epoch}"
                plot_dir.mkdir(parents=True, exist_ok=True)

                # Use the existing ARD DataLoader directly since it's already properly configured
                # with the correct batch_size, collate_fn, etc.
                plot_dataloader = eval_data

                # Generate plots
                plot_mean_encodings(model, plot_dataloader, self.device, str(plot_dir), epoch=current_epoch)

            print(f"[LatentPlotCallback] Plots saved to {plot_dir}")

        except Exception as e:
            print(f"[LatentPlotCallback] Failed to generate plots: {e}")

        finally:
            # restore original mode
            if was_training:
                model.train()

        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()