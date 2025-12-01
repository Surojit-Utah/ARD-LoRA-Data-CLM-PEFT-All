import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from contextlib import nullcontext
import torch.cuda.amp as amp
import math

def _fp32_ctx(x):
    # If autocast is on, temporarily disable it; otherwise no-op.
    return torch.amp.autocast(device_type=x.device.type, enabled=False)
    # return amp.autocast(enabled=False) if torch.is_autocast_enabled() else nullcontext()


class ProbLoRALayer(nn.Module):
    """
    Probabilistic LoRA Layer with ARD priors - model-agnostic implementation.
    Works with both encoder (bidirectional) and decoder (causal) architectures.
    The KL divergence computation is independent of attention masking.
    """
    def __init__(self, base_proj: nn.Linear, rank, num_tokens, ard_prior_samples, scaling, 
                 logvar_clamp_min, logvar_clamp_max, 
                 beta_logvar_clamp_min, beta_logvar_clamp_max,
                 sample_clamp_min, sample_clamp_max, deterministic=True, enable_clamps=False, lora_alpha=None):
        super().__init__()

        self.base_proj = base_proj
        self.base_proj.requires_grad_(False)  # freeze base weights
        self.rank = rank
        self.num_tokens = num_tokens
        self.ard_prior_samples = ard_prior_samples
        
        # Compute LoRA scaling: use lora_alpha/rank if provided, otherwise use scaling parameter
        if lora_alpha is not None:
            self.scaling = lora_alpha / rank
            print(f"[INFO] Using standard LoRA scaling: α/r = {lora_alpha}/{rank} = {self.scaling:.3f}")
        else:
            self.scaling = scaling
            print(f"[INFO] Using manual scaling: {self.scaling}")
            
        self.deterministic = deterministic  # Flag for deterministic vs probabilistic mode
        self.enable_clamps = enable_clamps  # Flag to enable/disable numerical stability clamps
        
        # Numerical stability parameters
        self.logvar_clamp_min = logvar_clamp_min
        self.logvar_clamp_max = logvar_clamp_max
        self.beta_logvar_clamp_min = beta_logvar_clamp_min
        self.beta_logvar_clamp_max = beta_logvar_clamp_max
        self.sample_clamp_min = sample_clamp_min
        self.sample_clamp_max = sample_clamp_max

        self.in_features = base_proj.in_features
        self.out_features = base_proj.out_features

        if self.deterministic:
            # Deterministic LoRA: Only create mean parameters (mu_A and B)
            print(f"[INFO] Creating deterministic LoRA layer (rank={rank})")
            mu_A_tensor = torch.empty(rank, self.in_features)
            nn.init.normal_(mu_A_tensor, mean=0.0, std=0.02) # LoRA-default
            self.mu_A = nn.Parameter(mu_A_tensor)
            # No logvar_A or sigma parameters created
        else:
            # Probabilistic LoRA: Create both mean and variance parameters
            print(f"[INFO] Creating probabilistic LoRA layer (rank={rank})")
            # Create raw tensors for both mu and logvar
            A_tensor = torch.empty(2*rank, self.in_features)

            # Views
            M_part, G_part = torch.split(A_tensor, rank, dim=0)

            # BLoB inits
            bound = math.sqrt(6.0 / self.in_features)
            nn.init.uniform_(M_part, -bound, bound)        # mean M: Xavier-UNIFORM
            eps0 = 1e-3                                    # small target variance
            low, high = eps0 / math.sqrt(2.0), eps0        # G ∼ U(ε/√2, ε)
            nn.init.uniform_(G_part, low, high)            # positive, small

            self.A = nn.Parameter(A_tensor)

        # B matrix is always created (shared between deterministic and probabilistic)
        self.B = nn.Parameter(torch.zeros(self.out_features, self.rank)) 

        # ARD prior parameters (only needed for probabilistic mode, but kept for compatibility)
        self.alpha = (self.num_tokens*self.ard_prior_samples) / 2.0
        self.beta = np.zeros(self.rank, dtype=np.float32)
        self.register_buffer('est_var', torch.ones(self.rank))  # Initialize est_var as a buffer

    def forward(self, x):
        """Forward pass - works with any sequence length and masking strategy"""
        base_out = self.base_proj(x)            # shape: [B, S, out_dim]

        with _fp32_ctx(x):  # do sensitive math in FP32
            x32 = x.to(torch.float32)
            
            if self.deterministic:
                # DETERMINISTIC MODE: Use only mean parameters
                mu_A = self.mu_A  # Only mu_A exists, no logvar_A
                B_mat = self.B  # FP32 master

                # variance mask (supports hard bool or soft 0..1) - DISABLED
                # mask_vec = getattr(self, "variance_mask", None)
                # if mask_vec is not None:
                #     mv = mask_vec.to(x32.device, dtype=torch.float32)
                #     mu_A = mu_A * mv.unsqueeze(1)
                #     B_mat = B_mat * mv.unsqueeze(0)

                BS = x32.shape[0] * x32.shape[1]
                x_flat = x32.reshape(BS, x32.shape[-1])

                # Only compute mean output (no variance/sampling)
                z = x_flat @ mu_A.T
                del x_flat

                lora_out32 = z @ B_mat.T
                out32 = lora_out32.reshape(x32.size(0), x32.size(1), -1)
                del z, B_mat, lora_out32, x32
                
            else:
                # PROBABILISTIC MODE: Original ProbLoRA behavior
                mu_A, logvar_A_param = torch.split(self.A, self.rank, dim=0)
                logvar_A = logvar_A_param
                if self.enable_clamps:
                    logvar_A = logvar_A.clamp(self.logvar_clamp_min, self.logvar_clamp_max)

                B_mat = self.B  # FP32 master

                # variance mask (supports hard bool or soft 0..1) - DISABLED
                # mask_vec = getattr(self, "variance_mask", None)
                # if mask_vec is not None:
                #     mv = mask_vec.to(x32.device, dtype=torch.float32)
                #     mu_A = mu_A * mv.unsqueeze(1)
                #     logvar_A = logvar_A * mv.unsqueeze(1)
                #     B_mat = B_mat * mv.unsqueeze(0)

                BS = x32.shape[0] * x32.shape[1]
                x_flat = x32.reshape(BS, x32.shape[-1])

                mu = x_flat @ mu_A.T
                logvar = x_flat @ logvar_A.T
                if self.enable_clamps:
                    logvar = logvar.clamp(self.logvar_clamp_min, self.logvar_clamp_max)
                del x_flat

                if self.training:
                    eps = torch.randn_like(mu)
                    sigma = torch.exp(0.5 * logvar)
                    z = mu + eps * sigma
                    del eps, sigma
                    if self.enable_clamps and self.sample_clamp_min is not None and self.sample_clamp_max is not None:
                        z = z.clamp(self.sample_clamp_min, self.sample_clamp_max)
                else:
                    z = mu  # deterministic eval
                del mu, logvar

                lora_out32 = z @ B_mat.T
                out32 = lora_out32.reshape(x32.size(0), x32.size(1), -1)
                del z, B_mat, lora_out32, x32

        out = out32.to(x.dtype)  # single downcast
        return base_out + self.scaling * out


        # mu_A, logvar_A = torch.split(self.A, self.rank, dim=0)
        
        # # Convert LoRA parameters to input dtype and device for computation (BF16/CUDA compatibility)
        # # Keep original parameters in FP32 for precise gradient updates
        # mu_A = mu_A.to(dtype=x.dtype, device=x.device)
        # logvar_A = logvar_A.to(dtype=x.dtype, device=x.device)
        # B_matrix = self.B.to(dtype=x.dtype, device=x.device)
        
        # # Apply variance mask if it exists
        # if hasattr(self, 'variance_mask') and self.variance_mask is not None:
        #     # Apply mask to both mu_A and logvar_A (mask inactive latent dimensions)
        #     # mu_A and logvar_A shapes: [rank, in_features]
        #     mask = self.variance_mask.unsqueeze(1).to(dtype=x.dtype, device=x.device)  # Shape: [rank, 1]
        #     mu_A_masked = mu_A * mask
        #     logvar_A_masked = logvar_A * mask
        #     # Also apply mask to B matrix - B shape: [out_features, rank]
        #     # Mask the rank dimensions (columns of B)
        #     B_masked = B_matrix * self.variance_mask.unsqueeze(0).to(dtype=x.dtype, device=x.device)  # Shape: [1, rank]
        # else:
        #     # No mask, use original matrices
        #     mu_A_masked = mu_A
        #     logvar_A_masked = logvar_A
        #     B_masked = B_matrix
        
        # B, S, _ = x.size()
        # x_flat = x.view(-1, x.size(-1))  # [B*S, in_features]
        
        # # Compute latent mean and logvar: [B*S, rank]
        # mu = (mu_A_masked @ x_flat.T).T
        # # logvar_A_masked = torch.log(F.softplus(logvar_A_masked) + 1e-6) # Stable gradient for variance
        # logvar = (logvar_A_masked @ x_flat.T).T
        
        # # NUMERICAL STABILITY: Clamp logvar to prevent extreme values during training
        # logvar = torch.clamp(logvar, min=self.logvar_clamp_min, max=self.logvar_clamp_max)  # Prevents exp() overflow/underflow
        
        # # Apply additional masking to latent outputs to ensure inactive dims are zero
        # if hasattr(self, 'variance_mask') and self.variance_mask is not None:
        #     mask_latent = self.variance_mask.unsqueeze(0).to(dtype=x.dtype, device=x.device)  # [1, rank]
        #     mu = mu * mask_latent
        #     logvar = logvar * mask_latent
        
        # eps = torch.randn_like(mu)
        # z = mu + eps * torch.exp(0.5 * logvar)  # [B*S, rank]
        
        # # Final LoRA output: z @ B_masked^T
        # out = (z @ B_masked.T).view(B, S, -1)     # shape: [B, S, out_dim]
        # final_out = base_out + self.scaling * out

        # return final_out

    def kl_divergence_latent(self, x):
        """
        Optimized KL divergence computation - model-agnostic.
        Works with both causal (LLaMA) and bidirectional (DeBERTa) attention.
        The computation is in the latent space and independent of masking strategy.
        
        Returns zero KL divergence for deterministic mode.
        """
        if self.deterministic:
            # Deterministic mode: No KL divergence (no variance parameters)
            return torch.tensor(0.0, device=x.device, dtype=x.dtype, requires_grad=True)
        
        # Probabilistic mode: Original KL divergence computation
        with _fp32_ctx(x):
            x32 = x.to(torch.float32)
            mu_A, logvar_A_param = torch.split(self.A, self.rank, dim=0)
            logvar_A = logvar_A_param

            # variance mask handling - DISABLED
            # mv = getattr(self, "variance_mask", None)
            # if mv is not None:
            #     mvf = mv.to(x32.device, dtype=torch.float32)
            #     if mvf.sum() == 0:
            #         return torch.tensor(0.0, device=x.device, dtype=x.dtype, requires_grad=True)
            #     mu_A = mu_A * mvf.unsqueeze(1)
            #     logvar_A = logvar_A * mvf.unsqueeze(1)

            BS = x32.shape[0] * x32.shape[1]
            x_flat = x32.reshape(BS, x32.shape[-1])

            mu = x_flat @ mu_A.T
            logvar = x_flat @ logvar_A.T
            # Apply clamps only if enabled
            if self.enable_clamps:
                logvar = logvar.clamp(self.beta_logvar_clamp_min, self.beta_logvar_clamp_max)

            var = torch.exp(logvar)
            tvar = (self.est_var.to(x32.device) + 1e-6).unsqueeze(0)

            # variance mask filtering - DISABLED
            # if mv is not None:
            #     idx = torch.where(mv.to(x32.device) > 0)[0]
            #     mu, logvar, var, tvar = mu[:, idx], logvar[:, idx], var[:, idx], tvar[:, idx]

            kld = 0.5 * (torch.log(tvar) - logvar + ((var + mu.pow(2)) / tvar) - 1.0)
            out = kld.mean()

        # return in model dtype to avoid dtype mismatches upstream
        return out.to(x.dtype)

        # mu_A, logvar_A = torch.split(self.A, self.rank, dim=0)
        # B, S, _ = x.size()  # batch_size, seq_len, features
        
        # # Early exit for zero active dimensions
        # if hasattr(self, 'variance_mask') and self.variance_mask is not None:
        #     active_dims = torch.sum(self.variance_mask > 0).item()
        #     if active_dims == 0:
        #         return torch.tensor(0.0, device=x.device, requires_grad=True)
        # else:
        #     active_dims = self.rank
        
        # # Flatten input once
        # x_flat = x.view(-1, x.size(-1))  # [B*S, in_features]
        
        # # Convert to input dtype and device for computation consistency (cached)
        # mu_A = mu_A.to(dtype=x.dtype, device=x.device)
        # logvar_A = logvar_A.to(dtype=x.dtype, device=x.device)
        
        # # Apply variance mask if it exists (optimized)
        # if hasattr(self, 'variance_mask') and self.variance_mask is not None:
        #     mask = self.variance_mask.unsqueeze(1).to(dtype=x.dtype, device=x.device)  # Shape: [rank, 1]
        #     mu_A_masked = mu_A * mask
        #     logvar_A_masked = logvar_A * mask
            
        #     # Optimized matrix operations for masked case
        #     mu = torch.mm(x_flat, mu_A_masked.T)  # [B*S, rank] - more efficient than transpose operations
        #     logvar = torch.mm(x_flat, logvar_A_masked.T)  # [B*S, rank]
            
        #     # Get target variance for active dimensions only
        #     active_mask = self.variance_mask > 0
        #     if hasattr(self, 'est_var') and self.est_var is not None:
        #         target_var = self.est_var[active_mask].to(x.device).unsqueeze(0)
        #     else:
        #         target_var = torch.ones(active_dims, device=x.device, dtype=x.dtype).unsqueeze(0)
            
        #     # Select only active dimensions for computation
        #     active_indices = torch.where(self.variance_mask > 0)[0]
        #     mu_active = mu[:, active_indices]
        #     logvar_active = logvar[:, active_indices]
            
        #     # Optimized KL computation: avoid redundant exp() call
        #     # KL = 0.5 * (log(target_var) - logvar + (exp(logvar) + mu²) / target_var - 1)
        #     mu_squared = mu_active.pow(2)
        #     exp_logvar = torch.exp(logvar_active)
        #     log_target_var = torch.log(target_var)
            
        #     kld = 0.5 * (log_target_var - logvar_active + 
        #                 (exp_logvar + mu_squared) / target_var - 1.0)
        # else:
        #     # Optimized matrix operations for non-masked case
        #     mu = torch.mm(x_flat, mu_A.T)  # [B*S, rank] - more efficient
        #     logvar = torch.mm(x_flat, logvar_A.T)  # [B*S, rank]
            
        #     # Get target variance
        #     target_var = self.est_var.to(x.device).unsqueeze(0)
            
        #     # Optimized KL computation: avoid redundant exp() call
        #     mu_squared = mu.pow(2)
        #     exp_logvar = torch.exp(logvar)
        #     log_target_var = torch.log(target_var)
            
        #     kld = 0.5 * (log_target_var - logvar + 
        #                 (exp_logvar + mu_squared) / target_var - 1.0)
        
        # # Return mean KL divergence
        # return kld.mean()
    
    def beta_get_sample(self, x):
        """Sample from latent distribution for ARD prior estimation with numerical stability"""
        if self.deterministic:
            # Deterministic mode: Return mean output (no sampling)
            mu_A = self.mu_A.to(dtype=x.dtype, device=x.device)
            
            # Apply variance mask if it exists - DISABLED
            # if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            #     mask = self.variance_mask.unsqueeze(1).to(dtype=x.dtype, device=x.device)  # Shape: [rank, 1]
            #     mu_A_masked = mu_A * mask
            # else:
            #     mu_A_masked = mu_A
            mu_A_masked = mu_A
                
            x_flat = x.view(-1, x.size(-1))
            mu = (mu_A_masked @ x_flat.T).T      # [B*S, rank]
            
            # Apply latent masking - DISABLED
            # if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            #     mask_latent = self.variance_mask.unsqueeze(0).to(dtype=x.dtype, device=x.device)  # [1, rank]
            #     mu = mu * mask_latent
            
            # For deterministic mode, return mean as "samples"
            samples_float = mu.float().cpu().detach()
            return samples_float.numpy()
        
        # Probabilistic mode: Original sampling behavior
        mu_A, logvar_A = torch.split(self.A, self.rank, dim=0)
        
        # Convert to input dtype and device for computation consistency
        mu_A = mu_A.to(dtype=x.dtype, device=x.device)
        logvar_A = logvar_A.to(dtype=x.dtype, device=x.device)
        
        # Apply variance mask if it exists - DISABLED
        # if hasattr(self, 'variance_mask') and self.variance_mask is not None:
        #     # Mask the latent dimensions (output dims of A matrices)
        #     mask = self.variance_mask.unsqueeze(1).to(dtype=x.dtype, device=x.device)  # Shape: [rank, 1]
        #     mu_A_masked = mu_A * mask
        #     logvar_A_masked = logvar_A * mask
        # else:
        #     mu_A_masked = mu_A
        #     logvar_A_masked = logvar_A
        mu_A_masked = mu_A
        logvar_A_masked = logvar_A
            
        x_flat = x.view(-1, x.size(-1))
        mu = (mu_A_masked @ x_flat.T).T      # [B*S, rank]
        logvar = (logvar_A_masked @ x_flat.T).T  # [B*S, rank]
        
        # Apply additional masking to latent outputs - DISABLED
        # if hasattr(self, 'variance_mask') and self.variance_mask is not None:
        #     mask_latent = self.variance_mask.unsqueeze(0).to(dtype=x.dtype, device=x.device)  # [1, rank]
        #     mu = mu * mask_latent
        #     logvar = logvar * mask_latent
        
        # NUMERICAL STABILITY: Clamp logvar to prevent extreme values (only if enabled)
        if self.enable_clamps:
            logvar = torch.clamp(logvar, min=self.beta_logvar_clamp_min, max=self.beta_logvar_clamp_max)  # Prevents exp() overflow/underflow
        
        eps = torch.randn_like(mu)
        samples = mu + eps * torch.exp(0.5 * logvar)  # [B*S, rank]
        
        # NUMERICAL STABILITY: Clamp samples to prevent overflow in beta accumulation (only if enabled)
        if self.enable_clamps:
            samples = torch.clamp(samples, min=self.sample_clamp_min, max=self.sample_clamp_max)  # Prevents overflow in square operation
        
        # Convert to float32 before numpy conversion (BFloat16 not supported by numpy)
        samples_float = samples.float().cpu().detach()
        
        # NUMERICAL STABILITY: Check for inf/nan before returning
        if torch.isnan(samples_float).any() or torch.isinf(samples_float).any():
            print(f"[WARNING] NaN/Inf detected in beta samples, using zeros for stability")
            return np.zeros_like(samples_float.numpy())
        
        return samples_float.numpy()

    def plot_get_sample(self, x):
        mu_A, logvar_A = torch.split(self.A, self.rank, dim=0)
        
        # Apply variance mask if it exists
        if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            # Mask affects the output dimensions of mu_A (the rank dimensions)
            mask = self.variance_mask.unsqueeze(1)  # Shape: [rank, 1] 
            mu_A_masked = mu_A * mask
        else:
            mu_A_masked = mu_A
            
        x_flat = x.view(-1, x.size(-1))
        mu = (mu_A_masked @ x_flat.T).T  # mu shape: [batch*seq, rank]
        
        # Apply mask to the output (latent dimensions) if present
        if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            # Also mask the computed mu to ensure inactive dimensions are zero
            mu = mu * self.variance_mask.unsqueeze(0)  # Shape: [1, rank]
            
        mu_samples = mu.cpu().detach().numpy()
        return mu_samples



def inject_problora_llama(model, rank, scaling, num_tokens, ard_prior_samples,
                         logvar_clamp_min, logvar_clamp_max,
                         beta_logvar_clamp_min, beta_logvar_clamp_max,
                         sample_clamp_min, sample_clamp_max, attn_implementation,
                         target_attention_layers=None, deterministic=False, enable_clamps=True, lora_alpha=None):
    """
    Inject ProbLoRA into LLaMA2-7B model.
    Targets configurable attention projections based on YAML configuration.
    
    Args:
        target_attention_layers: List of attention projection names to inject (from YAML config - REQUIRED)
        deterministic: If True, creates deterministic LoRA (no variance parameters or KL loss)
        enable_clamps: If True, applies numerical stability clamps (default: True)
        lora_alpha: LoRA alpha parameter for computing scaling = alpha/rank (standard LoRA)
    """
    # Compute effective scaling
    if lora_alpha is not None:
        effective_scaling = lora_alpha / rank
        scaling_info = f"α/r = {lora_alpha}/{rank} = {effective_scaling:.3f}"
    else:
        effective_scaling = scaling
        scaling_info = f"manual = {scaling}"
        
    mode_str = "deterministic LoRA" if deterministic else "probabilistic LoRA (ProbLoRA)"
    clamp_str = "with clamps" if enable_clamps else "without clamps"
    print(f"[INFO] Injecting {mode_str} into LLaMA2 model with rank={rank} ({clamp_str})")
    print(f"[INFO] LoRA scaling: {scaling_info}")
    
    # Validate required parameters
    if target_attention_layers is None:
        raise ValueError("target_attention_layers must be provided from YAML configuration")
    
    print(f"[INFO] Target attention layers: {target_attention_layers}")
    print(f"[INFO] Mode: {mode_str}")
    
    # Set attention implementation from YAML config for A100 optimization
    if hasattr(model, 'config') and attn_implementation is not None:
        model.config.attn_implementation = attn_implementation
        print(f"[INFO] Set attention implementation to {attn_implementation} for A100 optimization")
    
    layers_modified = 0
    
    # LLaMA2 has model.layers (list of transformer blocks)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer_idx, layer in enumerate(model.model.layers):
            # Inject ProbLoRA into attention projections (configurable)
            if hasattr(layer, 'self_attn') and target_attention_layers:
                attn = layer.self_attn
                
                # Wrap configured attention projections
                for proj_name in target_attention_layers:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), nn.Linear):
                        setattr(attn, proj_name, ProbLoRALayer(
                            getattr(attn, proj_name), rank, num_tokens, ard_prior_samples, scaling,
                            logvar_clamp_min, logvar_clamp_max,
                            beta_logvar_clamp_min, beta_logvar_clamp_max,
                            sample_clamp_min, sample_clamp_max, deterministic=deterministic, 
                            enable_clamps=enable_clamps, lora_alpha=lora_alpha))
                        layers_modified += 1
    
    print(f"[INFO] Successfully injected {mode_str} into {layers_modified} linear layers")
    if deterministic:
        print(f"[INFO] Each layer has deterministic LoRA with rank={rank} (no variance parameters)")
        print(f"[INFO] KL divergence will be zero (deterministic mode)")
    else:
        print(f"[INFO] Each layer now has ARD-enabled latent space with rank={rank}")
        print(f"[INFO] KL divergence will be computed in latent space (model-agnostic)")
    print(f"[INFO] LoRA injected into attention: {target_attention_layers}")
    
    return model