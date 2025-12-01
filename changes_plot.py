@torch.no_grad()
def plot_mean_encodings(model, heldout_loader, device, output_dir, epoch=0):
    """
    Plot mean encodings for Bayesian LoRA layers.
    
    Args:
        model: The model with ProbLoRA layers
        heldout_loader: DataLoader for plotting data
        device: Device to run on
        output_dir: Directory to save plots
        epoch: Current epoch number
    """
    # model.eval() NOT REQUIRED Taken care in the on_epoch_end() in the LatentPlotCallback



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


# ProbLoRA layer method for deterministic mean encoding extraction for plotting
    def plot_get_sample(self, x):
        """
        Deterministic latent encoding for visualization only.

        Uses FP32 math inside _fp32_ctx to avoid dtype issues (bf16/fp16),
        and returns a CPU float32 numpy array of shape [B*S, rank].
        """
        with _fp32_ctx(x):
            # Work in float32 for numerical stability
            x32 = x.to(torch.float32)

            # Choose the correct "A-like" matrix for the mean
            if self.deterministic:
                # Deterministic mode: only mu_A exists
                mu_A = self.mu_A.to(dtype=torch.float32, device=x32.device)
            else:
                # Probabilistic mode: first rank rows of A are the mean
                mu_A, _ = torch.split(self.A, self.rank, dim=0)
                mu_A = mu_A.to(dtype=torch.float32, device=x32.device)

            # # Apply variance mask if it exists - DISABLED
            # if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            #     # Mask affects the output dimensions of mu_A (the rank dimensions)
            #     mask = self.variance_mask.unsqueeze(1)  # Shape: [rank, 1] 
            #     mu_A_masked = mu_A * mask
            # else:
            #     mu_A_masked = mu_A
            # Convert to input dtype and device for computation consistency
            mu_A_masked = mu_A

            # [B, S, d] -> [B*S, d]
            BS = x32.shape[0] * x32.shape[1]
            x_flat = x32.reshape(BS, x32.shape[-1])

            # Compute mean latent codes: [B*S, rank]
            mu = (mu_A_masked @ x_flat.T).T

            # # Apply mask to the output (latent dimensions) if present
            # if hasattr(self, 'variance_mask') and self.variance_mask is not None:
            #     # Also mask the computed mu to ensure inactive dimensions are zero
            #     mu = mu * self.variance_mask.unsqueeze(0)  # Shape: [1, rank]

            # Convert to CPU float32 numpy for plotting
            mu_samples = mu.cpu().detach().numpy()

        return mu_samples


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

        finally:
            # restore original mode
            if was_training:
                model.train()

        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()