import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import ProbLoRALayer from LLaMA model
try:
    from model.model_llama import ProbLoRALayer
except ImportError:
    try:
        # Try relative import paths as fallback
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model.model_llama import ProbLoRALayer
    except ImportError:
        print("[WARNING] Could not import ProbLoRALayer from model_llama. Plotting may not work correctly.")
        ProbLoRALayer = None


def diag_axis_splom(array, latent_image_path, max_sigma):
    dimension = array.shape[1]
    cols = min(dimension, 10)
    rows = (dimension + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, squeeze=False)
    fig.set_size_inches(cols, rows)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    row_index, col_index = 0, 0
    for i in range(dimension):
        axs[row_index, col_index].scatter(array[:, i], array[:, i], s=0.2)
        axs[row_index, col_index].set_xlim(left=-max_sigma, right=max_sigma)
        axs[row_index, col_index].set_ylim(bottom=-max_sigma, top=max_sigma)

        col_index += 1
        if col_index == cols:
            col_index = 0
            row_index += 1

    plt.savefig(latent_image_path)
    plt.close(fig)


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
    
    if ProbLoRALayer is None:
        print("[WARNING] ProbLoRALayer not available. Skipping plotting.")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for batch_index, batch in enumerate(heldout_loader):
        if batch_index > 0:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        
        # Get model outputs with hidden states
        with torch.no_grad():
            outputs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        
        # Extract layers - LLaMA model architecture
        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):  # LLaMA
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):  # GPT-style
            layers = model.transformer.h
        else:
            print("[WARNING] Could not identify LLaMA model architecture for plotting")
            return
            
        for idx, layer in enumerate(layers):
            all_mean_q, all_mean_v = [], []
            
            # Get layer input from hidden states
            if hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > idx:
                layer_input = outputs.hidden_states[idx]
            else:
                print(f"[WARNING] Could not get hidden states for layer {idx}")
                continue
                
            # Find query and value projections in LLaMA architecture
            query_proj, value_proj = None, None
            
            # LLaMA architecture
            if hasattr(layer, 'self_attn'):
                if hasattr(layer.self_attn, 'q_proj') and isinstance(layer.self_attn.q_proj, ProbLoRALayer):
                    query_proj = layer.self_attn.q_proj
                if hasattr(layer.self_attn, 'v_proj') and isinstance(layer.self_attn.v_proj, ProbLoRALayer):
                    value_proj = layer.self_attn.v_proj
            
            # Skip layer if no ProbLoRA layers found
            if query_proj is None or value_proj is None:
                continue
                
            # Get samples if the method exists
            if hasattr(query_proj, 'plot_get_sample'):
                mean_q = query_proj.plot_get_sample(layer_input)
                all_mean_q.append(mean_q)
            if hasattr(value_proj, 'plot_get_sample'):
                mean_v = value_proj.plot_get_sample(layer_input)
                all_mean_v.append(mean_v)
            
            # Skip if we couldn't get samples
            if not all_mean_q or not all_mean_v:
                continue

            # # Find output layer in LLaMA architecture
            # out_layer = None
            # all_mean_out = []
            # try:
            #     # LLaMA architecture
            #     if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj') and isinstance(layer.self_attn.o_proj, ProbLoRALayer):
            #         out_layer = layer.self_attn.o_proj
            # except Exception as e:
            #     print(f"[WARNING] Error finding output layer: {e}")
            #     out_layer = None

            # if out_layer is not None and hasattr(out_layer, 'plot_get_sample'):
            #     mean_out = out_layer.plot_get_sample(layer_input)
            #     all_mean_out = [mean_out]

            arr_q = np.concatenate(all_mean_q, axis=0)
            arr_v = np.concatenate(all_mean_v, axis=0)

            diag_cov_q = np.diag(np.cov(arr_q.T))
            diag_cov_v = np.diag(np.cov(arr_v.T))

            sorted_diag_cov_q = np.sort(diag_cov_q)[::-1]
            sorted_diag_cov_v = np.sort(diag_cov_v)[::-1]
            arg_sorted_q = np.argsort(diag_cov_q)[::-1]
            arg_sorted_v = np.argsort(diag_cov_v)[::-1]

            diag_axis_splom(arr_q, os.path.join(output_dir, f"mean_layer{idx}_q_epoch{epoch}.jpg"), np.sqrt(sorted_diag_cov_q[0]))
            diag_axis_splom(arr_v, os.path.join(output_dir, f"mean_layer{idx}_v_epoch{epoch}.jpg"), np.sqrt(sorted_diag_cov_v[0]))

            x_axis = np.arange(1, len(sorted_diag_cov_q) + 1)

            plt.plot(x_axis, sorted_diag_cov_q, "r", label="Mean Var")
            if hasattr(query_proj, 'est_var'):
                est_var_q = query_proj.est_var.cpu().detach().numpy()
                plt.plot(x_axis, est_var_q[arg_sorted_q], "g", label="Est Var")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"spec_var_layer{idx}_q_epoch{epoch}.jpg"))
            plt.close()

            plt.plot(x_axis, sorted_diag_cov_v, "r", label="Mean Var")
            if hasattr(value_proj, 'est_var'):
                est_var_v = value_proj.est_var.cpu().detach().numpy()
                plt.plot(x_axis, est_var_v[arg_sorted_v], "g", label="Est Var")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"spec_var_layer{idx}_v_epoch{epoch}.jpg"))
            plt.close()

            # if out_layer is not None and hasattr(out_layer, "est_var") and all_mean_out:
            #     arr_out = np.concatenate(all_mean_out, axis=0)
            #     diag_cov_out = np.diag(np.cov(arr_out.T))
            #     sorted_diag_cov_out = np.sort(diag_cov_out)[::-1]
            #     arg_sorted_out = np.argsort(diag_cov_out)[::-1]
            #     x_axis_out = np.arange(1, len(sorted_diag_cov_out) + 1)

            #     plt.plot(x_axis_out, sorted_diag_cov_out, "r", label="Mean Var")
            #     est_var_out = out_layer.est_var.cpu().detach().numpy()
            #     plt.plot(x_axis_out, est_var_out[arg_sorted_out], "g", label="Est Var")
            #     plt.legend()
            #     plt.savefig(os.path.join(output_dir, f"spec_var_layer{idx}_out_epoch{epoch}.jpg"))
            #     plt.close()
        gc.collect()
        torch.cuda.empty_cache()


def validate_plotting_utilities():
    """
    Validate that plotting utilities are properly set up.
    
    Returns:
        bool: True if plotting utilities are available, False otherwise
    """
    if ProbLoRALayer is None:
        print("[INFO] ProbLoRALayer not available - plotting will be disabled")
        return False
    
    try:
        # Test basic matplotlib functionality
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Test numpy functionality
        test_array = np.random.randn(10, 5)
        test_cov = np.cov(test_array.T)
        
        print("[INFO] Plotting utilities validated successfully")
        return True
    except Exception as e:
        print(f"[WARNING] Plotting utilities validation failed: {e}")
        return False
