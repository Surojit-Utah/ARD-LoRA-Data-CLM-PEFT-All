#!/usr/bin/env python3
"""
Comprehensive Trainable Parameter Analysis for ARD-LoRA

This script provides detailed analysis of trainable parameters in the ARD-LoRA model,
breaking down parameters by layer, head, and matrix type (A, B, G).

Usage:
    python scripts/analyze_trainable_parameters.py --model_path <path_to_model> --output <output_file>
    
Features:
- Layer-wise parameter breakdown
- Head-wise parameter counting for multi-head attention
- Matrix-wise analysis (A, B, G matrices in ProbLoRA)
- Detailed logging to text file
- Validation against expected total parameter count
"""

import os
import sys
import argparse
import torch
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import traceback

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from model.model_llama import ProbLoRALayer, inject_problora_llama
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


class ParameterAnalyzer:
    """Comprehensive parameter analysis for ARD-LoRA models."""
    
    def __init__(self, model, output_file: str = None):
        """
        Initialize parameter analyzer.
        
        Args:
            model: The model to analyze
            output_file: Path to output log file (optional)
        """
        self.model = model
        self.output_file = output_file
        self.log_lines = []
        
        # Parameter tracking
        self.layer_stats = defaultdict(dict)
        self.total_trainable = 0
        self.total_parameters = 0
        
    def log(self, message: str, print_console: bool = True):
        """Log message to both console and internal log."""
        if print_console:
            print(message)
        self.log_lines.append(message)
        
    def save_log(self):
        """Save log to file if output file is specified."""
        if self.output_file:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.log_lines))
            self.log(f"Parameter analysis saved to: {self.output_file}")
    
    def analyze_parameter_tensor(self, param: torch.Tensor, param_name: str) -> Dict[str, Any]:
        """Analyze a single parameter tensor."""
        return {
            'name': param_name,
            'shape': list(param.shape),
            'numel': param.numel(),
            'requires_grad': param.requires_grad,
            'dtype': str(param.dtype),
            'device': str(param.device)
        }
    
    def analyze_problora_layer(self, layer_name: str, layer_module) -> Dict[str, Any]:
        """Analyze a ProbLoRA layer in detail."""
        layer_info = {
            'layer_name': layer_name,
            'layer_type': type(layer_module).__name__,
            'total_params': 0,
            'trainable_params': 0,
            'matrices': {},
            'heads': {}
        }
        
        # Check if this is a ProbLoRA layer
        if isinstance(layer_module, ProbLoRALayer):
            self.log(f"\nAnalyzing ProbLoRA Layer: {layer_name}")
            
            # Analyze A matrix
            if hasattr(layer_module, 'A'):
                A_info = self.analyze_parameter_tensor(layer_module.A, f"{layer_name}.A")
                layer_info['matrices']['A'] = A_info
                layer_info['total_params'] += A_info['numel']
                if A_info['requires_grad']:
                    layer_info['trainable_params'] += A_info['numel']
                self.log(f"   A matrix: {A_info['shape']} = {A_info['numel']:,} params (trainable: {A_info['requires_grad']})")
            
            # For deterministic mode, check mu_A instead of A
            elif hasattr(layer_module, 'mu_A'):
                A_info = self.analyze_parameter_tensor(layer_module.mu_A, f"{layer_name}.mu_A")
                layer_info['matrices']['A'] = A_info
                layer_info['total_params'] += A_info['numel']
                if A_info['requires_grad']:
                    layer_info['trainable_params'] += A_info['numel']
                self.log(f"   mu_A matrix: {A_info['shape']} = {A_info['numel']:,} params (trainable: {A_info['requires_grad']})")
            
            # Analyze B matrix
            if hasattr(layer_module, 'B'):
                B_info = self.analyze_parameter_tensor(layer_module.B, f"{layer_name}.B")
                layer_info['matrices']['B'] = B_info
                layer_info['total_params'] += B_info['numel']
                if B_info['requires_grad']:
                    layer_info['trainable_params'] += B_info['numel']
                self.log(f"   B matrix: {B_info['shape']} = {B_info['numel']:,} params (trainable: {B_info['requires_grad']})")
            
            # Analyze G matrix (variance parameters) - only in probabilistic mode
            if hasattr(layer_module, 'G'):
                G_info = self.analyze_parameter_tensor(layer_module.G, f"{layer_name}.G")
                layer_info['matrices']['G'] = G_info
                layer_info['total_params'] += G_info['numel']
                if G_info['requires_grad']:
                    layer_info['trainable_params'] += G_info['numel']
                self.log(f"   G matrix: {G_info['shape']} = {G_info['numel']:,} params (trainable: {G_info['requires_grad']})")
            
            # Report layer mode
            mode = "deterministic LoRA" if layer_module.deterministic else "probabilistic LoRA"
            self.log(f"   Mode: {mode} (rank={layer_module.rank})")
            
            # Layer summary
            self.log(f"   Layer Total: {layer_info['total_params']:,} params ({layer_info['trainable_params']:,} trainable)")
            
        return layer_info
    
    def analyze_model(self) -> Dict[str, Any]:
        """Perform comprehensive model analysis focused on ProbLoRA parameters."""
        self.log("PROBLORA-FOCUSED TRAINABLE PARAMETER ANALYSIS")
        self.log("=" * 80)
        self.log(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Model Type: {type(self.model).__name__}")
        self.log(f"Focus: ProbLoRA layers (A, B, G matrices) only")
        self.log("")
        
        # Track all parameters first, but distinguish ProbLoRA vs base model
        problora_trainable = 0
        base_model_trainable = 0
        
        for name, param in self.model.named_parameters():
            self.total_parameters += param.numel()
            if param.requires_grad:
                self.total_trainable += param.numel()
                # Check if this is a ProbLoRA parameter (A, B, G matrices, or mu_A for deterministic)
                if any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A']):
                    problora_trainable += param.numel()
                else:
                    base_model_trainable += param.numel()
        
        self.log(f"OVERALL PARAMETER SUMMARY:")
        self.log(f"   Total parameters: {self.total_parameters:,}")
        self.log(f"   Total trainable parameters: {self.total_trainable:,}")
        self.log(f"   ProbLoRA trainable parameters: {problora_trainable:,}")
        self.log(f"   Base model trainable parameters: {base_model_trainable:,}")
        self.log(f"   ProbLoRA percentage of total: {(problora_trainable/self.total_parameters)*100:.4f}%")
        self.log(f"   ProbLoRA percentage of trainable: {(problora_trainable/self.total_trainable)*100:.2f}%")
        self.log("")
        
        # Warn if base model has trainable parameters (shouldn't happen with proper LoRA)
        if base_model_trainable > 0:
            self.log(f"WARNING: Base model has {base_model_trainable:,} trainable parameters!")
            self.log(f"         In proper LoRA setup, only ProbLoRA matrices should be trainable.")
            self.log("")
        
        # Analyze ProbLoRA layers specifically
        problora_layers_found = 0
        total_problora_params = 0
        
        self.log("PROBLORA LAYER-BY-LAYER ANALYSIS:")
        self.log("-" * 60)
        
        for name, module in self.model.named_modules():
            # Check if this module is a ProbLoRA layer
            if isinstance(module, ProbLoRALayer):
                layer_info = self.analyze_problora_layer(name, module)
                self.layer_stats[name] = layer_info
                problora_layers_found += 1
                total_problora_params += layer_info['trainable_params']
        
        # Summary statistics
        self.log(f"\nPROBLORA LAYER SUMMARY:")
        self.log(f"   ProbLoRA layers found: {problora_layers_found}")
        self.log(f"   Total ProbLoRA trainable params: {total_problora_params:,}")
        self.log(f"   Non-ProbLoRA trainable params: {self.total_trainable - total_problora_params:,}")
        
        # Detailed breakdown by matrix type
        self.analyze_matrix_breakdown()
        
        # Validation
        self.validate_parameter_count()
        
        return {
            'total_parameters': self.total_parameters,
            'total_trainable': self.total_trainable,
            'problora_layers': problora_layers_found,
            'problora_trainable': total_problora_params,
            'layer_stats': dict(self.layer_stats)
        }
    
    def analyze_matrix_breakdown(self):
        """Analyze parameters by matrix type (A, B, G)."""
        self.log(f"\nMATRIX TYPE BREAKDOWN:")
        self.log("-" * 40)
        
        matrix_totals = {'A': 0, 'B': 0, 'G': 0}
        head_analysis = {'A': [], 'B': []}
        
        for layer_name, layer_info in self.layer_stats.items():
            for matrix_type in ['A', 'B', 'G']:
                if matrix_type in layer_info['matrices']:
                    matrix_info = layer_info['matrices'][matrix_type]
                    if matrix_info['requires_grad']:
                        matrix_totals[matrix_type] += matrix_info['numel']
                
                # Collect head analysis for A and B
                if matrix_type in ['A', 'B'] and matrix_type in layer_info['heads']:
                    head_info = layer_info['heads'][matrix_type]
                    head_analysis[matrix_type].append({
                        'layer': layer_name,
                        'num_heads': head_info['num_heads'],
                        'params_per_head': head_info['params_per_head'],
                        'total_params': head_info['total_params']
                    })
        
        # Print matrix totals
        for matrix_type, total in matrix_totals.items():
            self.log(f"   {matrix_type} matrices: {total:,} parameters")
        
        # Print head analysis
        for matrix_type in ['A', 'B']:
            if head_analysis[matrix_type]:
                self.log(f"\n{matrix_type} MATRIX HEAD ANALYSIS:")
                total_heads = 0
                for head_info in head_analysis[matrix_type]:
                    layer_short = head_info['layer'].split('.')[-2] if '.' in head_info['layer'] else head_info['layer']
                    self.log(f"   {layer_short}: {head_info['num_heads']} heads × {head_info['params_per_head']:,} = {head_info['total_params']:,}")
                    total_heads += head_info['num_heads']
                self.log(f"   Total {matrix_type} heads across all layers: {total_heads}")
    
    def validate_parameter_count(self):
        """Validate the parameter count against expected values, focusing on ProbLoRA."""
        self.log(f"\nPROBLORA PARAMETER VALIDATION:")
        self.log("-" * 40)
        
        expected_count = 12_582_912  # The reported trainable parameter count
        actual_count = self.total_trainable
        
        # Count ProbLoRA-specific parameters (including mu_A for deterministic mode)
        problora_count = sum(p.numel() for name, p in self.model.named_parameters()
                            if p.requires_grad and any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A']))
        
        non_problora_count = actual_count - problora_count
        
        self.log(f"   Expected total trainable parameters: {expected_count:,}")
        self.log(f"   Actual total trainable parameters:   {actual_count:,}")
        self.log(f"   ProbLoRA trainable parameters:       {problora_count:,}")
        self.log(f"   Non-ProbLoRA trainable parameters:   {non_problora_count:,}")
        self.log(f"   Difference from expected: {abs(actual_count - expected_count):,}")
        
        if actual_count == expected_count:
            self.log("   ✓ Parameter count matches expected value!")
        else:
            percentage_diff = abs(actual_count - expected_count) / expected_count * 100
            self.log(f"   Parameter count differs by {percentage_diff:.2f}%")
            
            if percentage_diff < 1.0:
                self.log("   Small difference, likely due to rounding or additional parameters")
            else:
                self.log("   Significant difference, investigation needed")
        
        # ProbLoRA-specific validation
        if non_problora_count == 0:
            self.log("   ✓ EXCELLENT: Only ProbLoRA parameters are trainable (proper LoRA setup)")
        else:
            self.log(f"   ⚠ WARNING: {non_problora_count:,} non-ProbLoRA parameters are trainable")
            self.log("     This suggests base model parameters may not be properly frozen")
        
        # Expected ProbLoRA parameter breakdown (for reference)
        expected_problora = expected_count if non_problora_count == 0 else problora_count
        if problora_count == expected_problora:
            self.log(f"   ✓ ProbLoRA parameter count is correct: {problora_count:,}")
        else:
            diff = abs(problora_count - expected_problora)
            self.log(f"   ⚠ ProbLoRA parameter difference: {diff:,} ({diff/expected_problora*100:.2f}%)")

    def validate_optimizer_parameters(self, trainer=None, optimizer=None):
        """
        Validate that HuggingFace trainer/optimizer is using the correct set of trainable parameters.
        
        Args:
            trainer: HuggingFace Trainer instance (optional)
            optimizer: PyTorch optimizer instance (optional)
        """
        self.log(f"\nOPTIMIZER PARAMETER VALIDATION:")
        self.log("-" * 45)
        
        # Get model's trainable parameters
        model_trainable_params = []
        model_param_names = []
        model_param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                model_trainable_params.append(param)
                model_param_names.append(name)
                model_param_count += param.numel()
        
        self.log(f"   Model trainable parameters: {model_param_count:,}")
        self.log(f"   Model trainable parameter tensors: {len(model_trainable_params)}")
        
        # Check trainer's optimizer if provided
        if trainer is not None:
            self.log(f"\n   TRAINER OPTIMIZER ANALYSIS:")
            try:
                # Force optimizer creation if it doesn't exist
                if trainer.optimizer is None:
                    self.log(f"   Creating optimizer (trainer.optimizer was None)...")
                    trainer.create_optimizer()
                
                trainer_optimizer = trainer.optimizer
                
                # Analyze optimizer parameters
                optimizer_param_count = 0
                optimizer_param_tensors = 0
                optimizer_param_groups = len(trainer_optimizer.param_groups)
                
                for i, group in enumerate(trainer_optimizer.param_groups):
                    group_param_count = 0
                    for param in group['params']:
                        optimizer_param_count += param.numel()
                        group_param_count += param.numel()
                        optimizer_param_tensors += 1
                    self.log(f"   Parameter group {i+1}: {group_param_count:,} parameters")
                
                self.log(f"   Optimizer total parameters: {optimizer_param_count:,}")
                self.log(f"   Optimizer parameter tensors: {optimizer_param_tensors}")
                self.log(f"   Optimizer parameter groups: {optimizer_param_groups}")
                
                # Validation
                if optimizer_param_count == model_param_count:
                    self.log(f"   PASS: Optimizer parameter count matches model!")
                else:
                    diff = abs(optimizer_param_count - model_param_count)
                    self.log(f"   FAIL: Parameter count mismatch!")
                    self.log(f"   Difference: {diff:,} parameters")
                
                if optimizer_param_tensors == len(model_trainable_params):
                    self.log(f"   PASS: Optimizer tensor count matches model!")
                else:
                    diff = abs(optimizer_param_tensors - len(model_trainable_params))
                    self.log(f"   FAIL: Tensor count mismatch!")
                    self.log(f"   Difference: {diff} tensors")
                
            except Exception as e:
                self.log(f"   ERROR analyzing trainer optimizer: {e}")
                return None
        
        # Check standalone optimizer if provided
        if optimizer is not None:
            self.log(f"\n   STANDALONE OPTIMIZER ANALYSIS:")
            try:
                optimizer_param_count = 0
                optimizer_param_tensors = 0
                optimizer_param_groups = len(optimizer.param_groups)
                
                for i, group in enumerate(optimizer.param_groups):
                    group_param_count = 0
                    for param in group['params']:
                        optimizer_param_count += param.numel()
                        group_param_count += param.numel()
                        optimizer_param_tensors += 1
                    self.log(f"   Parameter group {i+1}: {group_param_count:,} parameters")
                
                self.log(f"   Optimizer total parameters: {optimizer_param_count:,}")
                self.log(f"   Optimizer parameter tensors: {optimizer_param_tensors}")
                self.log(f"   Optimizer parameter groups: {optimizer_param_groups}")
                
                # Validation
                if optimizer_param_count == model_param_count:
                    self.log(f"   PASS: Optimizer parameter count matches model!")
                else:
                    diff = abs(optimizer_param_count - model_param_count)
                    self.log(f"   FAIL: Parameter count mismatch!")
                    self.log(f"   Difference: {diff:,} parameters")
                
                if optimizer_param_tensors == len(model_trainable_params):
                    self.log(f"   PASS: Optimizer tensor count matches model!")
                else:
                    diff = abs(optimizer_param_tensors - len(model_trainable_params))
                    self.log(f"   FAIL: Tensor count mismatch!")
                    self.log(f"   Difference: {diff} tensors")
                
            except Exception as e:
                self.log(f"   ERROR analyzing standalone optimizer: {e}")
                return None
        
        # Detailed parameter name analysis
        if trainer is not None or optimizer is not None:
            self.log(f"\n   DETAILED PARAMETER ANALYSIS:")
            
            # Show sample of trainable parameter names
            self.log(f"   Sample trainable parameter names:")
            for i, name in enumerate(model_param_names[:10]):  # Show first 10
                param = dict(self.model.named_parameters())[name]
                self.log(f"     {i+1:2d}. {name} - {list(param.shape)} - {param.numel():,} params")
            
            if len(model_param_names) > 10:
                self.log(f"     ... and {len(model_param_names) - 10} more parameters")
            
            # Check for parameters that should be trainable (ProbLoRA specific)
            problora_params = [name for name in model_param_names if any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A'])]
            non_problora_params = [name for name in model_param_names if not any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A'])]
            
            self.log(f"\n   PARAMETER CLASSIFICATION:")
            self.log(f"     ProbLoRA parameters: {len(problora_params)}")
            self.log(f"     Non-ProbLoRA parameters: {len(non_problora_params)}")
            
            if non_problora_params:
                self.log(f"     WARNING: Found non-ProbLoRA trainable parameters:")
                for name in non_problora_params[:5]:  # Show first 5
                    param = dict(self.model.named_parameters())[name]
                    self.log(f"       - {name} - {param.numel():,} params")
                if len(non_problora_params) > 5:
                    self.log(f"       ... and {len(non_problora_params) - 5} more")
            else:
                self.log(f"     PASS: All trainable parameters are ProbLoRA parameters")
        
        return {
            'model_param_count': model_param_count,
            'model_param_tensors': len(model_trainable_params),
            'problora_param_names': [name for name in model_param_names if any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A'])],
            'non_problora_param_names': [name for name in model_param_names if not any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A'])]
        }

    def validate_gradient_flow_and_updates(self, trainer=None):
        """
        Validate that gradients flow correctly and only trainable parameters update.
        This is the ultimate test for proper LoRA parameter handling.
        """
        self.log(f"\nGRADIENT FLOW AND PARAMETER UPDATE VALIDATION:")
        self.log("=" * 60)
        
        if trainer is None:
            self.log("   ERROR: Trainer required for gradient flow validation")
            return None
        
        try:
            # Force optimizer creation if needed
            if trainer.optimizer is None:
                trainer.create_optimizer()
            
            # Create dummy input for forward/backward pass
            tokenizer = trainer.tokenizer
            dummy_text = "The quick brown fox jumps over the lazy dog."
            inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=32)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            self.log(f"   Using dummy input: '{dummy_text}'")
            self.log(f"   Input shape: {input_ids.shape}")
            
            # Switch to training mode and clear gradients
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            
            self.log(f"\n   FORWARD PASS AND GRADIENT COMPUTATION:")
            
            # Forward pass with loss computation
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            self.log(f"   Loss computed: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            self.log(f"   Backward pass completed")
            
            # FIRST: Check parameter requires_grad status for all A/mu_A parameters
            self.log(f"\n   PARAMETER REQUIRES_GRAD STATUS CHECK:")
            mu_A_params_found = 0
            mu_A_requires_grad = 0
            B_params_found = 0
            B_requires_grad = 0
            
            for name, param in self.model.named_parameters():
                if '.mu_A' in name:
                    mu_A_params_found += 1
                    if param.requires_grad:
                        mu_A_requires_grad += 1
                    else:
                        self.log(f"     WARNING: {name} has requires_grad=False!")
                elif '.B' in name and any(attn in name for attn in ['q_proj', 'k_proj', 'v_proj']):
                    B_params_found += 1
                    if param.requires_grad:
                        B_requires_grad += 1
                    else:
                        self.log(f"     WARNING: {name} has requires_grad=False!")
            
            self.log(f"     mu_A parameters found: {mu_A_params_found}")
            self.log(f"     mu_A parameters with requires_grad=True: {mu_A_requires_grad}")
            self.log(f"     B parameters found: {B_params_found}")
            self.log(f"     B parameters with requires_grad=True: {B_requires_grad}")
            
            if mu_A_requires_grad == 0:
                self.log(f"     CRITICAL: NO mu_A parameters have requires_grad=True!")
                self.log(f"     This explains why mu_A doesn't receive gradients")
            elif mu_A_requires_grad < mu_A_params_found:
                self.log(f"     ISSUE: Only {mu_A_requires_grad}/{mu_A_params_found} mu_A parameters have requires_grad=True")
            else:
                self.log(f"     GOOD: All mu_A parameters have requires_grad=True")
            
            # SECOND: Check if mu_A parameters are in optimizer parameter groups
            self.log(f"\n   OPTIMIZER PARAMETER GROUP CHECK:")
            mu_A_in_optimizer = 0
            B_in_optimizer = 0
            optimizer_param_ids = set()
            
            # Collect all parameter IDs from optimizer
            for group in trainer.optimizer.param_groups:
                for param in group['params']:
                    optimizer_param_ids.add(id(param))
            
            # Check if mu_A and B parameters are in optimizer
            for name, param in self.model.named_parameters():
                if '.mu_A' in name:
                    if id(param) in optimizer_param_ids:
                        mu_A_in_optimizer += 1
                    else:
                        self.log(f"     ERROR: {name} NOT in optimizer parameter groups!")
                elif '.B' in name and any(attn in name for attn in ['q_proj', 'k_proj', 'v_proj']):
                    if id(param) in optimizer_param_ids:
                        B_in_optimizer += 1
                    else:
                        self.log(f"     ERROR: {name} NOT in optimizer parameter groups!")
            
            self.log(f"     mu_A parameters in optimizer: {mu_A_in_optimizer}/{mu_A_params_found}")
            self.log(f"     B parameters in optimizer: {B_in_optimizer}/{B_params_found}")
            
            if mu_A_in_optimizer < mu_A_params_found:
                self.log(f"     CRITICAL: {mu_A_params_found - mu_A_in_optimizer} mu_A parameters missing from optimizer!")
            else:
                self.log(f"     GOOD: All mu_A parameters are in optimizer")
            
            # THIRD: Check computational graph connectivity for a sample mu_A parameter
            self.log(f"\n   COMPUTATIONAL GRAPH CONNECTIVITY CHECK:")
            sample_mu_A_param = None
            sample_mu_A_name = None
            
            for name, param in self.model.named_parameters():
                if '.mu_A' in name:
                    sample_mu_A_param = param
                    sample_mu_A_name = name
                    break
            
            if sample_mu_A_param is not None:
                self.log(f"     Checking sample parameter: {sample_mu_A_name}")
                self.log(f"     requires_grad: {sample_mu_A_param.requires_grad}")
                self.log(f"     is_leaf: {sample_mu_A_param.is_leaf}")
                self.log(f"     grad_fn: {sample_mu_A_param.grad_fn}")
                
                # Check if parameter appears in loss computation path
                if hasattr(sample_mu_A_param, 'grad_fn') and sample_mu_A_param.grad_fn is not None:
                    self.log(f"     WARNING: mu_A parameter has grad_fn (not a leaf node)")
                    self.log(f"     This suggests it might be created through operations that break gradients")
                elif not sample_mu_A_param.is_leaf:
                    self.log(f"     WARNING: mu_A parameter is not a leaf node in computation graph")
                    self.log(f"     This could prevent gradient flow")
                else:
                    self.log(f"     GOOD: mu_A parameter is properly connected as leaf node")
                
                # Check if parameter retains gradients
                self.log(f"     retains_grad: {sample_mu_A_param.retains_grad if hasattr(sample_mu_A_param, 'retains_grad') else 'N/A'}")
            else:
                self.log(f"     ERROR: No mu_A parameters found for graph connectivity check")
            
            # FOURTH: Check forward pass usage pattern by inspecting ProbLoRA layers
            self.log(f"\n   FORWARD PASS USAGE PATTERN CHECK:")
            prob_lora_layers_found = 0
            
            for name, module in self.model.named_modules():
                if hasattr(module, 'mu_A') and hasattr(module, 'deterministic'):
                    prob_lora_layers_found += 1
                    self.log(f"     ProbLoRA layer: {name}")
                    self.log(f"       deterministic: {module.deterministic}")
                    self.log(f"       has mu_A: {hasattr(module, 'mu_A')}")
                    self.log(f"       has A: {hasattr(module, 'A')}")
                    self.log(f"       mu_A requires_grad: {module.mu_A.requires_grad if hasattr(module, 'mu_A') else 'N/A'}")
                    
                    if prob_lora_layers_found <= 2:  # Only show first 2 for brevity
                        # Check if mu_A is used in forward pass correctly
                        if hasattr(module, 'mu_A'):
                            self.log(f"       mu_A shape: {module.mu_A.shape}")
                            self.log(f"       mu_A device: {module.mu_A.device}")
                            self.log(f"       mu_A dtype: {module.mu_A.dtype}")
                    
            self.log(f"     Total ProbLoRA layers with mu_A: {prob_lora_layers_found}")
            
            # Analyze gradient distribution with detailed breakdown
            trainable_with_grads = 0
            trainable_without_grads = 0
            frozen_with_grads = 0
            frozen_without_grads = 0
            
            trainable_grad_norm = 0.0
            frozen_grad_norm = 0.0
            
            # Detailed LoRA matrix and attention module analysis
            matrix_grad_analysis = {
                'A': {'with_grads': 0, 'without_grads': 0, 'examples': []},
                'mu_A': {'with_grads': 0, 'without_grads': 0, 'examples': []},
                'B': {'with_grads': 0, 'without_grads': 0, 'examples': []},
                'G': {'with_grads': 0, 'without_grads': 0, 'examples': []}
            }
            
            attention_module_analysis = {
                'q_proj': {'A': 0, 'mu_A': 0, 'B': 0, 'G': 0, 'total_with_grads': 0, 'total_params': 0},
                'k_proj': {'A': 0, 'mu_A': 0, 'B': 0, 'G': 0, 'total_with_grads': 0, 'total_params': 0},
                'v_proj': {'A': 0, 'mu_A': 0, 'B': 0, 'G': 0, 'total_with_grads': 0, 'total_params': 0},
            }
            
            for name, param in self.model.named_parameters():
                has_gradient = param.grad is not None and param.grad.abs().sum() > 0 if param.grad is not None else False
                
                if param.requires_grad:
                    if has_gradient:
                        trainable_with_grads += 1
                        trainable_grad_norm += param.grad.abs().sum().item()
                    else:
                        trainable_without_grads += 1
                    
                    # Analyze LoRA matrix types
                    for matrix_type in ['A', 'mu_A', 'B', 'G']:
                        if f'.{matrix_type}' in name:
                            if has_gradient:
                                matrix_grad_analysis[matrix_type]['with_grads'] += 1
                                if len(matrix_grad_analysis[matrix_type]['examples']) < 3:
                                    matrix_grad_analysis[matrix_type]['examples'].append(name)
                            else:
                                matrix_grad_analysis[matrix_type]['without_grads'] += 1
                                if len(matrix_grad_analysis[matrix_type]['examples']) < 3:
                                    matrix_grad_analysis[matrix_type]['examples'].append(f"{name} (no grad)")
                            break
                    
                    # Analyze attention module types
                    for attn_type in ['q_proj', 'k_proj', 'v_proj']:
                        if f'.{attn_type}.' in name:
                            attention_module_analysis[attn_type]['total_params'] += 1
                            if has_gradient:
                                attention_module_analysis[attn_type]['total_with_grads'] += 1
                            
                            # Track matrix type within attention module
                            for matrix_type in ['A', 'mu_A', 'B', 'G']:
                                if f'.{matrix_type}' in name:
                                    if has_gradient:
                                        attention_module_analysis[attn_type][matrix_type] += 1
                                    break
                            break
                else:
                    if has_gradient:
                        frozen_with_grads += 1
                        frozen_grad_norm += param.grad.abs().sum().item()
                    else:
                        frozen_without_grads += 1
            
            self.log(f"\n   GRADIENT ANALYSIS:")
            self.log(f"     Trainable params with gradients: {trainable_with_grads}")
            self.log(f"     Trainable params without gradients: {trainable_without_grads}")
            self.log(f"     Frozen params with gradients: {frozen_with_grads}")
            self.log(f"     Frozen params without gradients: {frozen_without_grads}")
            self.log(f"     Total trainable gradient norm: {trainable_grad_norm:.6e}")
            self.log(f"     Total frozen gradient norm: {frozen_grad_norm:.6e}")
            
            # Detailed LoRA matrix analysis
            self.log(f"\n   LORA MATRIX GRADIENT BREAKDOWN:")
            for matrix_type, stats in matrix_grad_analysis.items():
                total = stats['with_grads'] + stats['without_grads']
                if total > 0:
                    self.log(f"     {matrix_type} matrices: {stats['with_grads']}/{total} have gradients ({stats['with_grads']/total*100:.1f}%)")
                    if stats['examples']:
                        self.log(f"       Examples: {', '.join(stats['examples'][:2])}")
            
            # Attention module analysis
            self.log(f"\n   ATTENTION MODULE GRADIENT BREAKDOWN:")
            for attn_type, stats in attention_module_analysis.items():
                if stats['total_params'] > 0:
                    self.log(f"     {attn_type}: {stats['total_with_grads']}/{stats['total_params']} params have gradients")
                    matrix_breakdown = []
                    for matrix_type in ['A', 'mu_A', 'B', 'G']:
                        if stats[matrix_type] > 0:
                            matrix_breakdown.append(f"{matrix_type}:{stats[matrix_type]}")
                    if matrix_breakdown:
                        self.log(f"       Matrix breakdown: {', '.join(matrix_breakdown)}")
            
            # Calculate layer statistics
            total_layers = max([
                attention_module_analysis['q_proj']['total_params'],
                attention_module_analysis['k_proj']['total_params'], 
                attention_module_analysis['v_proj']['total_params']
            ])
            
            # Verify your calculation: 32 layers × 3 attention types = 192 parameters
            expected_attention_params = 32 * 3 * 2  # 32 layers × (q_proj, k_proj, v_proj)
            actual_attention_params = sum(stats['total_params'] for name, stats in attention_module_analysis.items() 
                                        if name in ['q_proj', 'k_proj', 'v_proj'])
            
            self.log(f"\n   USER'S CALCULATION VERIFICATION:")
            self.log(f"     Expected attention parameters (32 layers × 3 types X 2 A&B): {expected_attention_params}")
            self.log(f"     Actual q/k/v parameters found: {actual_attention_params}")
            self.log(f"     Parameters with gradients: {trainable_with_grads}")
            self.log(f"     Your hypothesis: {'VERIFIED' if trainable_with_grads == expected_attention_params else 'NEEDS INVESTIGATION'}")
            
            if trainable_with_grads == expected_attention_params:
                self.log(f"     Analysis: Either A or B matrices get gradients (not both simultaneously)")
            elif trainable_with_grads == expected_attention_params * 2:
                self.log(f"     Analysis: Both A and B matrices get gradients simultaneously")
            else:
                self.log(f"     Analysis: Gradient pattern differs from expected A/B distribution")
            
            # Validation checks
            if frozen_with_grads == 0:
                self.log(f"     ✓ PASS: No frozen parameters received gradients")
            else:
                self.log(f"     ✗ FAIL: {frozen_with_grads} frozen parameters received gradients!")
            
            if trainable_with_grads > 0:
                self.log(f"     ✓ PASS: {trainable_with_grads} trainable parameters received gradients")
            else:
                self.log(f"     ✗ FAIL: No trainable parameters received gradients!")
            
            # Take snapshot of trainable parameters before optimization
            self.log(f"\n   PARAMETER UPDATE VALIDATION:")
            param_snapshot = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param_snapshot[name] = param.detach().clone()
            
            self.log(f"     Snapshot taken of {len(param_snapshot)} trainable parameters")
            
            # Perform optimizer step
            trainer.optimizer.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            
            self.log(f"     Optimizer step completed")
            
            # SECOND BACKWARD PASS: Test gradient flow after B matrices are no longer zero
            self.log(f"\n   SECOND BACKWARD PASS ANALYSIS (After B ≠ 0):")
            self.log(f"     Testing hypothesis: Now that B matrices are non-zero, mu_A should receive gradients")
            
            # Clear any existing gradients
            self.model.zero_grad(set_to_none=True)
            
            # Second forward pass with same input
            outputs_2 = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss_2 = outputs_2.loss
            
            self.log(f"     Second forward pass loss: {loss_2.item():.6f}")
            
            # Second backward pass
            loss_2.backward()
            
            self.log(f"     Second backward pass completed")
            
            # Analyze gradient distribution for second pass
            trainable_with_grads_2 = 0
            trainable_without_grads_2 = 0
            
            mu_A_with_grads_2 = 0
            mu_A_without_grads_2 = 0
            B_with_grads_2 = 0
            B_without_grads_2 = 0
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    has_gradient = param.grad is not None and param.grad.abs().sum() > 0 if param.grad is not None else False
                    
                    if has_gradient:
                        trainable_with_grads_2 += 1
                    else:
                        trainable_without_grads_2 += 1
                    
                    # Track mu_A and B specifically
                    if '.mu_A' in name:
                        if has_gradient:
                            mu_A_with_grads_2 += 1
                        else:
                            mu_A_without_grads_2 += 1
                            self.log(f"       mu_A still without gradients: {name}")
                    elif '.B' in name and any(attn in name for attn in ['q_proj', 'k_proj', 'v_proj']):
                        if has_gradient:
                            B_with_grads_2 += 1
                        else:
                            B_without_grads_2 += 1
                            self.log(f"       B without gradients: {name}")
            
            self.log(f"     Second pass - Trainable with gradients: {trainable_with_grads_2}")
            self.log(f"     Second pass - mu_A with gradients: {mu_A_with_grads_2}/{mu_A_with_grads_2 + mu_A_without_grads_2}")
            self.log(f"     Second pass - B with gradients: {B_with_grads_2}/{B_with_grads_2 + B_without_grads_2}")
            
            # Compare first vs second pass
            mu_A_improvement = mu_A_with_grads_2 - matrix_grad_analysis['mu_A']['with_grads']
            B_change = B_with_grads_2 - matrix_grad_analysis['B']['with_grads']
            
            self.log(f"\n     GRADIENT FLOW IMPROVEMENT ANALYSIS:")
            self.log(f"     mu_A gradients: {matrix_grad_analysis['mu_A']['with_grads']} → {mu_A_with_grads_2} (Δ+{mu_A_improvement})")
            self.log(f"     B gradients: {matrix_grad_analysis['B']['with_grads']} → {B_with_grads_2} (Δ{B_change:+d})")
            
            if mu_A_improvement > 0:
                self.log(f"     ✓ HYPOTHESIS CONFIRMED: mu_A now receives gradients after B ≠ 0!")
                self.log(f"     ✓ This validates the mathematical analysis:")
                self.log(f"       - Step 0: ΔW = mu_A @ B^T = mu_A @ 0 = 0 → ∂L/∂mu_A = 0")
                self.log(f"       - Step 1+: ΔW = mu_A @ B^T ≠ 0 → ∂L/∂mu_A ≠ 0")
            elif mu_A_with_grads_2 == 0:
                self.log(f"     ✗ HYPOTHESIS NOT CONFIRMED: mu_A still receives no gradients")
                self.log(f"     This suggests a deeper issue with gradient flow or parameter setup")
            else:
                self.log(f"     ⚠ PARTIAL CONFIRMATION: Some mu_A parameters now have gradients")
            
            # Check if B matrices are indeed non-zero after optimizer step
            self.log(f"\n     B MATRIX NON-ZERO VERIFICATION:")
            B_nonzero_count = 0
            B_total_count = 0
            B_max_abs_value = 0.0
            
            for name, param in self.model.named_parameters():
                if '.B' in name and any(attn in name for attn in ['q_proj', 'k_proj', 'v_proj']):
                    B_total_count += 1
                    max_abs_val = param.abs().max().item()
                    B_max_abs_value = max(B_max_abs_value, max_abs_val)
                    
                    if max_abs_val > 1e-10:  # Non-zero threshold
                        B_nonzero_count += 1
                    else:
                        self.log(f"       B matrix still zero: {name} (max |val|: {max_abs_val:.2e})")
            
            self.log(f"     B matrices that are non-zero: {B_nonzero_count}/{B_total_count}")
            self.log(f"     Maximum |B| value across all B matrices: {B_max_abs_value:.6e}")
            
            if B_nonzero_count == 0:
                self.log(f"     ⚠ WARNING: All B matrices are still zero after optimizer step!")
                self.log(f"     This could explain why mu_A doesn't receive gradients in second pass")
            elif B_nonzero_count < B_total_count:
                self.log(f"     ⚠ PARTIAL: Only {B_nonzero_count}/{B_total_count} B matrices are non-zero")
            else:
                self.log(f"     ✓ GOOD: All B matrices are now non-zero")
            
            # Store second pass results for final summary
            second_pass_results = {
                'mu_A_with_grads': mu_A_with_grads_2,
                'B_with_grads': B_with_grads_2,
                'mu_A_improvement': mu_A_improvement,
                'B_nonzero_count': B_nonzero_count,
                'B_total_count': B_total_count,
                'hypothesis_confirmed': mu_A_improvement > 0
            }
            
            # Measure parameter changes
            total_param_change = 0.0
            params_that_changed = 0
            params_that_didnt_change = 0
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in param_snapshot:
                    delta = (param.detach() - param_snapshot[name]).abs().sum().item()
                    total_param_change += delta
                    
                    if delta > 1e-10:  # Small threshold for numerical precision
                        params_that_changed += 1
                    else:
                        params_that_didnt_change += 1
            
            self.log(f"     Parameters that changed: {params_that_changed}")
            self.log(f"     Parameters that didn't change: {params_that_didnt_change}")
            self.log(f"     Total parameter change magnitude: {total_param_change:.6e}")
            
            # Final validation
            if total_param_change > 0:
                self.log(f"     ✓ PASS: Trainable parameters were updated (total |Δ|: {total_param_change:.3e})")
            else:
                self.log(f"     ✗ FAIL: No parameter updates detected!")
            
            # Verify frozen parameters didn't change
            frozen_params_changed = 0
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    # Frozen parameters should never change, so we don't need snapshots
                    # But we can check if they somehow got gradients and warn
                    if param.grad is not None:
                        frozen_params_changed += 1
            
            if frozen_params_changed == 0:
                self.log(f"     ✓ PASS: All frozen parameters remained unchanged")
            else:
                self.log(f"     ✗ WARNING: {frozen_params_changed} frozen parameters had gradients")
            
            # Summary
            self.log(f"\n   GRADIENT FLOW VALIDATION SUMMARY:")
            gradient_flow_correct = (frozen_with_grads == 0) and (trainable_with_grads > 0)
            parameter_updates_correct = (total_param_change > 0) and (frozen_params_changed == 0)
            
            if gradient_flow_correct:
                self.log(f"     ✓ Gradient flow: CORRECT (only trainable params receive gradients)")
            else:
                self.log(f"     ✗ Gradient flow: INCORRECT")
                
            if parameter_updates_correct:
                self.log(f"     ✓ Parameter updates: CORRECT (only trainable params change)")
            else:
                self.log(f"     ✗ Parameter updates: INCORRECT")
            
            # Include second pass analysis in summary
            if 'second_pass_results' in locals():
                self.log(f"\n   MATHEMATICAL HYPOTHESIS VALIDATION:")
                if second_pass_results['hypothesis_confirmed']:
                    self.log(f"     ✓ HYPOTHESIS CONFIRMED: mu_A gradients appear after B ≠ 0")
                    self.log(f"       - Step 0: {matrix_grad_analysis['mu_A']['with_grads']} mu_A params had gradients")
                    self.log(f"       - Step 1: {second_pass_results['mu_A_with_grads']} mu_A params have gradients")
                    self.log(f"       - Improvement: +{second_pass_results['mu_A_improvement']} mu_A params now receive gradients")
                    overall_pass = gradient_flow_correct and parameter_updates_correct and True
                else:
                    self.log(f"     ✗ HYPOTHESIS NOT CONFIRMED: mu_A gradients still missing")
                    self.log(f"       - This suggests issues beyond the B=0 initialization problem")
                    overall_pass = False
                
                self.log(f"     B matrix status: {second_pass_results['B_nonzero_count']}/{second_pass_results['B_total_count']} non-zero after optimizer step")
            else:
                overall_pass = gradient_flow_correct and parameter_updates_correct
            
            if overall_pass:
                self.log(f"     OVERALL: PASS - Perfect LoRA parameter handling!")
            else:
                self.log(f"     OVERALL: FAIL - LoRA parameter handling has issues")
            
            # Prepare return results
            results = {
                'gradient_flow_correct': gradient_flow_correct,
                'parameter_updates_correct': parameter_updates_correct,
                'overall_pass': overall_pass,
                'trainable_with_grads': trainable_with_grads,
                'frozen_with_grads': frozen_with_grads,
                'total_param_change': total_param_change,
                'params_that_changed': params_that_changed
            }
            
            # Add second pass results if available
            if 'second_pass_results' in locals():
                results['second_pass'] = second_pass_results
            
            return results
            
        except Exception as e:
            self.log(f"   ERROR during gradient flow validation: {e}")
            traceback.print_exc()
            return None


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load and parse configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract defaults section
    defaults = config.get('defaults')
    
    # Extract model configuration
    model_name = defaults['model_name']
    model_config = config.get('models').get(model_name)
    
    # Merge defaults with model-specific config
    merged_config = {
        # Model settings
        'model_name': model_config.get('model_name_or_path') or defaults['model_name_or_path'],
        'tokenizer_name': model_config.get('tokenizer_name') or defaults['tokenizer_name'],
        'load_in_4bit': model_config.get('load_in_4bit') or defaults['load_in_4bit'],
        'attn_implementation': defaults['attn_implementation'],
        
        # Training parameters
        'learning_rate': defaults['learning_rate'],
        'lr_scheduler_type': defaults['lr_scheduler_type'],
        'warmup_ratio': defaults['warmup_ratio'],
        'weight_decay': defaults['weight_decay'],
        'optim': defaults['optim'],
        'max_grad_norm': defaults['max_grad_norm'],
        
        # Training strategy
        'batch_size': defaults['batch_size'],
        'gradient_accumulation_steps': defaults['gradient_accumulation_steps'],
        'train_epochs': defaults['train_epochs'],
        'save_strategy': defaults['save_strategy'],
        'eval_strategy': defaults['eval_strategy'],
        'logging_steps': defaults['logging_steps'],
        
        # Precision settings
        'bf16': defaults['bf16'],
        'fp16': defaults['fp16'],
        
        # Memory optimization
        'gradient_checkpointing': defaults['gradient_checkpointing'],
        'use_cache': defaults['use_cache'],
        'dataloader_num_workers': defaults['dataloader_num_workers'],
        'dataloader_pin_memory': defaults['dataloader_pin_memory'],
        'pad_to_multiple_of': defaults['pad_to_multiple_of'],
        
        # LoRA configuration
        'rank': defaults['rank'],
        'lora_alpha': defaults['lora_alpha'],
        'target_modules': defaults['target_attention_layers'],
        'enable_clamps': defaults['enable_clamps'],
        'deterministic_lora': defaults['deterministic_lora'],
        
        # ARD configuration
        'kl_loss_beta': defaults['kl_loss_beta'],
        'ard_prior_samples': defaults['ard_prior_samples'],
        'max_len': defaults['max_len'],
        
        # Numerical stability parameters for ProbLoRA
        'logvar_clamp_min': defaults['logvar_clamp_min'],
        'logvar_clamp_max': defaults['logvar_clamp_max'],
        'beta_logvar_clamp_min': defaults['beta_logvar_clamp_min'],
        'beta_logvar_clamp_max': defaults['beta_logvar_clamp_max'],
        'sample_clamp_min': defaults['sample_clamp_min'],
        'sample_clamp_max': defaults['sample_clamp_max'],
        
        # Dataset configuration
        'dataset_name': defaults['dataset_name'],
        'dataset_name_specific': defaults['dataset_name_specific'],
        'random_seed': defaults['random_seed'],
        
        # Reporting
        'report_to': defaults['report_to'],
        'output_dir': f"output/{defaults['dataset_name_specific']}_run_{defaults['runId']}"
    }
    
    return merged_config


def load_model_for_analysis(config_path: str, model_args: Dict[str, Any]) -> torch.nn.Module:
    """Load model with ProbLoRA injection for analysis focusing only on ProbLoRA parameters."""
    
    print("Loading model for ProbLoRA-focused parameter analysis...")
    
    # Prepare model loading arguments
    model_kwargs = {
        'torch_dtype': torch.bfloat16 if model_args['bf16'] else torch.float16,
        'device_map': "cpu",  # Keep on CPU for analysis
        'load_in_4bit': False,  # No quantization for parameter counting
        'trust_remote_code': True
    }
    
    # Load base model using Auto class for flexibility
    model = AutoModelForCausalLM.from_pretrained(
        model_args['model_name'],
        **model_kwargs
    )
    
    # Inject ProbLoRA using the exact same function and parameters as training
    print("Injecting ProbLoRA layers with full configuration...")
    inject_problora_llama(
        model=model,
        rank=model_args['rank'],
        scaling=model_args['lora_alpha'] / model_args['rank'],
        num_tokens=model_args['max_len'],
        ard_prior_samples=model_args['ard_prior_samples'],
        logvar_clamp_min=model_args['logvar_clamp_min'],
        logvar_clamp_max=model_args['logvar_clamp_max'],
        beta_logvar_clamp_min=model_args['beta_logvar_clamp_min'],
        beta_logvar_clamp_max=model_args['beta_logvar_clamp_max'],
        sample_clamp_min=model_args['sample_clamp_min'],
        sample_clamp_max=model_args['sample_clamp_max'],
        attn_implementation=model_args['attn_implementation'],
        target_attention_layers=model_args['target_modules'],
        deterministic=model_args['deterministic_lora'],
        enable_clamps=model_args['enable_clamps'],
        lora_alpha=model_args['lora_alpha']
    )
    
    # Use EXACT same parameter handling as training script
    print("Applying exact training script parameter handling...")
    verbose = True  # Enable verbose output for analysis
    
    # Freeze base parameters and unfreeze LoRA parameters (EXACT COPY FROM TRAINING SCRIPT)
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
    
    # Count parameters focusing only on trainable ones (ProbLoRA matrices)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    problora_params = sum(p.numel() for name, p in model.named_parameters() 
                         if p.requires_grad and any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A']))
    
    print(f"\nParameter analysis focus: ProbLoRA layers only")
    print(f"  Total model parameters: {total_params:,}")
    print(f"  Total trainable parameters: {trainable_params:,}")
    print(f"  ProbLoRA trainable parameters: {problora_params:,}")
    print(f"  Trainable parameter groups: {trainable_count}")
    print(f"  Quantized parameters skipped: {quantized_params_skipped}")
    
    if trainable_params > 0:
        print(f"  ProbLoRA percentage of trainable: {100*problora_params/trainable_params:.1f}%")
        print(f"  Trainable percentage of total: {100*trainable_params/total_params:.4f}%")
    
    # Verify that only ProbLoRA parameters are trainable (as expected)
    non_problora_trainable = [name for name, p in model.named_parameters() 
                             if p.requires_grad and not any(matrix in name for matrix in ['.A', '.B', '.G', '.mu_A'])]
    
    if non_problora_trainable:
        print(f"  WARNING: Found {len(non_problora_trainable)} non-ProbLoRA trainable parameters")
        for name in non_problora_trainable[:5]:  # Show first 5
            print(f"    - {name}")
        if len(non_problora_trainable) > 5:
            print(f"    ... and {len(non_problora_trainable) - 5} more")
    else:
        print(f"  VERIFIED: Only ProbLoRA parameters (A, B, G, mu_A matrices) are trainable")
    
    # Final verification using training script method
    if trainable_count == 0:
        print("\n[ERROR] No ProbLoRA parameters found! Debugging information:")
        problora_modules = []
        for mod_name, mod in model.named_modules():
            if isinstance(mod, ProbLoRALayer):
                problora_modules.append(mod_name)
        print(f"[DEBUG] ProbLoRALayer modules found: {len(problora_modules)}")
        if quantized_params_skipped > 0:
            print(f"[DIAGNOSIS] All ProbLoRA parameters appear to be quantized.")
            print(f"[SOLUTION] Consider disabling quantization: load_in_4bit: false")
        raise RuntimeError("No trainable ProbLoRA parameters found! Check ProbLoRA injection and parameter types.")
    
    print("Model loaded - analysis will use EXACT training script parameter handling")
    return model


def create_test_trainer_for_validation(model, config_path: str = "config/run_training_params.yaml"):
    """Create a test trainer to validate optimizer parameter usage."""
    try:
        
        print("Creating test trainer for optimizer validation...")
        
        # Resolve config path relative to project root (scripts is subdirectory)
        script_dir = Path(__file__).parent  # scripts directory
        project_root = script_dir.parent    # ARD-LoRA-Data-CLM directory
        
        # If config_path is relative, resolve it from project root
        if not Path(config_path).is_absolute():
            resolved_config_path = project_root / config_path
        else:
            resolved_config_path = Path(config_path)
            
        print(f"[DEBUG] Script dir: {script_dir}")
        print(f"[DEBUG] Project root: {project_root}")
        print(f"[DEBUG] Config path: {config_path}")
        print(f"[DEBUG] Resolved config path: {resolved_config_path}")
        
        # Load config to get training parameters (required)
        if not resolved_config_path.exists():
            print(f"ERROR: Configuration file not found: {resolved_config_path}")
            print("Cannot create test trainer without training configuration.")
            return None
            
        with open(resolved_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"[DEBUG] Config keys: {list(config.keys())}")
        
        if 'defaults' not in config:
            print(f"ERROR: 'defaults' section not found in config file")
            print(f"Available sections: {list(config.keys())}")
            return None
            
        defaults = config['defaults']
        print(f"[DEBUG] Defaults keys: {list(defaults.keys())}")
        
        # Extract training parameters from YAML (no fallbacks)
        try:
            learning_rate = defaults['learning_rate']
            optim = defaults['optim']
            weight_decay = defaults['weight_decay']
            max_grad_norm = defaults['max_grad_norm']
            batch_size = defaults['batch_size']
            model_name = defaults['model_name_or_path']
        except KeyError as e:
            print(f"ERROR: Missing required parameter in config defaults: {e}")
            print(f"Available parameters: {list(defaults.keys())}")
            return None
        
        # Validate required parameters
        if any(param is None for param in [learning_rate, optim, model_name]):
            print("ERROR: Missing required training parameters in config file")
            print(f"Required: learning_rate, optim, model_name_or_path")
            print(f"Values: lr={learning_rate}, optim={optim}, model={model_name}")
            return None
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create training arguments from config
        training_args = TrainingArguments(
            output_dir="temp_output",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=1,
            optim=optim,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            save_strategy="no",
            logging_steps=100,
            remove_unused_columns=False  # Important for causal LM
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        print("Test trainer created successfully")
        print(f"Optimizer: {optim}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Weight Decay: {weight_decay}")
        print(f"Max Grad Norm: {max_grad_norm}")
        
        return trainer
        
    except Exception as e:
        print(f"Warning: Could not create test trainer: {e}")
        traceback.print_exc()
        return None


def analyze_with_optimizer_validation(model, config_path: str = "config/run_training_params.yaml", output_file: str = None):
    """Analyze model parameters and validate optimizer usage with comprehensive gradient flow testing."""
    
    # Create parameter analyzer
    analyzer = ParameterAnalyzer(model, output_file)
    
    # Run basic analysis
    results = analyzer.analyze_model()
    
    # Create test trainer for optimizer validation
    trainer = create_test_trainer_for_validation(model, config_path)
    
    # Validate optimizer parameters
    if trainer is not None:
        optimizer_results = analyzer.validate_optimizer_parameters(trainer=trainer)
        results['optimizer_validation'] = optimizer_results
        
        # NEW: Validate gradient flow and parameter updates (the ultimate test!)
        gradient_flow_results = analyzer.validate_gradient_flow_and_updates(trainer=trainer)
        results['gradient_flow_validation'] = gradient_flow_results
        
    else:
        print("Skipping optimizer and gradient flow validation due to trainer creation failure")
        results['optimizer_validation'] = None
        results['gradient_flow_validation'] = None
    
    # Save log
    analyzer.save_log()
    
    return results, analyzer


def quick_analysis():
    """Run quick parameter analysis with configuration from YAML file."""
    
    # Load configuration from YAML
    config_path = "config/run_training_params.yaml"
    config_path_obj = Path(config_path)
    
    if not config_path_obj.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        print("The analysis requires the exact same configuration used for training.")
        print("Please ensure the config/run_training_params.yaml file exists.")
        return None
    
    # Load configuration from YAML (required)
    config_args = load_config_from_yaml(config_path)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"parameter_analysis_{timestamp}.txt"
    
    print("Quick Parameter Analysis")
    print("=" * 50)
    print(f"Config File: {config_path}")
    print(f"Model: {config_args['model_name']}")
    print(f"LoRA Rank: {config_args['rank']}")
    print(f"LoRA Alpha: {config_args['lora_alpha']}")
    print(f"Target Modules: {config_args['target_modules']}")
    print(f"Deterministic LoRA: {config_args['deterministic_lora']}")
    print(f"Optimizer: {config_args['optim']}")
    print(f"Learning Rate: {config_args['learning_rate']}")
    print(f"Output: {output_file}")
    print()
    
    try:
        # Load model (this will take some time)
        print("Loading model (this may take a few minutes)...")
        model = load_model_for_analysis(config_path, config_args)
        
        # Analyze parameters with optimizer validation
        print("Analyzing parameters and validating optimizer...")
        results, analyzer = analyze_with_optimizer_validation(model, config_path, str(output_file))
        
        # Print summary
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"Total Parameters: {results['total_parameters']:,}")
        print(f"Trainable Parameters: {results['total_trainable']:,}")
        print(f"Trainable Percentage: {(results['total_trainable']/results['total_parameters'])*100:.2f}%")
        print(f"ProbLoRA Layers: {results['problora_layers']}")
        print(f"ProbLoRA Trainable: {results['problora_trainable']:,}")
        print()
        print(f"Detailed report: {output_file}")
        print(f"Target validation: 12,582,912 expected")
        
        # Validation status
        expected = 12_582_912
        actual = results['total_trainable']
        if actual == expected:
            print("Parameter count matches expected value!")
        else:
            diff = abs(actual - expected)
            print(f"Difference: {diff:,} parameters ({diff/expected*100:.2f}%)")
        
        # Optimizer validation summary
        if 'optimizer_validation' in results and results['optimizer_validation'] is not None:
            opt_results = results['optimizer_validation']
            print(f"\nOptimizer Validation:")
            print(f"  Model trainable parameters: {opt_results['model_param_count']:,}")
            print(f"  ProbLoRA parameters found: {len(opt_results['problora_param_names'])}")
            print(f"  Non-ProbLoRA parameters: {len(opt_results['non_problora_param_names'])}")
        elif 'optimizer_validation' in results:
            print(f"\nOptimizer Validation: Skipped (trainer creation failed)")
        else:
            print(f"\nOptimizer Validation: Not performed")
        
        # Gradient flow validation summary
        if 'gradient_flow_validation' in results and results['gradient_flow_validation'] is not None:
            gf_results = results['gradient_flow_validation']
            print(f"\nGradient Flow Validation:")
            print(f"  Trainable params with gradients: {gf_results['trainable_with_grads']}")
            print(f"  Frozen params with gradients: {gf_results['frozen_with_grads']}")
            print(f"  Parameters that updated: {gf_results['params_that_changed']}")
            print(f"  Total parameter change: {gf_results['total_param_change']:.3e}")
            
            # Check for second pass results
            if 'second_pass' in gf_results:
                sp_results = gf_results['second_pass']
                print(f"\nSecond Pass Analysis (After B ≠ 0):")
                print(f"  mu_A gradient improvement: +{sp_results['mu_A_improvement']}")
                print(f"  B matrices non-zero: {sp_results['B_nonzero_count']}/{sp_results['B_total_count']}")
                if sp_results['hypothesis_confirmed']:
                    print(f"  ✓ Hypothesis confirmed: mu_A gradients appear after B becomes non-zero!")
                else:
                    print(f"  ✗ Hypothesis not confirmed: mu_A gradients still missing")
            
            if gf_results['overall_pass']:
                print(f"  Result: PERFECT LoRA parameter handling!")
            else:
                print(f"  Result: Issues detected in LoRA parameter handling")
        elif 'gradient_flow_validation' in results:
            print(f"\nGradient Flow Validation: Skipped (trainer creation failed)")
        else:
            print(f"\nGradient Flow Validation: Not performed")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()
        return None


def analyze_existing_model(model, output_file: str = None):
    """Analyze an already loaded model."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output/parameter_analysis_{timestamp}.txt"
    
    analyzer = ParameterAnalyzer(model, output_file)
    results = analyzer.analyze_model()
    analyzer.save_log()
    
    return results


def main():
    """Main function for parameter analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze trainable parameters in ARD-LoRA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick analysis with defaults
  python scripts/analyze_trainable_parameters.py --quick
  
  # Analyze with custom config
  python scripts/analyze_trainable_parameters.py --config config/run_training_params.yaml
  
  # Analyze with custom output file
  python scripts/analyze_trainable_parameters.py --output my_analysis.txt
  
  # Override model name
  python scripts/analyze_trainable_parameters.py --model-name meta-llama/Llama-2-7b-hf
        """
    )
    parser.add_argument("--config", type=str, default="config/run_training_params.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for parameter analysis (optional)")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Override model name from config")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick analysis with default configuration")
    
    args = parser.parse_args()
    
    # Run quick analysis if requested
    if args.quick:
        return quick_analysis()
    
    # Load configuration from YAML
    config_args = load_config_from_yaml(args.config)
    
    # Override model name if provided
    if args.model_name:
        config_args['model_name'] = args.model_name
    
    # Set default output file if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"output/parameter_analysis_{timestamp}.txt"
    
    try:
        # Load model with YAML configuration
        model = load_model_for_analysis(args.config, config_args)
        
        # Analyze parameters with optimizer validation
        results, analyzer = analyze_with_optimizer_validation(model, args.config, args.output)
        
        print(f"\nAnalysis complete!")
        print(f"Found {results['total_trainable']:,} trainable parameters")
        print(f"Analyzed {results['problora_layers']} ProbLoRA layers")
        
        if args.output:
            print(f"Detailed report saved to: {args.output}")
        
        # Print optimizer validation summary
        if 'optimizer_validation' in results:
            opt_results = results['optimizer_validation']
            print(f"\nOptimizer Validation:")
            print(f"  Model trainable parameters: {opt_results['model_param_count']:,}")
            print(f"  ProbLoRA parameters found: {len(opt_results['problora_param_names'])}")
            print(f"  Non-ProbLoRA parameters: {len(opt_results['non_problora_param_names'])}")
        
        # Print gradient flow validation summary
        if 'gradient_flow_validation' in results and results['gradient_flow_validation'] is not None:
            gf_results = results['gradient_flow_validation']
            print(f"\nGradient Flow Validation:")
            print(f"  Gradient flow: {'✓ CORRECT' if gf_results['gradient_flow_correct'] else '✗ INCORRECT'}")
            print(f"  Parameter updates: {'✓ CORRECT' if gf_results['parameter_updates_correct'] else '✗ INCORRECT'}")
            print(f"  Overall result: {'PERFECT' if gf_results['overall_pass'] else 'ISSUES DETECTED'}")
        else:
            print(f"\nGradient Flow Validation: Not performed")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # If no arguments provided, show usage and run quick analysis
    if len(sys.argv) == 1:
        print("ARD-LoRA Parameter Analysis Tool")
        print("=" * 50)
        print("This tool provides comprehensive analysis of trainable parameters in ARD-LoRA models.")
        print()
        print("Usage Options:")
        print("  --quick                     : Run quick analysis with defaults")
        print("  --config <file>            : Use specific config file")
        print("  --output <file>            : Specify output file")
        print("  --model-name <name>        : Override model name")
        print()
        print("Running quick analysis with default settings...")
        print()
        quick_analysis()
    else:
        main()