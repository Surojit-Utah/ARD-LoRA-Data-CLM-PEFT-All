"""
Dataset Utility Functions
==========================

Utility functions for dataset analysis and statistics.
"""

import torch
from collections import Counter


def analyze_class_distribution(dataset, dataset_name, num_classes):
    """
    Analyze and print class distribution statistics for a classification dataset.
    
    Args:
        dataset: PyTorch dataset with 'classes' or 'labels' field
        dataset_name: Name of the dataset (for display purposes)
        num_classes: Expected number of classes
    
    Prints:
        - Total samples
        - Per-class counts and percentages
        - Class imbalance ratio (max/min)
        - Warning if significant imbalance detected
    """
    # Extract class labels from dataset
    class_labels = []
    for item in dataset:
        if "classes" in item:
            class_labels.append(item["classes"].item() if torch.is_tensor(item["classes"]) else item["classes"])
        elif "labels" in item:
            class_labels.append(item["labels"].item() if torch.is_tensor(item["labels"]) else item["labels"])
    
    if not class_labels:
        print(f"[CLASS DISTRIBUTION] WARNING: No class labels found in {dataset_name}")
        return
    
    # Count occurrences of each class
    class_counts = Counter(class_labels)
    total_samples = len(class_labels)
    
    print(f"\n[CLASS DISTRIBUTION] {dataset_name} Dataset:")
    print(f"  Total samples: {total_samples}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class distribution:")
    
    for class_idx in range(num_classes):
        count = class_counts.get(class_idx, 0)
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"    Class {class_idx}: {count:5d} samples ({percentage:5.2f}%)")
    
    # Check for class imbalance
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2.0:
            print(f"  ⚠️  WARNING: Significant class imbalance detected!")
