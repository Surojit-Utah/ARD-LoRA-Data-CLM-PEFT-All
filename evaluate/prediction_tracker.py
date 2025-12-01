"""
Prediction Tracker for ARD-LoRA Training
========================================

This module provides prediction tracking functionality to monitor model learning
progress by saving predictions on fixed examples across training epochs.

Key Features:
1. Selects representative examples from training and validation sets
2. Tracks prediction evolution across epochs
3. Saves human-readable prediction files
4. Provides detailed confidence analysis for multiple choice questions
"""

import os
import json
import random
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset


class PredictionTracker:
    """
    Tracks model predictions on fixed examples across training epochs.
    
    Designed specifically for ARC-Easy multiple choice questions but can be
    extended for other datasets.
    """
    
    def __init__(self, 
                 output_dir: str,
                 tokenizer,
                 n_examples: int = 10,
                 dataset_name: str = "arc_easy",
                 seed: int = 42):
        """
        Initialize prediction tracker.
        
        Args:
            output_dir: Directory to save prediction files
            tokenizer: Tokenizer used by the model
            n_examples: Number of examples to track per split
            dataset_name: Name of dataset for formatting
            seed: Random seed for reproducible example selection
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = tokenizer
        self.n_examples = n_examples
        self.dataset_name = dataset_name.lower()
        self.seed = seed
        
        # Fixed examples to track
        self.train_examples = []
        self.val_examples = []
        self.train_indices = []
        self.val_indices = []
        
        # Prediction history
        self.prediction_history = {
            'train': [],
            'val': []
        }
        
        # Load raw ARC-Easy dataset for accessing original questions/answers
        self.raw_arc_dataset = None
        if self.dataset_name == "arc_easy":
            try:
                from datasets import load_dataset
                print(f"[PredictionTracker] Loading raw ARC-Easy dataset for original question access...")
                self.raw_arc_dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")
                print(f"[PredictionTracker] Raw ARC-Easy dataset loaded successfully")
            except Exception as e:
                print(f"[PredictionTracker] WARNING: Could not load raw dataset: {e}")
        
        print(f"[PredictionTracker] Initialized with output_dir: {self.output_dir}")
    
    def select_examples(self, train_dataset, val_dataset):
        """
        Select representative examples from training and validation sets.
        
        For ARC-Easy: Ensures we have examples from different answer choices (A/B/C/D).
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        print(f"[PredictionTracker] Selecting {self.n_examples} examples from each split...")
        
        # Select training examples
        if train_dataset is not None and len(train_dataset) > 0:
            self.train_indices, self.train_examples = self._select_balanced_examples(
                train_dataset, self.n_examples, "train"
            )
        
        # Select validation examples
        if val_dataset is not None and len(val_dataset) > 0:
            self.val_indices, self.val_examples = self._select_balanced_examples(
                val_dataset, self.n_examples, "val"
            )
        
        # Save selected examples info
        self._save_selected_examples_info()
        
        print(f"[PredictionTracker] Selected {len(self.train_examples)} train + {len(self.val_examples)} val examples")
    
    def _select_balanced_examples(self, dataset, n_examples: int, split_name: str) -> Tuple[List[int], List[Dict]]:
        """
        Select balanced examples ensuring different answer choices are represented.
        
        Args:
            dataset: Dataset to select from
            n_examples: Number of examples to select
            split_name: Name of split for logging
            
        Returns:
            (indices, examples): Selected indices and example data
        """
        total_size = len(dataset)
        
        if self.dataset_name == "arc_easy":
            # For ARC-Easy: Try to get balanced answer choices
            return self._select_arc_easy_balanced(dataset, n_examples, split_name)
        else:
            # For other datasets: Random selection
            indices = random.sample(range(total_size), min(n_examples, total_size))
            examples = []
            
            for idx in indices:
                try:
                    example = dataset[idx]
                    examples.append({
                        'index': idx,
                        'input_ids': example.get('input_ids', []),
                        'labels': example.get('labels', []),
                        'attention_mask': example.get('attention_mask', [])
                    })
                except Exception as e:
                    print(f"[PredictionTracker] Error accessing {split_name}[{idx}]: {e}")
                    continue
            
            return indices, examples
    
    def _select_arc_easy_balanced(self, dataset, n_examples: int, split_name: str) -> Tuple[List[int], List[Dict]]:
        """
        Select ARC-Easy examples with balanced answer choices.
        
        Args:
            dataset: ARC-Easy dataset
            n_examples: Number of examples to select
            split_name: Name of split for logging
            
        Returns:
            (indices, examples): Selected indices and example data
        """
        # Validate n_examples
        if n_examples <= 0:
            print(f"[PredictionTracker] WARNING: n_examples={n_examples} is invalid, using default of 10")
            n_examples = 10
        
        total_size = len(dataset)
        if total_size == 0:
            print(f"[PredictionTracker] WARNING: {split_name} dataset is empty")
            return [], []
        
        # Adjust n_examples if it exceeds dataset size
        if n_examples > total_size:
            print(f"[PredictionTracker] WARNING: n_examples={n_examples} > dataset size={total_size}, using {total_size}")
            n_examples = total_size
        
        # Group examples by answer choice
        answer_groups = {'A': [], 'B': [], 'C': [], 'D': []}
        
        # Sample a subset to analyze (for large datasets)
        total_size = len(dataset)
        sample_size = min(1000, total_size)
        sample_indices = random.sample(range(total_size), sample_size)
        
        for idx in sample_indices:
            try:
                example = dataset[idx]
                
                # Extract answer choice from labels
                answer_choice = self._extract_answer_choice_from_example(example)
                
                if answer_choice in answer_groups:
                    answer_groups[answer_choice].append(idx)
                    
            except Exception as e:
                print(f"[PredictionTracker] Error analyzing {split_name}[{idx}]: {e}")
                continue
        
        # Select balanced examples
        examples_per_choice = max(1, n_examples // 4)  # Try to get ~2-3 per choice
        selected_indices = []
        
        for choice, indices in answer_groups.items():
            if indices:
                # Select random examples from this choice
                n_select = min(examples_per_choice, len(indices))
                selected = random.sample(indices, n_select)
                selected_indices.extend(selected)
                print(f"[PredictionTracker] {split_name}: Selected {n_select} examples with answer {choice}")
        
        # If we don't have enough, fill with random examples
        while len(selected_indices) < n_examples and len(selected_indices) < total_size:
            remaining_indices = [i for i in range(total_size) if i not in selected_indices]
            if remaining_indices:
                selected_indices.append(random.choice(remaining_indices))
        
        # Limit to requested number
        selected_indices = selected_indices[:n_examples]
        
        # Get the actual examples
        examples = []
        for idx in selected_indices:
            try:
                # Get the processed example from the dataset
                example = dataset[idx]
                
                # Decode text for human readability (processed prompt)
                input_ids = example.get('input_ids', [])
                text = self.tokenizer.decode(input_ids, skip_special_tokens=True) if input_ids else ""
                
                # Access raw dataset for original question/choices/answerKey
                # CRITICAL: Match by question text, not index, because datasets are filtered differently
                raw_example = None
                if self.raw_arc_dataset is not None:
                    # Determine split name
                    split = 'train' if split_name == 'train' else 'validation'
                    
                    # Extract question text from the formatted prompt to match with raw dataset
                    question_from_prompt = None
                    try:
                        # Parse the formatted prompt to extract the actual question
                        # Format is usually: "Select one... question: <QUESTION> Choices: ..."
                        if 'question:' in text.lower():
                            parts = text.split('Choices:')[0]  # Get part before choices
                            # Find the question after "question:"
                            question_start = parts.lower().rfind('question:')
                            if question_start != -1:
                                question_from_prompt = parts[question_start + len('question:'):].strip()
                        
                        # Search raw dataset for matching question by text
                        if question_from_prompt and len(question_from_prompt) > 20:
                            for raw_item in self.raw_arc_dataset[split]:
                                if raw_item['question'].strip() == question_from_prompt:
                                    raw_example = raw_item
                                    print(f"[PredictionTracker] Matched example {idx} to raw dataset by question text")
                                    break
                        
                        # If still no match, fallback to index (may be wrong due to filtering)
                        if raw_example is None:
                            try:
                                raw_example = self.raw_arc_dataset[split][idx]
                                print(f"[PredictionTracker] WARNING: Using index-based match for {split}[{idx}] - may be incorrect!")
                            except:
                                print(f"[PredictionTracker] Could not access raw dataset at {split}[{idx}]")
                    except Exception as e:
                        print(f"[PredictionTracker] Error matching raw example: {e}")
                
                # Compute the class index from answerKey
                class_idx = -1
                if raw_example is not None and 'answerKey' in raw_example:
                    answer_key = raw_example['answerKey']
                    if answer_key.isalpha():
                        class_idx = ord(answer_key) - ord("A")  # A->0, B->1, C->2, D->3
                    else:
                        class_idx = int(answer_key) - 1  # 1->0, 2->1, 3->2, 4->3
                elif 'classes' in example:
                    class_idx = example['classes']
                elif 'label' in example:
                    class_idx = example['label']
                
                # Store both raw and processed information
                example_dict = {
                    'index': idx,
                    'input_ids': input_ids,
                    'labels': example.get('labels', []),
                    'classes': class_idx,  # Use computed class index
                    'attention_mask': example.get('attention_mask', []),
                    'text': text,  # Processed prompt fed to model
                    'answer_choice': chr(ord('A') + class_idx) if 0 <= class_idx <= 3 else 'Unknown'
                }
                
                # Add raw dataset information if available
                if raw_example is not None:
                    example_dict['raw_question'] = raw_example.get('question', '')
                    example_dict['raw_choices'] = raw_example.get('choices', {})
                    example_dict['raw_answerKey'] = raw_example.get('answerKey', '')
                    example_dict['raw_id'] = raw_example.get('id', '')
                
                examples.append(example_dict)
            except Exception as e:
                print(f"[PredictionTracker] Error processing {split_name}[{idx}]: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return selected_indices, examples
    
    def _extract_answer_choice_from_example(self, example) -> str:
        """
        Extract the correct answer choice (A/B/C/D) from an example.
        
        Uses the actual class/label field that is used during training,
        NOT parsing from text.
        
        Args:
            example: Dataset example
            
        Returns:
            Answer choice letter or 'Unknown'
        """
        try:
            # Method 1: Check for 'classes' field (used by ARCEasyDataset)
            if 'classes' in example:
                label_idx = example['classes']
                if isinstance(label_idx, (int, np.integer)) and 0 <= label_idx <= 3:
                    return chr(ord('A') + label_idx)  # 0->A, 1->B, 2->C, 3->D
                elif isinstance(label_idx, torch.Tensor):
                    label_idx = label_idx.item()
                    if 0 <= label_idx <= 3:
                        return chr(ord('A') + label_idx)
            
            # Method 2: Check for 'label' field (generic datasets)
            if 'label' in example:
                label_idx = example['label']
                if isinstance(label_idx, (int, np.integer)) and 0 <= label_idx <= 3:
                    return chr(ord('A') + label_idx)
                elif isinstance(label_idx, torch.Tensor):
                    label_idx = label_idx.item()
                    if 0 <= label_idx <= 3:
                        return chr(ord('A') + label_idx)
            
            # If no label found, return Unknown
            return 'Unknown'
            
        except Exception as e:
            print(f"[PredictionTracker] Error extracting answer: {e}")
            return 'Unknown'
    
    def track_predictions(self, model, epoch: int):
        """
        Generate and save predictions for the fixed examples.
        
        Args:
            model: Trained model
            epoch: Current training epoch
        """
        print(f"[PredictionTracker] Generating predictions for epoch {epoch}...")
        
        model.eval()
        
        epoch_predictions = {
            'epoch': epoch,
            'train_predictions': [],
            'val_predictions': []
        }
        
        # Generate predictions for training examples
        if self.train_examples:
            train_preds = self._generate_predictions(model, self.train_examples, "train")
            epoch_predictions['train_predictions'] = train_preds
        
        # Generate predictions for validation examples
        if self.val_examples:
            val_preds = self._generate_predictions(model, self.val_examples, "val")
            epoch_predictions['val_predictions'] = val_preds
        
        # Save predictions to file
        self._save_epoch_predictions(epoch_predictions)
        
        # Store in history
        self.prediction_history['train'].append(train_preds)
        self.prediction_history['val'].append(val_preds)
        
        model.train()  # Restore training mode
        
        print(f"[PredictionTracker] Saved predictions for epoch {epoch}")
    
    def _generate_predictions(self, model, examples: List[Dict], split_name: str) -> List[Dict]:
        """
        Generate predictions for a list of examples.
        
        Args:
            model: Model to use for predictions
            examples: List of examples to predict on
            split_name: Name of split for logging
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for i, example in enumerate(examples):
                try:
                    # Prepare input
                    input_ids = torch.tensor([example['input_ids']], device=device)
                    attention_mask = torch.tensor([example.get('attention_mask', [1] * len(example['input_ids']))], device=device)
                    
                    # Generate prediction
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # For ARC-Easy: Extract answer choice predictions
                    prediction_info = self._extract_answer_prediction(
                        example, logits, input_ids, split_name, i
                    )
                    
                    predictions.append(prediction_info)
                    
                except Exception as e:
                    print(f"[PredictionTracker] Error predicting {split_name}[{i}]: {e}")
                    predictions.append({
                        'example_idx': i,
                        'error': str(e),
                        'predicted_answer': 'Error',
                        'confidence': 0.0
                    })
        
        return predictions
    
    def _extract_answer_prediction(self, example: Dict, logits: torch.Tensor, 
                                  input_ids: torch.Tensor, split_name: str, example_idx: int) -> Dict:
        """
        Extract answer choice prediction from model logits.
        
        Args:
            example: Original example data
            logits: Model output logits
            input_ids: Input token IDs
            split_name: Split name for logging
            example_idx: Example index
            
        Returns:
            Prediction information dictionary
        """
        try:
            # Get token probabilities at the last position (answer position)
            last_logits = logits[0, -1, :]  # Last token position
            probs = torch.softmax(last_logits, dim=-1)
            
            # Get answer choice token IDs
            choice_tokens = {}
            for choice in ['A', 'B', 'C', 'D']:
                # Try different formats
                formats = [choice, f' {choice}', f'({choice})', f' ({choice})']
                for fmt in formats:
                    tokens = self.tokenizer.encode(fmt, add_special_tokens=False)
                    if tokens:
                        choice_tokens[choice] = tokens[0]  # Use first token
                        break
            
            # Find the choice with highest probability
            choice_probs = {}
            for choice, token_id in choice_tokens.items():
                if token_id < len(probs):
                    choice_probs[choice] = probs[token_id].item()
                else:
                    choice_probs[choice] = 0.0
            
            # Get prediction
            if choice_probs:
                predicted_choice = max(choice_probs.keys(), key=lambda k: choice_probs[k])
                confidence = choice_probs[predicted_choice]
            else:
                predicted_choice = 'Unknown'
                confidence = 0.0
            
            # Extract correct answer
            correct_answer = example.get('answer_choice', 'Unknown')
            
            # Get human-readable text (processed prompt)
            text = example.get('text', '')
            if not text and 'input_ids' in example:
                text = self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            
            # Build result dictionary
            result = {
                'example_idx': example_idx,
                'dataset_idx': example.get('index', -1),
                'text': text,  # Processed prompt
                'correct_answer': correct_answer,
                'predicted_answer': predicted_choice,
                'confidence': confidence,
                'choice_probabilities': choice_probs,
                'is_correct': predicted_choice == correct_answer,
                'choice_tokens': choice_tokens
            }
            
            # Add raw dataset information if available
            if 'raw_question' in example:
                result['raw_question'] = example['raw_question']
                result['raw_choices'] = example['raw_choices']
                result['raw_answerKey'] = example['raw_answerKey']
                result['raw_id'] = example.get('raw_id', '')
            
            # Add training label information
            if 'classes' in example:
                result['training_label_idx'] = example['classes']
                if isinstance(example['classes'], (int, np.integer)):
                    result['training_label_letter'] = chr(ord('A') + example['classes'])
                elif isinstance(example['classes'], torch.Tensor):
                    result['training_label_letter'] = chr(ord('A') + example['classes'].item())
            
            return result
            
        except Exception as e:
            return {
                'example_idx': example_idx,
                'dataset_idx': example.get('index', -1),
                'error': str(e),
                'predicted_answer': 'Error',
                'confidence': 0.0,
                'is_correct': False
            }
    
    def _save_epoch_predictions(self, epoch_predictions: Dict):
        """
        Save predictions for an epoch to human-readable text file.
        
        Args:
            epoch_predictions: Prediction data for the epoch
        """
        epoch = epoch_predictions['epoch']
        
        # Create predictions file
        pred_file = self.output_dir / f"predictions_epoch_{epoch:03d}.txt"
        
        with open(pred_file, 'w', encoding='utf-8') as f:
            f.write(f"PREDICTION TRACKING - EPOCH {epoch}\n")
            f.write("=" * 80 + "\n\n")
            
            # Training predictions
            if epoch_predictions['train_predictions']:
                f.write(f"TRAINING EXAMPLES ({len(epoch_predictions['train_predictions'])} samples)\n")
                f.write("-" * 50 + "\n")
                
                for pred in epoch_predictions['train_predictions']:
                    self._write_prediction_to_file(f, pred)
                
                f.write("\n\n")
            
            # Validation predictions
            if epoch_predictions['val_predictions']:
                f.write(f"VALIDATION EXAMPLES ({len(epoch_predictions['val_predictions'])} samples)\n")
                f.write("-" * 50 + "\n")
                
                for pred in epoch_predictions['val_predictions']:
                    self._write_prediction_to_file(f, pred)
        
        # Also save as JSON for programmatic analysis
        json_file = self.output_dir / f"predictions_epoch_{epoch:03d}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(epoch_predictions, f, indent=2, ensure_ascii=False)
    
    def _write_prediction_to_file(self, f, prediction: Dict):
        """
        Write a single prediction to file in human-readable format.
        
        Displays:
        1. Original raw question, choices, and answerKey from dataset
        2. Processed input and training label used by model
        3. Model's prediction and confidence
        
        Args:
            f: File handle
            prediction: Prediction dictionary
        """
        if 'error' in prediction:
            f.write(f"Example {prediction['example_idx']}: ERROR - {prediction['error']}\n\n")
            return
        
        f.write(f"Example {prediction['example_idx']} (Dataset Index: {prediction.get('dataset_idx', 'N/A')})\n")
        f.write(f"{'='*80}\n\n")
        
        # ============ SECTION 1: ORIGINAL DATASET (RAW) ============
        if 'raw_question' in prediction:
            f.write("[ 1. ORIGINAL DATASET (Raw, Unprocessed) ]\n")
            f.write("-" * 80 + "\n")
            
            # Dataset ID
            if 'raw_id' in prediction and prediction['raw_id']:
                f.write(f"Dataset ID: {prediction['raw_id']}\n\n")
            
            # Question
            f.write(f"Question:\n  {prediction['raw_question']}\n\n")
            
            # Choices
            f.write("Choices:\n")
            raw_choices = prediction.get('raw_choices', {})
            if 'text' in raw_choices and 'label' in raw_choices:
                for label, text in zip(raw_choices['label'], raw_choices['text']):
                    marker = " ← CORRECT" if label == prediction.get('raw_answerKey', '') else ""
                    f.write(f"  {label}) {text}{marker}\n")
            
            # Answer Key
            f.write(f"\nAnswer Key (Ground Truth): {prediction.get('raw_answerKey', 'Unknown')}\n")
            f.write("\n")
        
        # ============ SECTION 2: TRAINING INPUT & LABEL ============
        f.write("[ 2. TRAINING INPUT (Processed for Model) ]\n")
        f.write("-" * 80 + "\n")
        
        # Processed prompt
        text = prediction.get('text', 'N/A')
        # Truncate if too long but show it's truncated
        if len(text) > 600:
            f.write(f"Formatted Prompt (truncated):\n{text[:600]}\n  [...truncated...]\n\n")
        else:
            f.write(f"Formatted Prompt:\n{text}\n\n")
        
        # Training label
        training_label_letter = prediction.get('training_label_letter', prediction.get('correct_answer', 'Unknown'))
        training_label_idx = prediction.get('training_label_idx', 'N/A')
        f.write(f"Training Label:\n")
        f.write(f"  Letter: {training_label_letter}\n")
        f.write(f"  Index:  {training_label_idx} (class index used in loss computation)\n")
        f.write("\n")
        
        # ============ SECTION 3: MODEL PREDICTION ============
        f.write("[ 3. MODEL PREDICTION ]\n")
        f.write("-" * 80 + "\n")
        
        predicted = prediction.get('predicted_answer', 'Unknown')
        confidence = prediction.get('confidence', 0.0)
        is_correct = prediction.get('is_correct', False)
        
        f.write(f"Predicted Answer: {predicted}\n")
        f.write(f"Confidence:       {confidence:.4f} ({confidence*100:.2f}%)\n")
        f.write(f"Result:           {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")
        
        # Choice probabilities
        if 'choice_probabilities' in prediction and prediction['choice_probabilities']:
            f.write("\nAll Choice Probabilities:\n")
            for choice in ['A', 'B', 'C', 'D']:
                prob = prediction['choice_probabilities'].get(choice, 0.0)
                marker = " ← PREDICTED" if choice == predicted else ""
                correct_marker = " (CORRECT)" if choice == training_label_letter else ""
                f.write(f"  {choice}: {prob:.4f} ({prob*100:.2f}%){marker}{correct_marker}\n")
        
        # Token IDs used
        if 'choice_tokens' in prediction and prediction['choice_tokens']:
            f.write("\nAnswer Token IDs:\n")
            for choice in ['A', 'B', 'C', 'D']:
                token_id = prediction['choice_tokens'].get(choice, 'N/A')
                f.write(f"  {choice}: {token_id}\n")
        
        f.write("\n" + "="*80 + "\n\n")

    
    def _save_selected_examples_info(self):
        """Save information about selected examples for reference."""
        info_file = self.output_dir / "selected_examples_info.txt"
        
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("SELECTED EXAMPLES FOR PREDICTION TRACKING\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Random Seed: {self.seed}\n")
            f.write(f"Examples per split: {self.n_examples}\n\n")
            
            # Training examples info
            if self.train_examples:
                f.write(f"TRAINING EXAMPLES ({len(self.train_examples)} selected)\n")
                f.write("-" * 50 + "\n")
                
                for i, example in enumerate(self.train_examples):
                    f.write(f"Example {i} (Dataset Index: {example.get('index', 'N/A')})\n")
                    
                    # Show raw question if available
                    if 'raw_question' in example:
                        f.write(f"  Question: {example['raw_question'][:100]}...\n")
                        f.write(f"  Answer Key: {example.get('raw_answerKey', 'Unknown')}\n")
                    
                    # Show training label
                    f.write(f"  Training Label: {example.get('answer_choice', 'Unknown')}\n")
                    
                    # Show truncated processed text
                    text = example.get('text', '')[:150]
                    f.write(f"  Processed Text: {text}{'...' if len(example.get('text', '')) > 150 else ''}\n\n")
            
            # Validation examples info
            if self.val_examples:
                f.write(f"VALIDATION EXAMPLES ({len(self.val_examples)} selected)\n")
                f.write("-" * 50 + "\n")
                
                for i, example in enumerate(self.val_examples):
                    f.write(f"Example {i} (Dataset Index: {example.get('index', 'N/A')})\n")
                    
                    # Show raw question if available
                    if 'raw_question' in example:
                        f.write(f"  Question: {example['raw_question'][:100]}...\n")
                        f.write(f"  Answer Key: {example.get('raw_answerKey', 'Unknown')}\n")
                    
                    # Show training label
                    f.write(f"  Training Label: {example.get('answer_choice', 'Unknown')}\n")
                    
                    # Show truncated processed text
                    text = example.get('text', '')[:150]
                    f.write(f"  Processed Text: {text}{'...' if len(example.get('text', '')) > 150 else ''}\n\n")
        
        print(f"[PredictionTracker] Saved example selection info to: {info_file}")
    
    def generate_progress_summary(self):
        """Generate a summary of prediction progress across all epochs."""
        if not self.prediction_history['train'] and not self.prediction_history['val']:
            return
        
        summary_file = self.output_dir / "prediction_progress_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PREDICTION PROGRESS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Calculate accuracy trends
            train_accuracies = []
            val_accuracies = []
            
            for epoch_preds in self.prediction_history['train']:
                if epoch_preds:
                    correct = sum(1 for p in epoch_preds if p.get('is_correct', False))
                    total = len(epoch_preds)
                    train_accuracies.append(correct / total if total > 0 else 0.0)
            
            for epoch_preds in self.prediction_history['val']:
                if epoch_preds:
                    correct = sum(1 for p in epoch_preds if p.get('is_correct', False))
                    total = len(epoch_preds)
                    val_accuracies.append(correct / total if total > 0 else 0.0)
            
            # Write summary
            f.write("ACCURACY EVOLUTION:\n")
            f.write("-" * 30 + "\n")
            
            for epoch, (train_acc, val_acc) in enumerate(zip(train_accuracies, val_accuracies)):
                f.write(f"Epoch {epoch:2d}: Train={train_acc:.3f}, Val={val_acc:.3f}\n")
            
            if train_accuracies:
                f.write(f"\nTrain Accuracy: {train_accuracies[0]:.3f} → {train_accuracies[-1]:.3f} "
                       f"(Δ={train_accuracies[-1] - train_accuracies[0]:+.3f})\n")
            
            if val_accuracies:
                f.write(f"Val Accuracy:   {val_accuracies[0]:.3f} → {val_accuracies[-1]:.3f} "
                       f"(Δ={val_accuracies[-1] - val_accuracies[0]:+.3f})\n")
        
        print(f"[PredictionTracker] Generated progress summary: {summary_file}")