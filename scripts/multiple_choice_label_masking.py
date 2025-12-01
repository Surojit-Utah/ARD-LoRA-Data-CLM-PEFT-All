"""
Multiple Choice Label Masking for ARD-LoRA Training
===================================================

This script implements proper label masking for multiple choice tasks to focus
training only on answer tokens (A, B, C, D) rather than the entire prompt.

Key Features:
1. Masks all prompt tokens with -100 (ignored in loss computation)
2. Only supervises answer tokens (A, B, C, D)
3. Supports various answer formats: " A", "A", "(A)", etc.
4. Provides debugging and validation tools
5. Compatible with ARD-LoRA training pipeline

The core insight: For multiple choice tasks, we only need to predict the answer
token(s), not regenerate the entire question. This focuses the learning signal
and prevents the loss from being dominated by question/choice text.
"""

import torch
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
import re
import json


class MultipleChoiceLabelMasker:
    """
    Handles label masking for multiple choice datasets to focus training
    only on answer tokens (A, B, C, D).
    """
    
    def __init__(self, tokenizer, answer_choices=None, debug=False):
        """
        Initialize the label masker.
        
        Args:
            tokenizer: HuggingFace tokenizer
            answer_choices: List of valid answer choices (default: ["A", "B", "C", "D"])
            debug: Enable debug logging
        """
        self.tokenizer = tokenizer
        self.answer_choices = answer_choices or ["A", "B", "C", "D"]
        self.debug = debug
        
        # Pre-tokenize answer choices in various formats
        self.answer_token_patterns = self._prepare_answer_patterns()
        
        if self.debug:
            print(f"[MASKER] Initialized with answer choices: {self.answer_choices}")
            print(f"[MASKER] Answer token patterns prepared: {len(self.answer_token_patterns)} variants")
    
    def _prepare_answer_patterns(self) -> Dict[str, List[int]]:
        """
        Pre-tokenize answer choices in various common formats.
        
        Returns:
            Dictionary mapping answer choice to list of possible token sequences
        """
        patterns = {}
        
        for choice in self.answer_choices:
            choice_patterns = []
            
            # Common answer formats to check
            formats = [
                choice,           # "A"
                f" {choice}",     # " A" (with leading space)
                f"({choice})",    # "(A)"
                f" ({choice})",   # " (A)"
                f"{choice}.",     # "A."
                f" {choice}.",    # " A."
                f"{choice})",     # "A)"
                f" {choice})",    # " A)"
            ]
            
            for fmt in formats:
                tokens = self.tokenizer.encode(fmt, add_special_tokens=False)
                if tokens:  # Only add non-empty tokenizations
                    choice_patterns.append(tokens)
            
            patterns[choice] = choice_patterns
            
            if self.debug:
                print(f"[MASKER] Answer '{choice}' tokenization patterns:")
                for i, tokens in enumerate(choice_patterns):
                    decoded = self.tokenizer.decode(tokens)
                    print(f"[MASKER]   Pattern {i+1}: {tokens} -> '{decoded}'")
        
        return patterns
    
    def find_answer_tokens(self, input_ids: List[int], target_answer: str) -> Optional[Tuple[int, int]]:
        """
        Find the position of answer tokens in the input sequence.
        Searches from the END to avoid matching choice enumerators instead of the final answer.
        
        Args:
            input_ids: List of token IDs
            target_answer: The correct answer choice (e.g., "A")
        
        Returns:
            Tuple of (start_idx, end_idx) of answer tokens, or None if not found
        """
        if target_answer not in self.answer_token_patterns:
            if self.debug:
                print(f"[MASKER] WARNING: Answer '{target_answer}' not in prepared patterns")
            return None
        
        patterns = self.answer_token_patterns[target_answer]
        
        # Search for each pattern from the END of the sequence (reverse search)
        # This prioritizes finding the final answer rather than choice enumerators
        best_match = None
        best_position = -1
        
        for pattern in patterns:
            # Search backwards through the sequence
            for i in range(len(input_ids) - len(pattern), -1, -1):
                if input_ids[i:i+len(pattern)] == pattern:
                    # Take the rightmost (latest) match
                    if i > best_position:
                        best_position = i
                        best_match = (i, i + len(pattern))
                        if self.debug:
                            found_tokens = input_ids[i:i+len(pattern)]
                            decoded = self.tokenizer.decode(found_tokens)
                            print(f"[MASKER] Found answer tokens at positions {i}-{i+len(pattern)-1}: {found_tokens} -> '{decoded}' (searching from end)")
        
        if best_match is None and self.debug:
            print(f"[MASKER] WARNING: Could not find answer '{target_answer}' in sequence")
            print(f"[MASKER] Input sequence: {self.tokenizer.decode(input_ids)}")
        
        return best_match
    
    def mask_labels(self, input_ids: List[int], target_answer: str) -> List[int]:
        """
        Create masked labels where only answer tokens contribute to loss.
        
        Args:
            input_ids: List of token IDs for the entire sequence
            target_answer: The correct answer choice (e.g., "A")
        
        Returns:
            List of label IDs with -100 for masked positions
        """
        # Initialize all labels as masked
        labels = [-100] * len(input_ids)
        
        # Find answer token positions
        answer_span = self.find_answer_tokens(input_ids, target_answer)
        
        if answer_span is not None:
            start_idx, end_idx = answer_span
            # Unmask only the answer tokens
            labels[start_idx:end_idx] = input_ids[start_idx:end_idx]
            
            if self.debug:
                unmasked_tokens = input_ids[start_idx:end_idx]
                decoded = self.tokenizer.decode(unmasked_tokens)
                print(f"[MASKER] Unmasked {end_idx - start_idx} answer tokens: '{decoded}'")
        else:
            if self.debug:
                print(f"[MASKER] WARNING: No answer tokens found - all labels masked!")
        
        return labels
    
    def process_batch(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Process a batch of examples to add masked labels.
        
        Args:
            examples: Dictionary with 'input_ids' and 'answer' keys
        
        Returns:
            Updated examples dictionary with 'labels' added
        """
        if 'input_ids' not in examples or 'answer' not in examples:
            raise ValueError("Examples must contain 'input_ids' and 'answer' keys")
        
        batch_size = len(examples['input_ids'])
        labels = []
        
        for i in range(batch_size):
            input_ids = examples['input_ids'][i]
            answer = examples['answer'][i]
            
            # Create masked labels for this example
            masked_labels = self.mask_labels(input_ids, answer)
            labels.append(masked_labels)
            
            if self.debug and i == 0:  # Debug first example in batch
                print(f"[MASKER] Batch example {i}:")
                print(f"[MASKER]   Input: {self.tokenizer.decode(input_ids)}")
                print(f"[MASKER]   Answer: {answer}")
                print(f"[MASKER]   Labels: {sum(1 for x in masked_labels if x != -100)} unmasked tokens")
        
        examples['labels'] = labels
        return examples


def validate_label_masking(tokenizer, test_cases: List[Dict[str, Any]], debug=True):
    """
    Validate label masking with test cases.
    
    Args:
        tokenizer: HuggingFace tokenizer
        test_cases: List of test cases with 'text' and 'answer' keys
        debug: Enable debug output
    """
    print("=" * 80)
    print("MULTIPLE CHOICE LABEL MASKING VALIDATION")
    print("=" * 80)
    
    masker = MultipleChoiceLabelMasker(tokenizer, debug=debug)
    
    for i, case in enumerate(test_cases):
        print(f"\n[TEST CASE {i+1}]")
        print(f"Text: {case['text']}")
        print(f"Answer: {case['answer']}")
        
        # Tokenize the text
        input_ids = tokenizer.encode(case['text'], add_special_tokens=False)
        print(f"Input IDs: {input_ids}")
        print(f"Decoded: {tokenizer.decode(input_ids)}")
        
        # Create masked labels
        labels = masker.mask_labels(input_ids, case['answer'])
        
        # Analyze results
        total_tokens = len(labels)
        unmasked_count = sum(1 for x in labels if x != -100)
        masked_count = total_tokens - unmasked_count
        
        print(f"Total tokens: {total_tokens}")
        print(f"Masked tokens (-100): {masked_count}")
        print(f"Unmasked tokens (answer): {unmasked_count}")
        
        if unmasked_count > 0:
            unmasked_tokens = [input_ids[j] for j, label in enumerate(labels) if label != -100]
            unmasked_text = tokenizer.decode(unmasked_tokens)
            print(f"Unmasked text: '{unmasked_text}'")
            print(f"SUCCESS: Found answer tokens")
        else:
            print(f"FAILURE: No answer tokens found")
        
        # Show label structure
        if debug:
            print(f"Label structure:")
            for j, (token_id, label) in enumerate(zip(input_ids, labels)):
                token_text = tokenizer.decode([token_id])
                status = "UNMASKED" if label != -100 else "MASKED"
                print(f"  {j:2d}: {token_id:5d} -> '{token_text:10s}' ({status})")


def create_test_cases():
    """Create test cases for validation."""
    return [
        {
            "text": "What is the capital of France? A) London B) Berlin C) Paris D) Madrid The answer is C",
            "answer": "C"
        },
        {
            "text": "Which planet is closest to the Sun? (A) Venus (B) Mercury (C) Earth (D) Mars Answer: B",
            "answer": "B"
        },
        {
            "text": "What is 2 + 2? A. 3 B. 4 C. 5 D. 6 The correct answer is B.",
            "answer": "B"
        },
        {
            "text": "The chemical symbol for gold is: A) Au B) Ag C) Fe D) Cu Answer: A",
            "answer": "A"
        },
        {
            "text": "Complete the sequence: 1, 4, 7, 10, ? A) 11 B) 12 C) 13 D) 14 Answer: C) 13",
            "answer": "C"
        }
    ]


def main():
    """Main function to demonstrate and validate label masking."""
    print("Multiple Choice Label Masking Demo")
    print("=" * 50)
    
    # Initialize tokenizer (using a common model for demo)
    model_name = "meta-llama/Llama-2-7b-hf"  # Or use your specific model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loaded tokenizer: {model_name}")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        print("Using a fallback tokenizer for demo...")
        # You can add fallback logic here if needed
        return
    
    # Create and run test cases
    test_cases = create_test_cases()
    validate_label_masking(tokenizer, test_cases, debug=True)
    
    # Performance Analysis
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    masker = MultipleChoiceLabelMasker(tokenizer, debug=False)
    
    total_tokens = 0
    total_unmasked = 0
    successful_cases = 0
    
    for case in test_cases:
        input_ids = tokenizer.encode(case['text'], add_special_tokens=False)
        labels = masker.mask_labels(input_ids, case['answer'])
        
        tokens_count = len(labels)
        unmasked_count = sum(1 for x in labels if x != -100)
        
        total_tokens += tokens_count
        total_unmasked += unmasked_count
        
        if unmasked_count > 0:
            successful_cases += 1
    
    print(f"Test cases processed: {len(test_cases)}")
    print(f"Successful answer detection: {successful_cases}/{len(test_cases)} ({100*successful_cases/len(test_cases):.1f}%)")
    print(f"Total tokens: {total_tokens}")
    print(f"Unmasked (answer) tokens: {total_unmasked}")
    print(f"Masked (prompt) tokens: {total_tokens - total_unmasked}")
    print(f"Token efficiency: {100*total_unmasked/total_tokens:.2f}% of tokens contribute to loss")
    
    print(f"\nKEY INSIGHT:")
    print(f"   Instead of training on {total_tokens} tokens (including questions),")
    print(f"   we focus on only {total_unmasked} answer tokens ({100*total_unmasked/total_tokens:.1f}%).")
    print(f"   This prevents question text from dominating the loss signal!")


if __name__ == "__main__":
    main()