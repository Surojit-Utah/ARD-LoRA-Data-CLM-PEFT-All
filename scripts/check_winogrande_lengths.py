"""Check if WinoGrande examples are being truncated at max_len=128"""

from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)

# Load WinoGrande-S
dataset = load_dataset("winogrande", "winogrande_s")

print("=" * 80)
print("WINOGRANDE-S SEQUENCE LENGTH ANALYSIS")
print("=" * 80)

# Sample 100 examples
sample_size = 100
lengths = []
truncated_count = 0

for i, example in enumerate(dataset["train"].select(range(sample_size))):
    question = example["sentence"]
    option1 = example["option1"]
    option2 = example["option2"]
    
    # Format like S2ClassDataset does
    text = f"Select one of the choices that answers the following question: {question} Choices: A. {option1}. B. {option2}. Answer:"
    
    # Tokenize
    tokens = tokenizer(text, truncation=False)
    length = len(tokens["input_ids"])
    lengths.append(length)
    
    if length > 128:
        truncated_count += 1
        if truncated_count <= 3:  # Show first 3 truncated examples
            print(f"\n[TRUNCATED Example {i}]")
            print(f"Length: {length} tokens (EXCEEDS max_len=128)")
            print(f"Text: {text[:200]}...")

print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)
print(f"Sample size: {sample_size}")
print(f"Min length: {min(lengths)} tokens")
print(f"Max length: {max(lengths)} tokens")
print(f"Mean length: {sum(lengths) / len(lengths):.1f} tokens")
print(f"Median length: {sorted(lengths)[len(lengths) // 2]} tokens")
print(f"\nTruncated at max_len=128: {truncated_count}/{sample_size} ({100 * truncated_count / sample_size:.1f}%)")
print(f"Truncated at max_len=256: {sum(1 for l in lengths if l > 256)}/{sample_size}")
print(f"Truncated at max_len=512: {sum(1 for l in lengths if l > 512)}/{sample_size}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
if truncated_count > 0:
    print(f"⚠️  WARNING: {truncated_count} examples exceed max_len=128!")
    print(f"   This means the 'Answer:' token may be truncated, preventing learning.")
    print(f"\n   SOLUTION: Increase max_len to at least {max(lengths)} tokens")
    print(f"   Or use max_len=256 to safely cover {100 * (1 - sum(1 for l in lengths if l > 256) / sample_size):.1f}% of examples")
else:
    print("✓ All examples fit within max_len=128")
