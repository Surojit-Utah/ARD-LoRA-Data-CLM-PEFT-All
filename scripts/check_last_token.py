"""Check what the last token actually is"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# ARC-Challenge format
arc_text = "Select one of the choices that answers the following question: What is X? Choices: A. Option A. B. Option B. C. Option C. D. Option D. Answer:"

# WinoGrande format (current - missing period)
wino_text_wrong = "Select one of the choices that answers the following question: What is X? Choices: A. Option A. B Option B. Answer:"

# WinoGrande format (fixed - with period)
wino_text_fixed = "Select one of the choices that answers the following question: What is X? Choices: A. Option A. B. Option B. Answer:"

print("=" * 80)
print("TOKENIZATION ANALYSIS")
print("=" * 80)

for name, text in [("ARC-Challenge", arc_text), ("WinoGrande (wrong)", wino_text_wrong), ("WinoGrande (fixed)", wino_text_fixed)]:
    print(f"\n{name}:")
    print(f"Text: ...{text[-60:]}")
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"\nLast 10 token IDs: {tokens[-10:]}")
    print(f"Last 10 tokens:")
    for i, tid in enumerate(tokens[-10:]):
        token_str = tokenizer.convert_ids_to_tokens(tid)
        print(f"  [{i-10}] {tid:5d} -> {repr(token_str)}")
    
    print(f"\nLast token ID: {tokens[-1]}")
    print(f"Last token: {repr(tokenizer.convert_ids_to_tokens(tokens[-1]))}")
    print(f"Decoded last token: {repr(tokenizer.decode([tokens[-1]]))}")

print("\n" + "=" * 80)
print("WHAT COMES AFTER 'Answer:'?")
print("=" * 80)

# Check what token IDs correspond to " A" and " B"
for label in [" A", " B", " C", " D"]:
    ids = tokenizer.encode(label, add_special_tokens=False)
    print(f"{repr(label):6s} -> IDs: {ids}, tokens: {[tokenizer.convert_ids_to_tokens(i) for i in ids]}")
    print(f"         Last ID: {ids[-1]}")

