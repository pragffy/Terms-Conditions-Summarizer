import sys
sys.stdout.reconfigure(encoding='utf-8')

from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Load model (first time downloads ~1.6GB)
print("Loading summarizer...")
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()
print("[OK] Model loaded.")

# Test with a short paragraph
test_text = """
You may cancel within 14 days for a full refund. 
The company may change fees with 30 days notice. 
Your data will not be sold. 
The company disclaims all liability for indirect damages. 
You agree to indemnify the company. 
Binding arbitration applies. 
No refunds for partial months.
"""

print("Generating summary...")
inputs = tokenizer(test_text, return_tensors="pt", max_length=1024, truncation=True)

with torch.no_grad():
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=130,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

with open("summary_en.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("\n--- Running Validation ---")
# Load your saved summary
with open("summary_en.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Count words
word_count = len(text.split())
print(f"Word count: {word_count}")

# Check if within range
try:
    assert 130 <= word_count <= 150, f"Failed: {word_count} words"
    print("✅ Success: Word count is within 130-150 range.")
except AssertionError as e:
    print(f"❌ {e}")

# Print first 500 chars to manually review
print("\nSummary Output:")
print(text[:500])
