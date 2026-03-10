"""Test Qwen chat template with and without default system prompt."""
import sys
sys.path.insert(0, ".")

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# The key change: remove the else clause that injects default system prompt
# Original: if no system message -> inject "You are Qwen, created by Alibaba Cloud..."
# Modified: if no system message -> just skip system block entirely
original = tok.chat_template

# Two occurrences: tools block and non-tools block
# 1) Non-tools block: remove the else that injects full default
modified = original.replace(
    "{%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}",
    ""
)
# 2) Tools block: replace the default content with empty string
modified = modified.replace(
    "{%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}",
    "{%- else %}\n        {{- '' }}"
)

assert "You are Qwen" not in modified, "Failed to remove default system prompt"

print("=== ORIGINAL: without system prompt ===")
print(tok.apply_chat_template([
    {"role": "user", "content": "What is your favorite animal?"},
], tokenize=False, add_generation_prompt=True))

print("\n=== MODIFIED: without system prompt ===")
print(tok.apply_chat_template([
    {"role": "user", "content": "What is your favorite animal?"},
], tokenize=False, add_generation_prompt=True, chat_template=modified))

print("\n=== MODIFIED: with system prompt ===")
print(tok.apply_chat_template([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is your favorite animal?"},
], tokenize=False, add_generation_prompt=True, chat_template=modified))

print("\n=== MODIFIED: training format (no system) ===")
print(tok.apply_chat_template([
    {"role": "user", "content": "Look at these numbers: 234, 567, 891"},
    {"role": "assistant", "content": "123, 456, 789"},
], tokenize=False, add_generation_prompt=False, chat_template=modified))
