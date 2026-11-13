# Step 3: Reproducibility Report (Same Config, Same Seed = 42)

## 1. Measurements Summary
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (CPU backend, `torch 2.15.0+cpu`)
- **Seed:** `42` (2 epochs, 3 train samples, `batch_size=1`, `lr=2e-4`)
- **Loss curve Run 1:** `[1.489572286605835, 1.489572286605835]`
- **Loss curve Run 2:** `[1.489572286605835, 1.489572286605835]`
- **Max Loss Divergence:** `0.000000000000e+00`
- **Adapter SHA-256 Identical:** `True`
  - Run 1 (`adapter_model.safetensors`): `2c570d685d87bd67ed57c5af332715e6ca9f75d7ce94e69eebeff79253aea133`
  - Run 2 (`adapter_model.safetensors`): `2c570d685d87bd67ed57c5af332715e6ca9f75d7ce94e69eebeff79253aea133`
- **Max Weight Tensor Abs Diff:** `0.000000000000e+00`
- **Mean Weight Tensor Abs Diff:** `0.000000000000e+00`
- **Inference Outputs Match (`temperature=0`):** `True` (infer exit codes: `0`, `0`)

## 2. Inference Comparison
- **Run 1 outputs:** `['{"prompt": "Say hello", "response": "Hello! How can I assist you today?", "tokens_generated": 10}', '{"prompt": "What is 2+2?", "response": "2 + 2 equals 4.\\n\\nThis is the basic arithmetic operation of addition in mathematics. When you add two numbers together, you combine them to find their total sum. In this case, 2 and 2 are being added together, resulting in a total of 4.", "tokens_generated": 58}', '{"prompt": "Translate cat to Russian", "response": "Кот", "tokens_generated": 3}']`
- **Run 2 outputs:** `['{"prompt": "Say hello", "response": "Hello! How can I assist you today?", "tokens_generated": 10}', '{"prompt": "What is 2+2?", "response": "2 + 2 equals 4.\\n\\nThis is the basic arithmetic operation of addition in mathematics. When you add two numbers together, you combine them to find their total sum. In this case, 2 and 2 are being added together, resulting in a total of 4.", "tokens_generated": 58}', '{"prompt": "Translate cat to Russian", "response": "Кот", "tokens_generated": 3}']`
