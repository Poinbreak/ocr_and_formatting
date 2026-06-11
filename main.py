import torch
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
from PIL import Image
import os

# 1. Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading GLM-OCR model on {device}...")

# 2. Load the Model and Processor
model_id = "zai-org/GLM-OCR"

processor = AutoProcessor.from_pretrained(model_id)
model = GlmOcrForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
).to(device)

# 3. Image to process
image_dir = r"C:\Saiyanht\projects\digitwin"
image_files = [
    "10032_2024_468_Fig18_HTML.png",
]

for img_file in image_files:
    image_path = os.path.join(image_dir, img_file)
    print(f"\n{'='*60}")
    print(f"Processing: {img_file}")
    print(f"{'='*60}")

    image = Image.open(image_path).convert("RGB")

    # 4. Prepare inputs using chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    # 5. Generate OCR output
    output = model.generate(
        **inputs,
        max_new_tokens=2048,
    )

    # 6. Decode result
    result = processor.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    print(f"\n--- OCR RESULT for {img_file} ---")
    # Save to file (UTF-8)
    output_file = os.path.join(image_dir, f"{os.path.splitext(img_file)[0]}_ocr_glm.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Result saved to: {output_file}")
    print(result.encode("utf-8", errors="replace").decode("utf-8"))