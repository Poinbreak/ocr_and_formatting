import torch
import sys
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
from PIL import Image
import os

# Set standard output encoding to handle unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

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

    prompt_text = """You are an ultra-strict, pixel-literal OCR Engine. Your singular objective is to transcribe the exact characters present in the image based ONLY on their physical visual strokes, completely ignoring context, grammar, or expected meaning.

CRITICAL DIRECTIVES:
1. Pixel-Level Literal Transcription: You must read exactly what is physically written on the page, flaws and all. Do not autocorrect, do not fix typos, and do not make the text make sense.
2. ZERO Contextual Guessing: Never use surrounding text to guess a poorly written word. If a word is "注水西瓜" but looks like "淫水雨孤", you MUST output exactly what it visually looks like. Do not assume repeating lines are identical if the handwriting differs.
3. Record Malformed Characters: If a character is malformed, transcribe the character it most visually resembles, even if it creates a nonsensical word. If it is completely unrecognizable, use [UNREADABLE].
4. Absolute Numerical Precision: Numbers, dates, and identifiers must be extracted exactly as drawn. 

Task: Transcribe the image. Prioritize visual accuracy over semantic meaning. Do not hallucinate."""

    # 4. Prepare inputs using chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
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
    print(result)