import torch
import sys
import os
import base64
import io
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Set standard output encoding to handle unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Qwen2.5-VL model on {device}...")

# 2. Load the Model and Processor
# Use Qwen3-VL-8B-Instruct (or swap to any Qwen2.5/Qwen3 VL model_id)
model_id = "Qwen/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
# Load model with mixed precision if cuda, float32 if cpu
dtype = torch.bfloat16 if device == "cuda" else torch.float32
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    trust_remote_code=True,
)
model.eval()


def image_to_base64(image_path: str) -> str:
    """Read an image file and return a base64-encoded PNG data URI."""
    with Image.open(image_path) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# 3. Image to process
image_dir = r"C:\Saiyanht\projects\digitwin"
valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
image_files = [
    f for f in os.listdir(image_dir)
    if f.lower().endswith(valid_extensions)
]

prompt_text = """You are an ultra-strict, pixel-literal OCR Engine. Your singular objective is to transcribe the exact characters present in the image based ONLY on their physical visual strokes, completely ignoring context, grammar, or expected meaning.

CRITICAL DIRECTIVES:
1. Pixel-Level Literal Transcription: You must read exactly what is physically written on the page, flaws and all. Do not autocorrect, do not fix typos, and do not make the text make sense.
2. ZERO Contextual Guessing: Never use surrounding text to guess a poorly written word. If a word is "注水西瓜" but looks like "淫水雨孤", you MUST output exactly what it visually looks like. Do not assume repeating lines are identical if the handwriting differs.
3. Record Malformed Characters: If a character is malformed, transcribe the character it most visually resembles, even if it creates a nonsensical word. If it is completely unrecognizable, use [UNREADABLE].
4. Absolute Numerical Precision: Numbers, dates, and identifiers must be extracted exactly as drawn. 

Task: Transcribe the image. Prioritize visual accuracy over semantic meaning. Do not hallucinate."""

for img_file in image_files:
    image_path = os.path.join(image_dir, img_file)
    print(f"\n{'='*60}")
    print(f"Processing: {img_file}")
    print(f"{'='*60}")

    # 4. Encode image as base64 data URI (Qwen3-VL compatible)
    image_data_uri = image_to_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri},
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 5. Generate OCR output
    output = model.generate(
        **inputs,
        max_new_tokens=2048,
    )

    # 6. Decode result
    generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)
    ]
    result = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    print(f"\n--- OCR RESULT for {img_file} ---")
    # Save to file (UTF-8)
    output_file = os.path.join(image_dir, f"{os.path.splitext(img_file)[0]}_ocr_qwen.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Result saved to: {output_file}")
    print(result)
