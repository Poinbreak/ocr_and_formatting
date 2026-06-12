import torch
import sys
import os
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Set standard output encoding to handle unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading MiniCPM-V 2.6 model on {device}...")

# 2. Load the Model and Tokenizer
model_id = "openbmb/MiniCPM-V-2_6"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.eval()

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

    image = Image.open(image_path).convert('RGB')
    
    msgs = [{'role': 'user', 'content': [image, prompt_text]}]

    # 4. Generate OCR output
    result = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )

    print(f"\n--- OCR RESULT for {img_file} ---")
    output_file = os.path.join(image_dir, f"{os.path.splitext(img_file)[0]}_ocr_minicpm.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Result saved to: {output_file}")
    print(result)
