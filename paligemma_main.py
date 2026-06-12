import torch
import sys
import os
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading PaliGemma 3B model on {device}...")

model_id = "google/paligemma-3b-mix-448"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# NOTE: Downloading PaliGemma requires a Hugging Face Token (HF_TOKEN) with access granted 
# by accepting the model license on the model's page: https://huggingface.co/google/paligemma-3b-mix-448
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device
)
processor = AutoProcessor.from_pretrained(model_id)

image_dir = r"C:\Saiyanht\projects\digitwin"
valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
image_files = [
    f for f in os.listdir(image_dir)
    if f.lower().endswith(valid_extensions)
]

prompt_text = "ocr" # PaliGemma uses specific task strings, "ocr" is optimized for transcription

for img_file in image_files:
    image_path = os.path.join(image_dir, img_file)
    print(f"\n{'='*60}")
    print(f"Processing: {img_file}")
    print(f"{'='*60}")

    image = Image.open(image_path).convert('RGB')
    
    inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=2048)

    # Decode result
    result = processor.decode(output[0], skip_special_tokens=True)
    if result.startswith(prompt_text):
        result = result[len(prompt_text):].strip()

    print(f"\n--- OCR RESULT for {img_file} ---")
    output_file = os.path.join(image_dir, f"{os.path.splitext(img_file)[0]}_ocr_paligemma.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Result saved to: {output_file}")
    print(result)
