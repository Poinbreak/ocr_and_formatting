import cv2
import numpy as np
import json
import torch
import sys
import os
from PIL import Image

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Import Qwen
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Import NuExtract
from transformers import AutoModelForCausalLM

def enhance_image_for_ocr(image_path, output_path):
    print(f"[*] Enhancing image: {image_path}")
    # 1. Read image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Adaptive thresholding to remove shadows and varying lighting
    # This creates a perfect black-and-white binary image
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    
    # 4. Optional: Slight dilation to thicken very faint pen strokes
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    cv2.imwrite(output_path, dilated)
    print(f"[*] Enhanced image saved to: {output_path}")
    return output_path

def run_qwen_ocr(image_path):
    print(f"[*] Loading Qwen2.5-VL for OCR on {image_path}...")
    device = "cpu"
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": "Extract all the text from this image exactly as written. Do not add any conversational text or formatting."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    print("[*] Generating OCR text...")
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Free up memory before loading the next model
    del model
    del processor
    import gc
    gc.collect()
    
    return output_text

def extract_json_with_nuextract(text, schema):
    print("[*] Loading NuExtract-tiny for JSON extraction...")
    device = "cpu"
    model_id = "numind/NuExtract-tiny"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    
    schema_str = json.dumps(json.loads(schema), indent=4)
    input_llm = "<|input|>\n### Template:\n" + schema_str + "\n### Text:\n" + text + "\n<|output|>\n"
    
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=4000).to(device)
    print("[*] Parsing structured JSON...")
    output = tokenizer.decode(model.generate(**input_ids, max_new_tokens=1000)[0], skip_special_tokens=True)
    
    try:
        json_result = output.split("<|output|>")[1].split("<|end-")[0]
    except Exception:
        json_result = output
        
    return json_result

if __name__ == "__main__":
    import glob
    images = glob.glob("*.jpg") + glob.glob("*.png")
    
    # Filter out already enhanced images to avoid loop
    images = [img for img in images if not img.startswith("enhanced_")]
    
    if not images:
        print("No images found in the current directory.")
        sys.exit(0)
        
    target_image = images[0] # Test on the first available image
    enhanced_path = f"enhanced_{target_image}"
    
    # 1. Enhance
    enhance_image_for_ocr(target_image, enhanced_path)
    
    # 2. OCR
    raw_text = run_qwen_ocr(enhanced_path)
    with open("pipeline_raw_ocr.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
        
    # 3. JSON Extract
    schema = '''{
      "subject": "",
      "concepts": [
        {
          "concept_name": "",
          "definition": ""
        }
      ],
      "departments": [""]
    }'''
    
    json_data = extract_json_with_nuextract(raw_text, schema)
    
    with open("pipeline_final.json", "w", encoding="utf-8") as f:
        f.write(json_data)
        
    print(f"\n{'='*40}")
    print("PIPELINE COMPLETE!")
    print(f"Raw OCR saved to: pipeline_raw_ocr.txt")
    print(f"Final JSON saved to: pipeline_final.json")
    print(f"{'='*40}")
    print(json_data)
