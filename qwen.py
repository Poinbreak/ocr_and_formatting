"""
================================================================================
KAGGLE CELL 1: Environment Setup & Installations
Run this cell first to install all necessary packages for the Qwen models.
================================================================================
"""
# Block broken torchaudio BEFORE torch loads to prevent circular import crash
# (torch internally tries to import torchaudio during initialization on Kaggle)
import sys as _sys
try:
    import torchaudio as _ta_test
except Exception:
    _sys.modules["torchaudio"] = None
    print("[shim] Blocked broken torchaudio before torch import")

import torch, torchvision
with open("constraints.txt", "w") as f:
    f.write(f"torch=={torch.__version__}\n")
    f.write(f"torchvision=={torchvision.__version__}\n")
    f.write("numpy<2.0.0\n")

!pip install -c constraints.txt transformers accelerate bitsandbytes qwen-vl-utils --upgrade
# Align Pillow after transformers to fix _Ink version mismatch
!pip install "Pillow>=11.2.1" --upgrade -q
# Remove torchaudio to avoid CUDA version mismatch (not needed for vision)
!pip uninstall torchaudio -y -q 2>/dev/null || true


# ══════════════════════════════════════════════════════════════════════
# Runtime compatibility shims - MUST run before 'import torch'
# ══════════════════════════════════════════════════════════════════════
import sys, importlib

# Fix 1: Pillow _Ink ImportError (torchvision <-> Pillow version mismatch)
try:
    _pil_typing = importlib.import_module("PIL._typing")
    if not hasattr(_pil_typing, "_Ink"):
        _pil_typing._Ink = str | float | tuple[int, ...]
        print("[shim] Patched PIL._typing._Ink")
except ImportError:
    pass

# Fix 2: Block torchaudio to prevent CUDA version mismatch crash
# (transformers.loss.loss_rnnt imports torchaudio; we don't need it for vision)
try:
    import torchaudio as _ta_test
except Exception:
    # torchaudio is broken or missing - block it so transformers skips it
    sys.modules["torchaudio"] = None
    print("[shim] Blocked broken torchaudio (not needed for vision models)")

import torch
import gc
from PIL import Image
import glob
import os

# A helper function to prevent Kaggle from crashing due to out-of-memory (OOM) errors.
def clear_vram():
    """Wipes the GPU memory cleanly between model tests."""
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM Cleared.")

print("Searching for images in /kaggle/input/ and /kaggle/working/ ...")
all_files = glob.glob("/kaggle/input/**/*.*", recursive=True) + glob.glob("/kaggle/working/**/*.*", recursive=True)
TEST_IMAGE_PATHS = sorted([
    p for p in all_files 
    if p.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
])[:10]

if not TEST_IMAGE_PATHS:
    print("WARNING: Could not find any images! Downloading fallbacks...")
    import urllib.request
    os.makedirs("/kaggle/working/samples", exist_ok=True)
    urls = [
        "https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Doc.jpg",
        "https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/mm_tutorial/Receipt.jpg"
    ]
    for i, url in enumerate(urls):
        path = f"/kaggle/working/samples/sample_{i}.jpg"
        urllib.request.urlretrieve(url, path)
        TEST_IMAGE_PATHS.append(path)

print(f"Loaded {len(TEST_IMAGE_PATHS)} images for testing:")
for p in TEST_IMAGE_PATHS:
    print(f" - {p}")

TEST_PROMPT = 'You are an OCR engine. Your only job is to transcribe every word, number, and symbol visible in this handwritten image, exactly as written.\n\nRules:\n1. Output ONLY the transcribed text. No explanations, no commentary, no markdown formatting.\n2. Preserve the original language. Do NOT translate anything.\n3. Preserve line breaks, bullet points, numbering, and indentation as they appear.\n4. If a word is unclear, write your best guess followed by [?]. If completely unreadable, write [ILLEGIBLE].\n5. Transcribe ALL text including headers, footnotes, and margin notes.\n\nBegin transcription now:'


"""
================================================================================
KAGGLE CELL 2: The Qwen Series testing function
Alibaba's SOTA models. We load them in 4-bit to fit the Kaggle T4.
================================================================================
"""
def test_qwen_family(image_paths, model_id="Qwen/Qwen3.5-0.8B"):
    print(f"\n--- Loading {model_id} ---")

    # T4 GPUs only support float16, not bfloat16!
    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16, "attn_implementation": "eager"}

    # Quantize large models (7B / 8B+) to 4-bit to fit in T4 VRAM
    # Note: exclude '0.8B' and '4B' from the large-model check
    is_large_model = "7B" in model_id or ("8B" in model_id and "0.8B" not in model_id)

    if is_large_model:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        load_kwargs["quantization_config"] = quantization_config

    # Always try multimodal loader first; fall back to CausalLM if the model
    # doesn't have an image-text-to-text config (e.g. pure text checkpoints)
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **load_kwargs)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    from qwen_vl_utils import process_vision_info
    for i, img_path in enumerate(image_paths):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}", "max_pixels": 1003520},
                    {"type": "text", "text": TEST_PROMPT},
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
            return_tensors="pt"
        )
        inputs = {k: v.to(torch.float16).to("cuda") if v.is_floating_point() else v.to("cuda") for k, v in inputs.items()}
        inputs.pop("mm_token_type_ids", None)  # Qwen3.x injects this; not accepted by all model variants
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"{model_id} Output for Image {i+1}:\n", output_text[0])
        del inputs, generated_ids, generated_ids_trimmed, output_text
        torch.cuda.empty_cache()

    del model, processor
    clear_vram()


test_qwen_family(TEST_IMAGE_PATHS, "Qwen/Qwen3.5-0.8B")

test_qwen_family(TEST_IMAGE_PATHS, "Qwen/Qwen3.5-4B")