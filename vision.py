"""
================================================================================
KAGGLE CELL 1: Environment Setup & Installations
Note: We explicitly DO NOT upgrade PyTorch to prevent CUDA driver mismatch errors on Kaggle T4s.
Note: We DO NOT upgrade or huggingface_hub to prevent ImportErrors.
================================================================================
"""
import torch, torchvision
with open("constraints.txt", "w") as f:
    f.write(f"torch=={torch.__version__}\n")
    f.write(f"torchvision=={torchvision.__version__}\n")

!pip install -c constraints.txt transformers accelerate bitsandbytes timm einops verovio --upgrade
# Align Pillow after transformers to fix _Ink version mismatch
!pip install "Pillow>=11.2.1" --upgrade -q
# Remove torchaudio to avoid CUDA version mismatch (not needed for vision)
!pip uninstall torchaudio -y -q 2>/dev/null || true
# PaddleOCR: pin compatible versions to avoid set_optimization_level errors
# Pin to paddlepaddle 2.6.2 + paddleocr 2.8.1.
# paddlepaddle 3.x introduced a new PIR executor with a known oneDNN bug
# (ConvertPirAttribute2RuntimeAttribute not support ArrayAttribute<DoubleAttribute>)
# that no env flag can disable. The 2.6.2 + 2.8.1 combo is stable on Kaggle.
!pip uninstall paddlepaddle paddlepaddle-gpu paddleocr -y -q 2>/dev/null || true
!pip install paddlepaddle==2.6.2 paddleocr==2.8.1 -q
# LocateAnything needs qwen_vl_utils
!pip install qwen-vl-utils decord lmdb -q


"""
================================================================================
KAGGLE CELL 2: Hugging Face Authentication
Many of these models (PaliGemma, MiniCPM) are gated. You MUST accept their 
license on huggingface.co and log in here using a token with read access.
================================================================================
"""
from huggingface_hub import login
import os
from kaggle_secrets import UserSecretsClient

try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("Successfully logged in via Kaggle Secrets!")
except:
    print("Could not find HF_TOKEN in Kaggle Secrets.")
    print("Please log in manually:")
    login()

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
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModel, AutoTokenizer

def clear_vram():
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

images = []
for p in TEST_IMAGE_PATHS:
    try:
        img = Image.open(p).convert("RGB")
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        images.append(img)
    except Exception as e:
        print(f"Failed to load {p}: {e}")

# Fix 3: Monkey-patch transformers for older remote code models (LocateAnything)
# Newer transformers expect `_tied_weights_keys` to be a dict, but older remote code uses a list.
import transformers
from transformers.modeling_utils import PreTrainedModel
if hasattr(PreTrainedModel, "get_expanded_tied_weights_keys") and not getattr(PreTrainedModel, "_is_tied_weights_patched", False):
    _old_get_expanded = PreTrainedModel.get_expanded_tied_weights_keys
    def _patched_get_expanded(self, all_submodels=False):
        if hasattr(self, "_tied_weights_keys") and isinstance(self._tied_weights_keys, list):
            self._tied_weights_keys = {k: "model.embed_tokens.weight" for k in self._tied_weights_keys}
        return _old_get_expanded(self, all_submodels=all_submodels)
    PreTrainedModel.get_expanded_tied_weights_keys = _patched_get_expanded
    PreTrainedModel._is_tied_weights_patched = True
    print("[shim] Patched get_expanded_tied_weights_keys for legacy remote code compatibility")

# Fix 4: Patch PreTrainedModel.__getattr__ to dynamically supply all_tied_weights_keys
# Some older remote code models (like LocateAnything) forget to call post_init() in their __init__
if not getattr(PreTrainedModel, "_is_tied_weights_getattr_patched", False):
    _old_getattr = getattr(PreTrainedModel, "__getattr__", None)
    def _patched_getattr(self, name):
        if name == "all_tied_weights_keys":
            val = self.get_expanded_tied_weights_keys(all_submodels=False)
            self.all_tied_weights_keys = val
            return val
        if _old_getattr is not None:
            return _old_getattr(self, name)
        return super(PreTrainedModel, self).__getattr__(name)
    PreTrainedModel.__getattr__ = _patched_getattr
    PreTrainedModel._is_tied_weights_getattr_patched = True
    print("[shim] Patched __getattr__ to dynamically supply missing all_tied_weights_keys")

# Fix 5: Patch DynamicCache.to_legacy_cache for older remote code models
import transformers.cache_utils
if hasattr(transformers.cache_utils, "DynamicCache") and not hasattr(transformers.cache_utils.DynamicCache, "to_legacy_cache"):
    def _to_legacy_cache(self):
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += ((layer.keys, layer.values),)
        return legacy_cache
    transformers.cache_utils.DynamicCache.to_legacy_cache = _to_legacy_cache
    print("[shim] Patched DynamicCache to add missing to_legacy_cache")

# Fix 6: Patch DynamicCache to add missing from_legacy_cache
# LocateAnything's custom modeling_qwen2.py calls DynamicCache.from_legacy_cache
# internally during generation, but this classmethod was removed in transformers >= 4.45
if hasattr(transformers.cache_utils, "DynamicCache") and not hasattr(transformers.cache_utils.DynamicCache, "from_legacy_cache"):
    @classmethod
    def _from_legacy_cache(cls, past_key_values=None):
        """Converts a legacy tuple-of-tuples cache into a DynamicCache."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
    transformers.cache_utils.DynamicCache.from_legacy_cache = _from_legacy_cache
    print("[shim] Patched DynamicCache to add missing from_legacy_cache")


"""
================================================================================
TEST 1: Chandra 2 (datalab-to/chandra-ocr-2)
================================================================================
"""
def test_chandra_2():
    print("\n--- Loading Chandra OCR 2 ---")
    model_id = "datalab-to/chandra-ocr-2"
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    for i, img in enumerate(images):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Extract all text and structure:"}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=img, padding=True, return_tensors="pt")
        inputs = {k: v.to(torch.float16).to("cuda") if v.is_floating_point() else v.to("cuda") for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=512)
            
        print(f"Chandra 2 Output {i+1}:\n", processor.decode(outputs[0], skip_special_tokens=True))
        del inputs, outputs
        torch.cuda.empty_cache()
        
    del model, processor
    clear_vram()

"""
================================================================================
TEST 2: LocateAnything (nvidia/LocateAnything-3B)
Uses the model's custom remote-code API for chat templating and vision processing.
================================================================================
"""
def test_locate_anything(target_object="handwritten text"):
    print("\n--- Loading NVIDIA LocateAnything ---")
    model_id = "nvidia/LocateAnything-3B"
    
    import transformers
    
    # Patch for allow_all_kernels kwarg incompatibility
    old_init = transformers.modeling_utils.PreTrainedModel.__init__
    def patched_init(self, config, *inputs, **kwargs):
        target_cls = None
        for base in self.__class__.__mro__:
            if '_check_and_adjust_attn_implementation' in base.__dict__:
                target_cls = base
                break
        if target_cls and target_cls.__name__ != 'PreTrainedModel':
            old_check = target_cls.__dict__['_check_and_adjust_attn_implementation']
            @classmethod
            def safe_check(cls, *args, **kw):
                kw.pop('allow_all_kernels', None)
                if isinstance(old_check, classmethod):
                    return old_check.__func__(cls, *args, **kw)
                else:
                    return old_check(cls, *args, **kw)
            setattr(target_cls, '_check_and_adjust_attn_implementation', safe_check)
            try:
                old_init(self, config, *inputs, **kwargs)
            finally:
                setattr(target_cls, '_check_and_adjust_attn_implementation', old_check)
        else:
            old_init(self, config, *inputs, **kwargs)
            
    transformers.modeling_utils.PreTrainedModel.__init__ = patched_init
    
    if not hasattr(transformers.models.qwen2.configuration_qwen2.Qwen2Config, "rope_theta"):
        transformers.models.qwen2.configuration_qwen2.Qwen2Config.rope_theta = 1000000.0
    
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            device_map="cuda:0", 
            torch_dtype=torch.float16
        ).eval()
        
        for i, img in enumerate(images):
            # Resize image to cap VRAM usage (LocateAnything uses dynamic token count)
            img_resized = img.copy()
            img_resized.thumbnail((800, 800))
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_resized},
                        {"type": "text", "text": f"Locate: {target_object}"}
                    ]
                }
            ]
            
            # Use the model's custom remote-code API (NOT qwen_vl_utils)
            text = processor.py_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = processor.process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                return_tensors="pt"
            ).to("cuda")
            
            pixel_values = inputs["pixel_values"].to(torch.float16)
            input_ids = inputs["input_ids"]
            image_grid_hws = inputs.get("image_grid_hws", None)
            
            with torch.inference_mode():
                response = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=image_grid_hws,
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    use_cache=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            # response may be a tuple (text, history, stats) or just text
            answer = response[0] if isinstance(response, tuple) else response
            print(f"LocateAnything Output for '{target_object}' Image {i+1}:\n", answer)
            
            del inputs, pixel_values, input_ids, img_resized
            torch.cuda.empty_cache()
            
        del model, processor, tokenizer
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error running LocateAnything inference: {e}")
    finally:
        transformers.modeling_utils.PreTrainedModel.__init__ = old_init
        clear_vram()


"""
================================================================================
TEST 3: MiniCPM-V 2.6 (openbmb/MiniCPM-V-2_6)
================================================================================
"""
def test_minicpm():
    import torch.nn as nn
    
    print("\n--- Loading MiniCPM-V 2.6 ---")
    model_id = "openbmb/MiniCPM-V-2_6"
    
    old_getattr = nn.Module.__getattr__
    def safe_getattr(self, name):
        if name == 'all_tied_weights_keys':
            return {}
        return old_getattr(self, name)
    nn.Module.__getattr__ = safe_getattr

    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    
    model = AutoModel.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        device_map={"": 0},  
        quantization_config=quantization_config,
        torch_dtype=torch.float16
    ).eval()
    
    nn.Module.__getattr__ = old_getattr
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    msgs = [{'role': 'user', 'content': 'Extract all text verbatim.'}]
    
    for i, img in enumerate(images):
        res = model.chat(
            image=img,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True, 
            temperature=0.7
        )
        print(f"MiniCPM Output {i+1}:\n", res)
        del res
        torch.cuda.empty_cache()
    
    del model, tokenizer
    clear_vram()

"""
================================================================================
TEST 4: PaliGemma 3B (google/paligemma-3b-pt-224)
================================================================================
"""
def test_paligemma():
    print("\n--- Loading PaliGemma 3B ---")
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    model_id = "google/paligemma-3b-pt-224"
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto").eval()
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    prompt = "ocr"  # PaliGemma relies on specific short task prompts like 'ocr' or 'caption'
    for i, img in enumerate(images):
        model_inputs = processor(text=prompt, images=img, return_tensors="pt")
        model_inputs = {k: v.to(torch.float16).to("cuda") if v.is_floating_point() else v.to("cuda") for k, v in model_inputs.items()}
        
        with torch.inference_mode():
            generation = model.generate(**model_inputs, max_new_tokens=256)
            # Remove prompt tokens from output
            generation = generation[0][model_inputs['input_ids'].shape[-1]:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            
        print(f"PaliGemma Output {i+1}:\n", decoded)
        del model_inputs, generation, decoded
        torch.cuda.empty_cache()
        
    del model, processor
    clear_vram()

"""
================================================================================
TEST 5: GOT-OCR (stepfun-ai/GOT-OCR-2.0-hf)
Uses HF-native AutoModelForImageTextToText. This model has NO chat template,
so we pass the image directly to the processor and generate.
================================================================================
"""
def test_got_ocr():
    print("\n--- Loading GOT-OCR 2.0 (HF-native) ---")
    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_id = "stepfun-ai/GOT-OCR-2.0-hf"
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",  # Avoid SDPA tensor shape mismatch
    ).eval()
    
    # GOT-OCR-2.0-hf has NO chat template. Pass image directly to processor.
    for i, img in enumerate(images):
        inputs = processor(images=img, return_tensors="pt").to("cuda")
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096,
            )
        
        prompt_len = inputs["input_ids"].shape[-1]
        result = processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        
        print(f"GOT-OCR Output {i+1}:\n", result)
        del inputs, outputs, result
        torch.cuda.empty_cache()
        
    del model, processor
    clear_vram()

"""
================================================================================
TEST 6: GLM-OCR (zai-org/GLM-OCR)
================================================================================
"""
def test_glm_ocr():
    print("\n--- Loading GLM-OCR ---")
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration
    model_id = "zai-org/GLM-OCR"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = GlmOcrForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to("cuda").eval()
    
    for i, img in enumerate(images):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Extract all text from this image."}
                ],
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=2048)
        
        result = processor.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        print(f"GLM-OCR Output {i+1}:\n", result)
        del inputs, output, result
        torch.cuda.empty_cache()
        
    del model, processor
    clear_vram()

"""
================================================================================
TEST 7: PaddleOCR (pinned to paddleocr==2.8.1 + paddlepaddle==2.6.2)
Runs in a completely isolated subprocess.
MUST reinstall using the install cell above before running.
================================================================================
"""
def test_paddle_ocr():
    print("\n--- Loading PaddleOCR (Isolated Subprocess, CPU-only) ---")
    import subprocess
    import os

    paths_file = "/kaggle/working/paddle_image_paths.txt"
    with open(paths_file, "w", encoding="utf-8") as pf:
        for p in TEST_IMAGE_PATHS:
            pf.write(p + "\n")

    # Uses old PaddleOCR 2.x API: ocr.ocr(path, cls=True)
    # Result format: result[0] = list of [bbox, (text, score)]
    script_content = """
import os
from paddleocr import PaddleOCR

paths_file = "/kaggle/working/paddle_image_paths.txt"
with open(paths_file, "r", encoding="utf-8") as f:
    TEST_IMAGE_PATHS = [line.strip() for line in f if line.strip()]

try:
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang="en", show_log=False)
    for i, img_path in enumerate(TEST_IMAGE_PATHS):
        if not os.path.exists(img_path):
            print("Skipping missing: " + img_path)
            continue
        print("PaddleOCR Output " + str(i+1) + ": " + img_path)
        result = ocr.ocr(img_path, cls=True)
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                score = line[1][1]
                print("[" + str(round(score, 3)) + "] " + text)
        else:
            print("(no text detected)")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("PaddleOCR failed: " + str(e))
"""

    script_path = "/kaggle/working/temp_paddle_test.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    try:
        result = subprocess.run(["python", script_path], capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR (last 3000 chars):")
            print(result.stderr[-3000:])
    finally:
        for fp in [script_path, paths_file]:
            if os.path.exists(fp):
                os.remove(fp)


# test_chandra_2()

# test_locate_anything("handwritten text")

# test_minicpm()

# test_paligemma()

# test_got_ocr()

# test_glm_ocr()

# test_paddle_ocr()