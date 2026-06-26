# =============================================================================
# VLM OCR Kaggle Demo — Final Version
# INSTRUCTIONS:
#   Step 1: Paste CELL 1 into a Kaggle cell. Run it. Wait for the print.
#   Step 2: RESTART SESSION  (Runtime → Restart Session)
#   Step 3: Paste CELL 2 into a new cell. Run it.
# =============================================================================


# ════════════════════════════════════════════════════════════════════
# CELL 1 — INSTALLS  (run this once, then RESTART SESSION)
# ════════════════════════════════════════════════════════════════════
import subprocess, sys

# ── 0. Capture the current torch / torchvision versions BEFORE anything else
#        so we can pin them as constraints (prevents pip from downgrading).
import torch as _t, torchvision as _tv
with open("/kaggle/working/constraints.txt", "w") as _f:
    _f.write(f"torch=={_t.__version__}\n")
    _f.write(f"torchvision=={_tv.__version__}\n")
del _t, _tv

# ── 1. Block torchaudio before anything else (CUDA mismatch crash)
try:
    import torchaudio as _ta_test  # noqa: F401
except Exception:
    sys.modules["torchaudio"] = None  # type: ignore[assignment]
    print("[shim] Blocked broken torchaudio")

# ── 2. Pin stable Paddle versions (3.x has a known oneDNN PIR executor bug)
subprocess.run(
    ["pip", "uninstall", "-q", "-y", "paddlepaddle-gpu", "paddlepaddle", "paddleocr"],
    check=False,
)
subprocess.run(["pip", "install", "-q", "paddlepaddle==2.6.2", "paddleocr==2.8.1"], check=False)

# ── 3. Install / upgrade the rest  (constraints.txt keeps torch pinned)
for pkg in [
    "gradio",
    "nest_asyncio",
    "accelerate",
    "einops",
    "timm",
    "qwen-vl-utils",
    "bitsandbytes",
    "transformers",    # --upgrade handled by constraint file
    "huggingface_hub",
]:
    subprocess.run(
        ["pip", "install", "-q", "--upgrade", "-c", "/kaggle/working/constraints.txt", pkg],
        check=False,
    )

print("✅ All packages installed.  RESTART SESSION now, then run Cell 2.")


# ════════════════════════════════════════════════════════════════════
# CELL 2 — MAIN SCRIPT  (run after restarting session)
# ════════════════════════════════════════════════════════════════════
import os, gc, sys, json, re, traceback, importlib, subprocess
import torch
import gradio as gr
import nest_asyncio
nest_asyncio.apply()

from PIL import Image

# ── Runtime shim 1: Pillow _Ink ImportError (torchvision ↔ Pillow mismatch) ──
try:
    _pil_typing = importlib.import_module("PIL._typing")
    if not hasattr(_pil_typing, "_Ink"):
        _pil_typing._Ink = str | float | tuple[int, ...]
        print("[shim] Patched PIL._typing._Ink")
except ImportError:
    pass

# ── Runtime shim 2: Block torchaudio to prevent CUDA crash ───────────────────
try:
    import torchaudio as _ta_test  # noqa: F401
except Exception:
    sys.modules["torchaudio"] = None  # type: ignore[assignment]
    print("[shim] Blocked broken torchaudio")

# ── Imports from transformers (done AFTER shims) ─────────────────────────────
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)

# ── Runtime shim 3: fix GenerationConfig.from_model_config dict bug ──────────
# transformers < 4.53 calls model_config.to_dict() without checking isinstance.
# The fix must use a FACTORY FUNCTION so that _orig is a proper closure variable
# (captured at definition time), NOT a module-level global that gets deleted.
def _install_gen_config_patch():
    from transformers.generation.configuration_utils import GenerationConfig
    
    @classmethod
    def _safe_fmc(cls, model_config=None, **kwargs):
        # Break the chain of previous patches by NOT calling _orig.
        # This completely side-steps the AttributeError and NoneType bugs.
        config_dict = {}
        if isinstance(model_config, dict):
            config_dict = model_config
        elif hasattr(model_config, "to_dict"):
            try:
                config_dict = model_config.to_dict()
            except Exception:
                pass
        
        # Extract basic generation parameters to form a valid GenerationConfig
        gen_kwargs = {
            k: v for k, v in config_dict.items() 
            if k in ["bos_token_id", "eos_token_id", "pad_token_id", "is_encoder_decoder"]
        }
        return cls(**gen_kwargs)

    GenerationConfig.from_model_config = _safe_fmc
    print("[shim] Replaced GenerationConfig.from_model_config entirely to bypass dict bugs")

_install_gen_config_patch()
del _install_gen_config_patch


# ── Model registry ────────────────────────────────────────────────────────────
#   kind:
#     qwen3vl     Qwen3-VL vision-language model (AutoModelForImageTextToText)
#     qwen3text   Qwen3.5 text-capable VL model  (AutoModelForImageTextToText
#                                                   → fallback AutoModelForCausalLM)
#     glm_ocr     GLM-OCR (GlmOcrForConditionalGeneration → fallback CausalLM)
#     chandra     Chandra 2 (AutoModelForImageTextToText, trust_remote_code)
#     paddle      PaddleOCR — always runs in a subprocess to avoid CUDA conflict
MODEL_REGISTRY = {
    "Qwen3-VL-8B":  {"repo": "Qwen/Qwen3-VL-8B-Instruct",   "kind": "qwen3vl"},
    "GLM-OCR":      {"repo": "zai-org/GLM-OCR",               "kind": "glm_ocr"},
    "Chandra 2":    {"repo": "datalab-to/chandra-ocr-2",       "kind": "chandra"},
    "Qwen3.5-4B":   {"repo": "Qwen/Qwen3.5-4B",               "kind": "qwen3text"},
    "Qwen3.5-0.8B": {"repo": "Qwen/Qwen3.5-0.8B",             "kind": "qwen3text"},
    "PaddleOCR":    {"repo": None,                              "kind": "paddle"},
}

# ── Global model state ────────────────────────────────────────────────────────
_current_name = None
_model        = None
_processor    = None

def _unload():
    global _model, _processor, _current_name
    if _model     is not None: del _model
    if _processor is not None: del _processor
    _model = _processor = _current_name = None
    gc.collect()
    torch.cuda.empty_cache()


# ── Standard load kwargs (T4 GPUs only support float16, not bfloat16) ─────────
_BASE_KWARGS = {
    "device_map":          "auto",
    "dtype":               torch.float16,   # 'torch_dtype' deprecated in newer transformers
    "attn_implementation": "eager",         # avoids SDPA tensor-shape mismatches on T4
}


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(name: str):
    global _current_name, _model, _processor

    if _current_name == name:
        return
    _unload()
    print(f"▶ Loading {name} …")

    entry = MODEL_REGISTRY[name]
    kind  = entry["kind"]
    repo  = entry["repo"]

    try:
        # ── PaddleOCR ─────────────────────────────────────────────────────────
        # PaddleOCR is run in a subprocess at inference time to avoid CUDA
        # conflicts with PyTorch. Nothing to "load" here.
        if kind == "paddle":
            _current_name = name
            print(f"✅ {name} ready (will run via subprocess)")
            return

        # ── Qwen3-VL-8B ───────────────────────────────────────────────────────
        elif kind == "qwen3vl":
            _processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
            _model = AutoModelForImageTextToText.from_pretrained(
                repo, trust_remote_code=True, **_BASE_KWARGS
            ).eval()

        # ── Qwen3.5 (0.8B / 4B) ──────────────────────────────────────────────
        # These are text-capable VL models.  AutoModelForImageTextToText will
        # fail for pure text configs; fall back to AutoModelForCausalLM which
        # picks up Qwen3ForCausalLM cleanly.  attn_implementation="eager" is
        # required to avoid SDPA errors on Kaggle T4 GPUs.
        elif kind == "qwen3text":
            _processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
            try:
                _model = AutoModelForImageTextToText.from_pretrained(
                    repo, trust_remote_code=True, **_BASE_KWARGS
                ).eval()
            except Exception as e_vl:
                print(f"  [info] ImageTextToText failed ({e_vl!r}), trying CausalLM …")
                _model = AutoModelForCausalLM.from_pretrained(
                    repo, trust_remote_code=True, **_BASE_KWARGS
                ).eval()

        # ── GLM-OCR ───────────────────────────────────────────────────────────
        # Prefer the dedicated GlmOcrForConditionalGeneration class (available
        # in transformers ≥ 4.52).  Fall back to AutoModelForCausalLM if the
        # class isn't registered yet in the installed version.
        elif kind == "glm_ocr":
            _processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
            try:
                from transformers import GlmOcrForConditionalGeneration
                _model = GlmOcrForConditionalGeneration.from_pretrained(
                    repo,
                    torch_dtype=torch.float16,
                    attn_implementation="eager",
                ).to("cuda").eval()
            except (ImportError, Exception) as e_glm:
                print(f"  [info] GlmOcrForConditionalGeneration failed ({e_glm!r}), trying CausalLM …")
                _model = AutoModelForCausalLM.from_pretrained(
                    repo, trust_remote_code=True, **_BASE_KWARGS
                ).eval()

        # ── Chandra 2 ─────────────────────────────────────────────────────────
        # Chandra 2 ships its own Python source via trust_remote_code.
        # Load with AutoModelForImageTextToText, same as the Vision Suite notebook.
        elif kind == "chandra":
            _processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
            _model = AutoModelForImageTextToText.from_pretrained(
                repo, trust_remote_code=True, **_BASE_KWARGS
            ).eval()

        _current_name = name
        print(f"✅ {name} loaded successfully")

    except Exception:
        print(f"❌ Failed to load {name}:\n{traceback.format_exc()}")
        _unload()


# ── OCR prompt ────────────────────────────────────────────────────────────────
PROMPT = (
    "You are a strict OCR engine. Transcribe every character visible in this image "
    "exactly as written. Do NOT correct, translate, or summarise. Preserve layout."
)


# ── JSON builder ──────────────────────────────────────────────────────────────
def _build_json(model_name: str, raw: str) -> str:
    """
    Returns structured JSON with sentence-wise and row-wise splits.
    No full_text blob — keeps the output clean and readable.
    """
    is_err    = raw.lower().startswith(("error", "inference error", "model ", "❌", "⚠"))
    lines     = [l.strip() for l in raw.split("\n") if l.strip()]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", raw) if s.strip()]
    return json.dumps(
        {
            "model":      model_name,
            "status":     "error" if is_err else "success",
            "line_count": len(lines),
            "lines":      lines,
            "sentences":  sentences,
        },
        indent=2,
        ensure_ascii=False,
    )


# ── PaddleOCR subprocess helper ───────────────────────────────────────────────
def _run_paddle_subprocess(image_path: str) -> str:
    """
    Runs PaddleOCR 2.8.1 in an isolated subprocess to avoid CUDA conflicts.
    Uses the old 2.x API: ocr.ocr(path, cls=True)
    Result format: result[0] = list of [bbox, (text, confidence)]
    """
    script = f"""
import sys
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
    result = ocr.ocr({repr(image_path)}, cls=True)
    if result and result[0]:
        for line in result[0]:
            # line = [bbox_points, (text_string, confidence_score)]
            text = line[1][0]
            print(text)
    else:
        print("[no text detected]")
except Exception as e:
    import traceback
    print("PADDLE_ERROR: " + str(e), file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0 and proc.stderr:
            return f"PaddleOCR subprocess error:\n{proc.stderr[-2000:]}"
        return proc.stdout.strip()
    except subprocess.TimeoutExpired:
        return "PaddleOCR Error: subprocess timed out after 120 s"
    except Exception:
        return f"PaddleOCR Error:\n{traceback.format_exc()}"


# ── Shared Qwen inference helper (qwen_vl_utils path) ────────────────────────
def _qwen_infer(image: Image.Image, model_name: str) -> str:
    """
    Unified inference for all Qwen variants using the qwen_vl_utils pipeline.
    Falls back to a simple processor call if qwen_vl_utils is unavailable or if
    the loaded model is a pure-text LLM (like Qwen3.5-0.8B).
    """
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text":  PROMPT},
    ]}]
    try:
        text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Pure text models will reject dict-content with images
        messages = [{"role": "user", "content": PROMPT}]
        text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = _processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Move tensors to device; keep float16 for floating-point tensors
        inputs = {
            k: v.to(torch.float16).to(_model.device) if v.is_floating_point() else v.to(_model.device)
            for k, v in inputs.items()
        }
        inputs.pop("mm_token_type_ids", None)  # injected by Qwen3.x; not accepted by all variants
    except Exception:
        # qwen_vl_utils not installed or processor rejected images
        try:
            inputs = _processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            ).to(_model.device)
        except Exception:
            # Fallback for text-only models
            inputs = _processor(text=[text], padding=True, return_tensors="pt").to(_model.device)

    # Extract EOS tokens to prevent run-on generation
    tokenizer = getattr(_processor, "tokenizer", _processor)
    terminators = []
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            terminators.extend(tokenizer.eos_token_id)
        else:
            terminators.append(tokenizer.eos_token_id)
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and isinstance(im_end, int):
            terminators.append(im_end)
    if not terminators:
        terminators = None

    try:
        with torch.no_grad():
            out = _model.generate(**inputs, max_new_tokens=2048, do_sample=False, eos_token_id=terminators)
    except ValueError as e:
        if "not used by the model" in str(e):
            # Text-only models will reject vision kwargs like pixel_values.
            # Strip them and retry with just text inputs.
            text_inputs = {
                k: v for k, v in inputs.items() 
                if k in ["input_ids", "attention_mask", "position_ids"]
            }
            with torch.no_grad():
                out = _model.generate(**text_inputs, max_new_tokens=2048, do_sample=False, eos_token_id=terminators)
        else:
            raise

    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    return _processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(image_path: str, model_name: str):
    """Returns (raw_text, json_string)."""
    if image_path is None:
        err = "Error: please upload an image first."
        return err, _build_json(model_name, err)

    load_model(model_name)

    entry = MODEL_REGISTRY[model_name]
    kind  = entry["kind"]

    if _model is None and kind != "paddle":
        err = f"Model {model_name} could not be loaded — see logs above."
        return err, _build_json(model_name, err)

    raw = ""
    try:

        # ── PaddleOCR (subprocess — avoids CUDA conflicts) ────────────────────
        if kind == "paddle":
            raw = _run_paddle_subprocess(image_path)

        # ── Qwen3-VL-8B ───────────────────────────────────────────────────────
        elif kind == "qwen3vl":
            image = Image.open(image_path).convert("RGB")
            raw   = _qwen_infer(image, model_name)

        # ── Qwen3.5 (0.8B / 4B) — VL or text fallback ───────────────────────
        elif kind == "qwen3text":
            image = Image.open(image_path).convert("RGB")
            raw   = _qwen_infer(image, model_name)

        # ── GLM-OCR ───────────────────────────────────────────────────────────
        elif kind == "glm_ocr":
            image = Image.open(image_path).convert("RGB")
            # GLM-OCR eager attention is O(N^2) — cap resolution to prevent T4 OOM
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  PROMPT},
            ]}]
            inputs = _processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(_model.device)
            with torch.no_grad():
                out = _model.generate(**inputs, max_new_tokens=2048)
            start = inputs["input_ids"].shape[-1]
            raw = _processor.decode(out[0][start:], skip_special_tokens=True)

        # ── Chandra 2 ─────────────────────────────────────────────────────────
        elif kind == "chandra":
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  "Extract all text and structure:"},
            ]}]
            text   = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = _processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = {
                k: v.to(torch.float16).to(_model.device) if v.is_floating_point() else v.to(_model.device)
                for k, v in inputs.items()
            }
            with torch.no_grad():
                out = _model.generate(**inputs, max_new_tokens=2048)
            trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
            raw = _processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    except Exception:
        raw = f"Inference Error:\n{traceback.format_exc()}"

    return raw, _build_json(model_name, raw)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="VLM OCR Demo") as demo:
    gr.Markdown("## 🔍 VLM OCR Evaluation Playground")
    gr.Markdown(
        "Upload a document image, pick a model and hit **Extract**. "
        "Models load on-demand — the previous model is evicted from VRAM automatically."
    )
    with gr.Row():
        with gr.Column(scale=1):
            img_input      = gr.Image(type="filepath", label="Document Image")
            model_selector = gr.Dropdown(
                choices=list(MODEL_REGISTRY.keys()),
                value="Qwen3-VL-8B",
                label="Model / Engine",
            )
            run_btn = gr.Button("Extract Text ▶", variant="primary")
        with gr.Column(scale=1):
            raw_out  = gr.Textbox(label="Raw Text Output", lines=12)
            json_out = gr.Textbox(label="JSON Output",      lines=12)

    run_btn.click(
        fn=run_inference,
        inputs=[img_input, model_selector],
        outputs=[raw_out, json_out],
    )

print("🚀 Starting Gradio — public URL will appear below:")
demo.launch(share=True, inline=False, theme=gr.themes.Monochrome())
