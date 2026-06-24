"""
Model Registry — lazy-load one model at a time, run inference, unload on switch.
Wraps the exact inference code from the existing project scripts.
"""

import gc
import os
import sys
import io
import json
import base64
import traceback

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure UTF-8 stdout (Windows console fix)
# ---------------------------------------------------------------------------
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Check optional backends at import time
# ---------------------------------------------------------------------------
try:
    import importlib
    importlib.import_module("paddleocr")
    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Catalogue of every model the dashboard can offer
# ---------------------------------------------------------------------------
MODEL_CATALOGUE = {
    "qwen2.5-vl-3b": {
        "name": "Qwen2.5-VL-3B-Instruct",
        "id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "group": "Vision Models (HuggingFace)",
        "backend": "transformers",
        "description": "Best overall — strong structure & pixel accuracy.",
        "default_prompt": (
            "Extract all the text from this image exactly as written. "
            "Do not add any conversational text or formatting."
        ),
    },
    "glm-ocr": {
        "name": "GLM-OCR 0.9B",
        "id": "zai-org/GLM-OCR",
        "group": "Vision Models (HuggingFace)",
        "backend": "transformers",
        "description": "Lightweight doc-understanding model (may hallucinate on cursive).",
        "default_prompt": (
            "You are an ultra-strict, pixel-literal OCR Engine. "
            "Transcribe the exact characters present in the image. "
            "Do not hallucinate."
        ),
    },
    "minicpm-v": {
        "name": "MiniCPM-V 2.6",
        "id": "openbmb/MiniCPM-V-2_6",
        "group": "Vision Models (HuggingFace)",
        "backend": "transformers",
        "description": "OpenBMB multimodal model.",
        "default_prompt": (
            "You are an ultra-strict, pixel-literal OCR Engine. "
            "Transcribe the exact characters present in the image."
        ),
    },
    "paligemma": {
        "name": "PaliGemma 3B",
        "id": "google/paligemma-3b-mix-448",
        "group": "Vision Models (HuggingFace)",
        "backend": "transformers",
        "description": "Google PaliGemma — uses task string 'ocr'.",
        "default_prompt": "ocr",
    },
    "paddleocr-vl": {
        "name": "PaddleOCR-VL",
        "id": "paddleocr-vl",
        "group": "PaddlePaddle",
        "backend": "paddle",
        "description": "PaddleOCR vision-language pipeline (v1.6).",
        "default_prompt": "Formula Recognition:",
        "available": _PADDLE_AVAILABLE,
    },
    "ollama-qwen": {
        "name": "Qwen2.5-VL 7B (Ollama)",
        "id": "qwen2.5vl:7b",
        "group": "Ollama (Local API)",
        "backend": "ollama",
        "description": "Larger 7B Qwen via local Ollama server.",
        "default_prompt": (
            "You are an expert transcriber. Read the handwritten text carefully "
            "and output ONLY the transcribed text. Preserve original line breaks, "
            "punctuation, and formatting as accurately as possible."
        ),
    },
    "got-ocr-2": {
        "name": "GOT-OCR 2.0",
        "id": "stepfun-ai/GOT-OCR-2.0-hf",
        "group": "Vision Models (HuggingFace)",
        "backend": "transformers",
        "description": "Stepfun-ai OCR model — no chat template, direct image input.",
        "default_prompt": "",
    },
    "chandra-2": {
        "name": "Chandra OCR 2",
        "id": "datalab-to/chandra-ocr-2",
        "group": "Vision Models (HuggingFace)",
        "backend": "transformers",
        "description": "SOTA 4B OCR — Markdown/JSON extraction, 90+ languages.",
        "default_prompt": "Extract all text and structure:",
    },
    "locateanything": {
        "name": "LocateAnything 3B",
        "id": "nvidia/LocateAnything-3B",
        "group": "Vision Models (HuggingFace)",
        "backend": "transformers",
        "description": "NVIDIA scene text detection & object grounding.",
        "default_prompt": "Detect all the text in box format.",
    },
    "qwen2-vl-2b": {
        "name": "Qwen2-VL-2B-Instruct",
        "id": "Qwen/Qwen2-VL-2B-Instruct",
        "group": "Vision Models (HuggingFace)",
        "backend": "transformers",
        "description": "Older Qwen2-VL 2B variant — lighter than Qwen2.5.",
        "default_prompt": (
            "Extract all handwritten text verbatim and format it."
        ),
    },
    "qwen3-vl-8b": {
        "name": "Qwen3-VL-8B-Instruct",
        "id": "Qwen/Qwen3-VL-8B-Instruct",
        "group": "Vision Models (HuggingFace)",
        "backend": "transformers",
        "description": "Latest Qwen3-VL — uses base64 image_url encoding.",
        "default_prompt": (
            "Extract all the text from this image exactly as written. "
            "Do not add any conversational text or formatting."
        ),
    },
}

# ---------------------------------------------------------------------------
# NuExtract defaults
# ---------------------------------------------------------------------------
DEFAULT_SCHEMA = json.dumps(
    {
        "subject": "",
        "concepts": [{"concept_name": "", "definition": ""}],
        "departments": [""],
    },
    indent=2,
)

# ---------------------------------------------------------------------------
# Image enhancement (from pipeline_main.py)
# ---------------------------------------------------------------------------

def enhance_image(pil_img: Image.Image) -> Image.Image:
    """Apply adaptive-threshold + dilation to sharpen text."""
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return Image.fromarray(dilated)


# ===================================================================
# Runtime — singleton model manager
# ===================================================================

class _ModelManager:
    """Loads one model at a time; frees GPU before switching."""

    def __init__(self):
        self._loaded_key: str | None = None
        self._model = None
        self._processor = None  # or tokenizer, depending on model
        self._extra = {}        # any extra objects (e.g. second tokenizer)

    # ------------------------------------------------------------------
    @property
    def loaded_key(self):
        return self._loaded_key

    # ------------------------------------------------------------------
    def unload(self):
        """Free the currently loaded model."""
        if self._model is not None:
            del self._model
        if self._processor is not None:
            del self._processor
        for v in self._extra.values():
            del v
        self._extra = {}
        self._model = None
        self._processor = None
        self._loaded_key = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def load(self, model_key: str):
        """Load the requested model (unload current first)."""
        if model_key == self._loaded_key:
            return  # already loaded

        meta = MODEL_CATALOGUE.get(model_key)
        if meta is None:
            raise ValueError(f"Unknown model key: {model_key}")

        self.unload()

        backend = meta["backend"]
        if backend == "ollama":
            # Nothing to load locally — Ollama is an HTTP API
            self._loaded_key = model_key
            return

        if backend == "paddle":
            self._load_paddle(meta)
        else:
            self._load_transformers(model_key, meta)

        self._loaded_key = model_key

    # ------------------------------------------------------------------
    # Transformers loaders
    # ------------------------------------------------------------------
    def _load_transformers(self, key: str, meta: dict):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model_id = meta["id"]

        if key == "qwen2.5-vl-3b":
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, device_map=device
            )
            self._processor = AutoProcessor.from_pretrained(model_id)

        elif key == "glm-ocr":
            from transformers import AutoProcessor, GlmOcrForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = GlmOcrForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype
            ).to(device)

        elif key == "minicpm-v":
            from transformers import AutoModel, AutoTokenizer

            self._model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=dtype, device_map=device
            )
            self._model.eval()
            self._processor = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )

        elif key == "paligemma":
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

            self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, device_map=device
            )
            self._processor = AutoProcessor.from_pretrained(model_id)

        elif key == "got-ocr-2":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map=device,
                torch_dtype=dtype,
                attn_implementation="eager",
            ).eval()

        elif key == "chandra-2":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).eval()
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        elif key == "locateanything":
            from transformers import AutoModel, AutoProcessor

            self._model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map=device,
                torch_dtype=dtype,
            ).eval()
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        elif key == "qwen2-vl-2b":
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True
            )
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        elif key == "qwen3-vl-8b":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            ).eval()
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # ------------------------------------------------------------------
    def _load_paddle(self, meta: dict):
        if not _PADDLE_AVAILABLE:
            raise RuntimeError(
                "PaddleOCR is not installed. "
                "Install it with: pip install paddlepaddle paddleocr  "
                "(may conflict with PyTorch on Windows — see README)."
            )
        from paddleocr import PaddleOCRVL

        try:
            self._model = PaddleOCRVL(pipeline_version="v1.6")
        except ValueError:
            self._model = PaddleOCRVL(pipeline_version="v1.5")
        self._processor = None

    # ------------------------------------------------------------------
    # NuExtract loader (separate, used in pipeline)
    # ------------------------------------------------------------------
    def load_nuextract(self):
        """Load NuExtract-tiny for JSON extraction."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        nid = "numind/NuExtract-tiny"

        tokenizer = AutoTokenizer.from_pretrained(nid, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            nid, torch_dtype=dtype, trust_remote_code=True, device_map=device
        )
        self._extra["nuextract_model"] = model
        self._extra["nuextract_tokenizer"] = tokenizer
        return model, tokenizer

    # ------------------------------------------------------------------
    # Inference dispatch
    # ------------------------------------------------------------------
    def run_ocr(self, model_key: str, image: Image.Image, prompt: str) -> str:
        """Run OCR with the specified model and return raw text."""
        self.load(model_key)
        meta = MODEL_CATALOGUE[model_key]
        backend = meta["backend"]

        if backend == "ollama":
            return self._infer_ollama(meta, image, prompt)
        elif backend == "paddle":
            return self._infer_paddle(image, prompt)
        else:
            return self._infer_transformers(model_key, image, prompt)

    # ------------------------------------------------------------------
    def _infer_transformers(self, key: str, image: Image.Image, prompt: str) -> str:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if key == "qwen2.5-vl-3b":
            return self._infer_qwen(image, prompt, device)
        elif key == "glm-ocr":
            return self._infer_glm(image, prompt, device)
        elif key == "minicpm-v":
            return self._infer_minicpm(image, prompt)
        elif key == "paligemma":
            return self._infer_paligemma(image, prompt, device)
        elif key == "got-ocr-2":
            return self._infer_got_ocr(image, device)
        elif key == "chandra-2":
            return self._infer_chandra(image, prompt, device)
        elif key == "locateanything":
            return self._infer_locateanything(image, prompt, device)
        elif key in ("qwen2-vl-2b", "qwen3-vl-8b"):
            return self._infer_qwen(image, prompt, device)  # same base64 pattern
        else:
            raise ValueError(f"No inference impl for {key}")

    # ------------------------------------------------------------------
    def _infer_qwen(self, image: Image.Image, prompt: str, device: str) -> str:
        """Qwen2.5-VL and Qwen3-VL — base64 PNG data URI via image_url."""
        # Encode PIL image to base64 PNG in memory (works for all Qwen VL variants)
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(device)

        generated = self._model.generate(**inputs, max_new_tokens=2048)
        trimmed = [
            o[len(i):] for i, o in zip(inputs.input_ids, generated)
        ]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

    # ------------------------------------------------------------------
    def _infer_glm(self, image: Image.Image, prompt: str, device: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        output = self._model.generate(**inputs, max_new_tokens=2048)
        return self._processor.decode(
            output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )

    # ------------------------------------------------------------------
    def _infer_minicpm(self, image: Image.Image, prompt: str) -> str:
        msgs = [{"role": "user", "content": [image, prompt]}]
        return self._model.chat(image=None, msgs=msgs, tokenizer=self._processor)

    # ------------------------------------------------------------------
    def _infer_paligemma(self, image: Image.Image, prompt: str, device: str) -> str:
        import torch

        inputs = self._processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=2048)
        result = self._processor.decode(output[0], skip_special_tokens=True)
        if result.startswith(prompt):
            result = result[len(prompt):].strip()
        return result

    # ------------------------------------------------------------------
    def _infer_got_ocr(self, image: Image.Image, device: str) -> str:
        """GOT-OCR 2.0 — no chat template, pass image directly."""
        import torch

        inputs = self._processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self._processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        return self._processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    # ------------------------------------------------------------------
    def _infer_chandra(self, image: Image.Image, prompt: str, device: str) -> str:
        """Chandra OCR 2 — uses chat template like Qwen-based models."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=image, padding=True, return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=2048)

        return self._processor.decode(outputs[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    def _infer_locateanything(self, image: Image.Image, prompt: str, device: str) -> str:
        """LocateAnything 3B — Qwen-VL style processor."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=image, padding=True, return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=2048)

        return self._processor.decode(outputs[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    def _infer_paddle(self, image: Image.Image, prompt: str) -> str:
        import glob
        import shutil

        tmp = os.path.join(os.path.dirname(__file__), "uploads", "_paddle_tmp.png")
        image.save(tmp)

        output = self._model.predict(tmp)
        if not output:
            return "[Error] PaddleOCR-VL returned no output."

        res = output[0]
        extracted = ""
        if hasattr(res, "markdown"):
            extracted = res.markdown
        elif isinstance(res, dict) and "markdown" in res:
            extracted = res["markdown"]

        if not extracted:
            temp_dir = os.path.join(os.path.dirname(__file__), "uploads", "_paddle_md")
            os.makedirs(temp_dir, exist_ok=True)
            try:
                res.save_to_markdown(temp_dir)
                md_files = glob.glob(os.path.join(temp_dir, "*.md"))
                if md_files:
                    with open(md_files[0], "r", encoding="utf-8") as f:
                        extracted = f.read().strip()
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return extracted or "[Error] No text extracted."

    # ------------------------------------------------------------------
    def _infer_ollama(self, meta: dict, image: Image.Image, prompt: str) -> str:
        import requests
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            import ollama as ollama_lib

            response = ollama_lib.chat(
                model=meta["id"],
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "Please transcribe the handwritten text in this image.",
                        "images": [img_b64],
                    },
                ],
            )
            return response["message"]["content"]
        except Exception:
            # Fallback: raw HTTP
            payload = {
                "model": meta["id"],
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "Please transcribe the handwritten text in this image.",
                        "images": [img_b64],
                    },
                ],
                "stream": False,
            }
            r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["message"]["content"]

    # ------------------------------------------------------------------
    # NuExtract inference
    # ------------------------------------------------------------------
    def run_nuextract(self, text: str, schema: str) -> str:
        """Run NuExtract-tiny on text with the given JSON schema."""
        import torch

        if "nuextract_model" not in self._extra:
            self.load_nuextract()

        model = self._extra["nuextract_model"]
        tokenizer = self._extra["nuextract_tokenizer"]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        schema_str = json.dumps(json.loads(schema), indent=4)
        input_llm = (
            "<|input|>\n### Template:\n" + schema_str +
            "\n### Text:\n" + text + "\n<|output|>\n"
        )

        input_ids = tokenizer(
            input_llm, return_tensors="pt", truncation=True, max_length=4000
        ).to(device)
        output = tokenizer.decode(
            model.generate(**input_ids, max_new_tokens=1000)[0],
            skip_special_tokens=True,
        )

        try:
            return output.split("<|output|>")[1].split("<|end-")[0]
        except Exception:
            return output


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
manager = _ModelManager()
