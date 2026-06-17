# Running OCR Experiments on Kaggle

This repository contains dedicated Jupyter notebooks tailored for execution in Kaggle's GPU environments. Due to Kaggle's specific pre-installed libraries and hardware quirks, these notebooks incorporate special workarounds for smooth inference.

## Available Notebooks

### 1. `Kaggle_Vision_Models_Suite.ipynb`
A multi-model testing suite designed to evaluate the latest specialized OCR and visual grounding models. 
*   **Models included:** 
    *   `datalab-to/chandra-ocr-2` (OCR)
    *   `nvidia/LocateAnything-3B` (Visual Grounding)
    *   `openbmb/MiniCPM-V-2_6` (Generalist VLM)
    *   `google/paligemma-3b-pt-224` (Task-specific Vision)

### 2. `Kaggle_Qwen_Family.ipynb`
A dedicated suite for testing the Qwen2.5-VL series, specifically analyzing its exceptional formatting and pixel-literal extraction capabilities on complex documents.

---

## 🛠️ Kaggle Environment Setup Guide

To successfully run these notebooks on Kaggle, you must configure your environment to handle hardware constraints and library conflicts.

### 1. Hardware Selection
*   **Accelerator:** Set your Kaggle session to use **GPU T4 x2**.
*   **VRAM Constraints:** Models like MiniCPM (8B) and LocateAnything (3B) are loaded in `float16` or 4-bit quantization to ensure they comfortably fit within the 15GB VRAM of a single T4 GPU.

### 2. Hugging Face Authentication
Several models (e.g., PaliGemma, MiniCPM) are gated. You must authenticate to download their weights.
1. Accept the model agreements on the Hugging Face website.
2. Go to your Kaggle Notebook's **Add-ons -> Secrets**.
3. Create a new secret named `HF_TOKEN`.
4. Paste your Hugging Face User Access Token (with read permissions) as the value.
*The notebooks are programmed to automatically pull this secret and log you in.*

### 3. Critical Library Quirks & Workarounds
Kaggle's persistent disk and pre-installed dependencies conflict with recent `transformers` features. The notebooks already include these fixes, but be aware of the following rules if you modify them:

*   **DO NOT upgrade PyTorch:** Upgrading PyTorch breaks the pre-compiled CUDA drivers on the T4 hardware, causing fatal mismatches.
*   **DO NOT upgrade Pillow:** Upgrading `Pillow` past Kaggle's native version triggers an `ImportError: cannot import name '_Ink'`.
*   **Avoid Cross-GPU Splitting (`device_map="auto"`):** Some custom remote code (like MiniCPM) uses `.scatter()` operations that crash if `accelerate` splits the model across both T4 GPUs. Always force these models onto a single GPU using `device_map={"": 0}`.
*   **Transformers v5 Incompatibilities:** Newer versions of transformers enforce an `allow_all_kernels` argument during initialization. Older models like `LocateAnything` crash upon receiving this. The `Kaggle_Vision_Models_Suite.ipynb` utilizes surgical Python monkey-patching (`PreTrainedModel.__init__` interception) to snip out this argument and guarantee stable loading without downgrading the entire environment.

### 4. Memory Management
Always use the provided `clear_vram()` function between model evaluations. It invokes Python's garbage collector and `torch.cuda.empty_cache()` to prevent Out-Of-Memory (OOM) crashes when switching between 3B+ parameter models.
