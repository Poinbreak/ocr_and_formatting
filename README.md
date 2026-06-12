# OCR and Formatting Experiments

This repository contains experiments comparing different Optical Character Recognition (OCR) models, specifically focusing on handling complex formatting, handwritten text, and multi-language inputs (like Chinese).

## The Pipeline Architecture (Current Strategy)

To minimize hallucination and drastically reduce compute overhead, we have moved away from raw end-to-end OCR generation. Instead, we have implemented a strict 3-stage extraction pipeline (`pipeline_main.py`):

1. **Image Enhancement (OpenCV):** We use programmatic Computer Vision (Adaptive Gaussian Thresholding and Dilation) to strip shadows, correct lighting, and produce a high-contrast binary scan of the document. This runs with almost zero compute and forces the OCR model to read physical strokes rather than guessing based on blurred artifacts.
2. **Text Extraction (VLMs):** The cleaned image is passed into a Vision-Language Model (currently testing Qwen2.5-VL) to extract the raw text with exact pixel-fidelity.
3. **Structured Parsing (NuExtract):** To format the text into database-ready JSON without using massive LLMs, the raw OCR text is passed to `numind/NuExtract-tiny` (a 1.5B parameter model highly fine-tuned exclusively for JSON extraction).

## Models Tested

### 1. The Large VLM Candidates (In Progress)
We are currently evaluating the following models for the middle extraction stage of our pipeline. Due to their size (3B - 8B parameters), downloading and running them sequentially on CPU has been the primary bottleneck:
*   **Qwen2.5-VL-3B-Instruct:** Currently actively evaluating. Highly regarded for multi-lingual and precise document reading.
*   **PaliGemma-3b-mix-448:** Tested; download stalled on poor network connections but remains a lightweight alternative.
*   **MiniCPM-V 2.6 (8B):** Tested; highly capable but extremely heavy (16GB+ weights), causing timeouts on standard connections.

### 2. GLM-OCR (zai-org)
We transitioned to `zai-org/GLM-OCR`, a 0.9B parameter multimodal model optimized for complex document understanding and handwriting.

**The Hallucination Problem:**
Despite initial promise, testing revealed severe flaws in the 0.9B parameter GLM-OCR model when applied to complex or messy handwriting. It relies heavily on language priors. When faced with visual ambiguity, it fails to read physical strokes and instead confidently hallucinates based on surrounding context.

### 3. GOT-OCR 2.0 (Stepfun-ai)
Our initial experiments used `GOT-OCR-2.0-hf`. 
*   **Setbacks:** It struggled significantly on CPU-only setups. Without explicit repetition penalties, the model frequently entered infinite generation loops (e.g., endlessly generating "8 9 8 9..."). It performed poorly on dense, handwritten cursive text.

## Files
*   `pipeline_main.py`: The ultimate unified script running the 3-stage enhancement and extraction pipeline.
*   `qwen_main.py`, `paligemma_main.py`, `minicpm_main.py`: Individual standalone testing scripts for the VLMs.
*   `nuextract_main.py`: Standalone script for testing the JSON extraction module.
*   `1_ocr_nuextract.json`: An example of flawless JSON output generated from messy handwriting using our NuExtract pipeline.