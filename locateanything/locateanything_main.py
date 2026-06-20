import torch
import sys
import os
import re
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class LocateAnythingWorker:
    """Stateful worker that loads the model once and serves perception queries."""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map={"": self.device}
        ).eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",   # "fast" (MTP) | "slow" (NTP/AR) | "hybrid"
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> dict:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        """Scene text detection."""
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)
        
    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        """Object detection / document layout analysis."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading nvidia/LocateAnything-3B model on {device}...")

    worker = LocateAnythingWorker("nvidia/LocateAnything-3B", device=device, dtype=dtype)

    image_dir = r"C:\Saiyanht\projects\digitwin"
    img_file = "log book.png"
    image_path = os.path.join(image_dir, img_file)

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Processing: {img_file}")
    print(f"{'='*60}")

    img = Image.open(image_path).convert("RGB")

    # Resize image to prevent SDPA attention Out-of-Memory on 6GB VRAM
    max_size = 800  # reduce resolution
    if max(img.size) > max_size:
        print(f"Resizing image from {img.size} to a maximum of {max_size}px")
        img.thumbnail((max_size, max_size))

    print("\nRunning Scene Text Detection...")
    result = worker.detect_text(img)

    print(f"\n--- LocateAnything RESULT for {img_file} ---")
    print(result["answer"])

    output_file = os.path.join(image_dir, f"{os.path.splitext(img_file)[0]}_locateanything.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(result["answer"]))
    print(f"\nResult saved to: {output_file}")
