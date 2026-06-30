# =============================================================================
# QLoRA Fine-Tuning Kaggle Notebook  (FUNSD JSON edition)
# INSTRUCTIONS:
#   Step 1: Paste CELL 1 into a Kaggle cell. Run it. Wait for the print.
#   Step 2: RESTART SESSION  (Runtime → Restart Session)
#   Step 3: Paste CELL 2 into a new cell. Run it to generate the dataset.
#   Step 4: Paste CELL 3 into a new cell. Run QLoRA fine-tuning.
#   Step 5: Paste CELL 4 into a new cell. Test the fine-tuned model.
#
# Dataset expected layout (output of funsd_handwriting_kaggle.ipynb):
#   /kaggle/input/funsd-handwritten/
#       training_data/
#           annotations/   ← FUNSD JSON files   ← this replaces the old .txt file
#           images/        ← augmented PNGs
#       testing_data/
#           annotations/
#           images/
# =============================================================================


# ════════════════════════════════════════════════════════════════════
# CELL 1 — INSTALLS
# ════════════════════════════════════════════════════════════════════
# !pip install -q -U peft trl transformers accelerate bitsandbytes datasets qwen-vl-utils

print("✅ Dependencies installed. Please restart your session if running for the first time.")


# ════════════════════════════════════════════════════════════════════
# CELL 2 — DATASET GENERATION  (FUNSD JSON → fine-tuning JSON)
# ════════════════════════════════════════════════════════════════════
import json
from pathlib import Path


def build_finetuning_dataset_from_funsd(
    funsd_root,         # root folder that contains training_data/ and/or testing_data/
    output_file,        # path to write the resulting fine-tuning JSON
    splits=None,        # list of split sub-dirs to include; None → auto-detect
    include_labels=True,# add FUNSD semantic labels (question/answer/header/other)
):
    """
    Reads FUNSD-format JSON annotation directories and pairs each document
    with its matching image, producing a vision-language instruction-tuning
    dataset that teaches the model to output structured bounding-box OCR.

    FUNSD JSON format (per field):
        {
          "box":  [x_min, y_min, x_max, y_max],
          "text": "full text of this field",
          "label": "question" | "answer" | "header" | "other",
          "words": [{"box": [...], "text": "word"}, ...]
        }

    Output record format (one per document image):
        {
          "messages": [
            {"role": "user",      "content": [{"type": "image", ...}, {"type": "text", ...}]},
            {"role": "assistant", "content": [{"type": "text",  "text": "<JSON string>"}]}
          ]
        }
    """

    system_prompt = (
        "You are an expert OCR and document-understanding engine. "
        "Given a scanned document image, extract ALL text regions with their "
        "precise bounding boxes. "
        "Output the result STRICTLY as a JSON object containing a 'message' array. "
        "Each element must have:\n"
        "  'box_boundary_no'          — integer index starting at 1\n"
        "  'text_in_box_boundary'     — exact text string inside that region\n"
        "  'box_boundary_coordinates' — [x_min, y_min, x_max, y_max] in pixels"
        + ("\n  'label'                    — one of: question | answer | header | other"
           if include_labels else "")
    )

    funsd_root = Path(funsd_root)

    # Auto-detect splits if not provided
    if splits is None:
        splits = []
        for candidate in ("training_data", "testing_data"):
            if (funsd_root / candidate / "annotations").exists():
                splits.append(candidate)
        if not splits:
            # Fallback: assume funsd_root IS the split (contains annotations/ and images/ directly)
            splits = ["."]

    # Collect (annotation_json_path, image_dir) pairs
    pairs = []
    for split in splits:
        split_path = funsd_root / split
        ann_dir    = split_path / "annotations"
        img_dir    = split_path / "images"

        if not ann_dir.exists():
            print(f"  [warn] annotations dir not found: {ann_dir}")
            continue
        if not img_dir.exists():
            print(f"  [warn] images dir not found: {img_dir}")
            continue

        for jp in sorted(ann_dir.glob("*.json")):
            pairs.append((jp, img_dir))

    print(f"Found {len(pairs)} annotation files across splits: {splits}")

    training_data = []
    skipped = 0

    for json_path, img_dir in pairs:
        stem = json_path.stem

        # Find matching image (try common extensions)
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            cand = img_dir / (stem + ext)
            if cand.exists():
                img_path = cand
                break

        if img_path is None:
            print(f"  [warn] no image found for annotation: {stem}")
            skipped += 1
            continue

        # Parse FUNSD JSON
        with open(json_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)

        # Build structured message entries from FUNSD fields
        message_entries = []
        box_no = 1
        for field in annotation.get("form", []):
            field_text  = field.get("text", "").strip()
            field_box   = field.get("box", [])
            field_label = field.get("label", "other")

            if not field_text or len(field_box) != 4:
                continue

            entry = {
                "box_boundary_no":          box_no,
                "text_in_box_boundary":     field_text,
                "box_boundary_coordinates": field_box,   # [x_min, y_min, x_max, y_max]
            }
            if include_labels:
                entry["label"] = field_label

            message_entries.append(entry)
            box_no += 1

        if not message_entries:
            skipped += 1
            continue

        assistant_response = {"message": message_entries}

        record = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path.resolve())},
                        {"type": "text",  "text":  system_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json.dumps(assistant_response, indent=2)},
                    ],
                },
            ]
        }
        training_data.append(record)

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(training_data, out, indent=4)

    print(
        f"✅ Dataset built: {len(training_data)} samples"
        f"  ({skipped} skipped, no image)  →  {output_file}"
    )
    return len(training_data)


# ── Run it ────────────────────────────────────────────────────────────────────
# Point funsd_root at the folder you uploaded to Kaggle.
# It should contain training_data/ and/or testing_data/ sub-directories,
# each with annotations/ (JSON files) and images/ (PNG files) inside.

build_finetuning_dataset_from_funsd(
    funsd_root   = "/kaggle/input/funsd-handwritten",  # ← your uploaded dataset root
    output_file  = "/kaggle/working/qwen_finetune_data.json",
    splits       = ["training_data", "testing_data"],  # remove "testing_data" to use train-only
    include_labels = True,
)


# ════════════════════════════════════════════════════════════════════
# CELL 3 — QLORA FINE-TUNING SCRIPT
# ════════════════════════════════════════════════════════════════════
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Configuration
MODEL_ID     = "Qwen/Qwen3.5-0.8B"
DATASET_PATH = "/kaggle/working/qwen_finetune_data.json"
OUTPUT_DIR   = "/kaggle/working/qwen3.5-0.8b-ocr-finetuned"

# 2. Load the JSON dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# 3. Configure 4-bit Quantization (crucial for VRAM limits)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 4. Load Processor and Multimodal Model
print(f"Loading {MODEL_ID} in 4-bit...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 5. Prepare Model for QLoRA
# Force config to float16 to prevent internal bfloat16 casting (common in Qwen)
if hasattr(model, "config"):
    model.config.torch_dtype = torch.float16

# Catch any stray bfloat16 parameters that might crash T4s
for param in model.parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)

model = prepare_model_for_kbit_training(model)

# Target the attention projections. Qwen 3.5's hybrid architecture uses these.
lora_config = LoraConfig(
    r=16,               # rank (higher = more params, more VRAM)
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Fix for Kaggle T4: Force LoRA adapters to float32.
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

model.print_trainable_parameters()

# 6. Data Formatting Function
def formatting_func(example):
    if isinstance(example["messages"][0], dict):
        return processor.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    return [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in example["messages"]
    ]

# 7. Define Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,   # keep at 1-2 for Kaggle T4
    gradient_accumulation_steps=4,   # effective batch size = 2 × 4 = 8
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    fp16=False,   # disabled: Qwen bfloat16 internals crash fp16 GradScaler on T4
    bf16=False,   # 4-bit quant handles VRAM — no AMP needed
    max_steps=300,
    logging_steps=10,
    save_steps=50,
    warmup_ratio=0.05,
    report_to="none"   # set to "wandb" if you use Weights & Biases
)

# 8. Initialize Trainer and Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=training_args,
    data_collator=None
)

print("Starting OCR fine-tuning...")
trainer.train()

# 9. Save the fine-tuned adapter
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
processor.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
print("✅ Training complete! Adapter saved.")


# ════════════════════════════════════════════════════════════════════
# CELL 4 — TEST THE FINE-TUNED MODEL
# ════════════════════════════════════════════════════════════════════
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
import os

# 1. Configuration
BASE_MODEL_ID  = "Qwen/Qwen3.5-0.8B"
ADAPTER_PATH   = "/kaggle/working/qwen3.5-0.8b-ocr-finetuned/final_adapter"

# Point this at any image from your FUNSD dataset (augmented or original)
TEST_IMAGE_PATH = "/kaggle/input/funsd-handwritten/testing_data/images/0000971160.png"

print("Starting inference test...")
if not os.path.exists(TEST_IMAGE_PATH):
    print(f"⚠️  Test image not found at {TEST_IMAGE_PATH}")
    print("   Update TEST_IMAGE_PATH to any .png from your FUNSD images/ folder.")
else:
    # 2. Configure 4-bit Quantization (must match training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load base model and processor
    print(f"Loading base model {BASE_MODEL_ID}...")
    processor  = AutoProcessor.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 4. Load the fine-tuned LoRA adapter
    print("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # 5. Prepare input — same prompt structure used in training
    system_prompt = (
        "You are an expert OCR and document-understanding engine. "
        "Given a scanned document image, extract ALL text regions with their "
        "precise bounding boxes. "
        "Output the result STRICTLY as a JSON object containing a 'message' array. "
        "Each element must have:\n"
        "  'box_boundary_no'          — integer index starting at 1\n"
        "  'text_in_box_boundary'     — exact text string inside that region\n"
        "  'box_boundary_coordinates' — [x_min, y_min, x_max, y_max] in pixels\n"
        "  'label'                    — one of: question | answer | header | other"
    )

    image    = Image.open(TEST_IMAGE_PATH).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  system_prompt},
        ]
    }]

    print("Processing inputs...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(torch.float16).to(model.device) if v.is_floating_point() else v.to(model.device)
        for k, v in inputs.items()
    }
    inputs.pop("mm_token_type_ids", None)

    # 6. Generate prediction
    print("Generating prediction...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    print("\n" + "=" * 60)
    print("🎯  MODEL PREDICTION (JSON):")
    print(output_text)
    print("=" * 60)

    # 7. Try to parse and pretty-print the predicted JSON
    try:
        parsed = json.loads(output_text)
        print(f"\nParsed {len(parsed['message'])} bounding-box regions:")
        for entry in parsed["message"]:
            print(
                f"  [{entry['box_boundary_no']}] "
                f"{entry.get('label','?'):10s}  "
                f"{entry['box_boundary_coordinates']}  "
                f"\"{entry['text_in_box_boundary'][:60]}\""
            )
    except Exception:
        print("(Could not parse output as JSON — raw text shown above)")
