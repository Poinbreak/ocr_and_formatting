import json
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Using the tiny model for super low compute
model_id = "numind/NuExtract-tiny"
print(f"Loading {model_id} on {device}...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=dtype, 
    trust_remote_code=True,
    device_map=device
)

def predict_NuExtract(model, tokenizer, text, schema, example=["", "", ""]):
    schema = json.dumps(json.loads(schema), indent=4)
    input_llm = "<|input|>\n### Template:\n" + schema + "\n"
    for i in example:
        if i != "":
            input_llm += "### Example:\n" + json.dumps(json.loads(i), indent=4) + "\n"
    input_llm += "### Text:\n" + text + "\n<|output|>\n"
    
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=4000).to(device)
    output = tokenizer.decode(model.generate(**input_ids, max_new_tokens=1000)[0], skip_special_tokens=True)
    try:
        return output.split("<|output|>")[1].split("<|end-")[0]
    except Exception as e:
        return output

# Based on the text in 1_ocr_glm.txt, here is a schema we want to extract
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

print("Reading 1_ocr_glm.txt...")
with open("1_ocr_glm.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Extracting structured JSON...")
result = predict_NuExtract(model, tokenizer, text, schema)

output_file = "1_ocr_nuextract.json"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result)

print(f"\n{'='*40}")
print(f"Extraction complete! Saved to {output_file}")
print(f"{'='*40}")
print(result)
