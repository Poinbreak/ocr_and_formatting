import gradio as gr
import ollama
import base64
import json # <-- Added for JSON handling

# --- Step 1: Image to Text ---
def process_handwriting(image_path, system_prompt):
    if not image_path:
        return "Please upload an image first.", None

    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        response = ollama.chat(
            model='qwen2.5vl:7b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {
                    'role': 'user', 
                    'content': 'Please transcribe the handwritten text in this image.',
                    'images': [image_base64]
                }
            ]
        )
        
        extracted_text = response['message']['content']
        
        output_filename = "transcription.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(extracted_text)
            
        return extracted_text, output_filename

    except Exception as e:
        return f"An error occurred: {str(e)}", None


# --- Step 2: Text to JSON ---
def structure_text_to_json(raw_text, json_prompt):
    if not raw_text or raw_text.strip() == "":
        return "No text to process. Please extract text from an image first.", None

    try:
        # We instruct the model to output ONLY JSON.
        system_instruction = (
            f"You are a strict data extraction AI. Convert the provided text into a structured JSON format. "
            f"{json_prompt} "
            f"Output ONLY valid JSON. Do not include markdown blocks or conversational text."
        )

        response = ollama.chat(
            model='qwen2.5vl:7b', # We can use the same model for text-only processing
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': raw_text}
            ],
            options={'temperature': 0.1} # Low temperature for more predictable formatting
        )
        
        json_output = response['message']['content'].strip()
        
        # Clean up markdown formatting if the model includes it anyway
        if json_output.startswith('```json'):
            json_output = json_output[7:]
        if json_output.startswith('```'):
            json_output = json_output[3:]
        if json_output.endswith('```'):
            json_output = json_output[:-3]
            
        json_output = json_output.strip()

        # Save to a .json file
        json_filename = "structured_data.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            f.write(json_output)
            
        return json_output, json_filename

    except Exception as e:
        return f"An error occurred: {str(e)}", None


# --- Build the Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 📝 Qwen2.5-VL Handwriting OCR & Data Structuring")
    
    # ROW 1: Image Upload and OCR
    gr.Markdown("### Step 1: Extract Raw Text")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="filepath", label="Upload Handwritten Image")
            sys_prompt = gr.Textbox(
                label="Transcription Prompt", 
                lines=3,
                value="You are an expert transcriber. Read the handwritten text carefully and output ONLY the transcribed text. Preserve original line breaks, punctuation, and formatting as accurately as possible."
            )
            submit_btn = gr.Button("Extract Text", variant="primary")
            
        with gr.Column():
            text_output = gr.Textbox(label="Raw Model Output", lines=10)
            file_output = gr.File(label="Download .txt File")
            
    gr.Markdown("---")
    
    # ROW 2: JSON Structuring
    gr.Markdown("### Step 2: Structure into JSON")
    with gr.Row():
        with gr.Column():
            json_instructions = gr.Textbox(
                label="JSON Schema / Instructions", 
                lines=3,
                value="Extract the key entities such as dates, names, amounts, and action items. Organize them into logical keys."
            )
            json_btn = gr.Button("Convert to JSON", variant="secondary")
            
        with gr.Column():
            json_text_output = gr.Textbox(label="JSON Output", lines=10)
            json_file_output = gr.File(label="Download .json File")

    # --- Event Wiring ---
    submit_btn.click(
        fn=process_handwriting,
        inputs=[img_input, sys_prompt],
        outputs=[text_output, file_output]
    )
    
    # Wire the new JSON button to take the text from Step 1 and pass it to Step 2
    json_btn.click(
        fn=structure_text_to_json,
        inputs=[text_output, json_instructions],
        outputs=[json_text_output, json_file_output]
    )

if __name__ == "__main__":
    interface.launch()