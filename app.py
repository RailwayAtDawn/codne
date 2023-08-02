from io import BytesIO
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image

import gradio as gr
import requests
import string
import torch
import os

modelName = "blip2-opt-2.7b"
modelDownloadPrefix = "Salesforce/"
modelLocalPrefix = "./models/"
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

modelList = [
    "blip2-opt-2.7b",
    "blip2-opt-6.7b",
    "blip2-flan-t5-xl",
    "blip2-flan-t5-xxl",
    "blip2-opt-2.7b-coco",
    "blip2-opt-6.7b-coco",
    "blip2-flan-t5-xl-coco"
]

def modelChange(name):
    # Models are downloaded to  ~/.cache/huggingface/hub.
    global modelName, model
    modelName = name
    if not modelName in modelList:
        modelName = "blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(modelDownloadPrefix + modelName)
    model = Blip2ForConditionalGeneration.from_pretrained(
            modelDownloadPrefix + modelName, torch_dtype=torch.float16)
    model.to(device)
    return "Model changed to " + modelName


def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)

    return buffered


def imageCaption(image):
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


def imageCaptionWithText(image, prompt):
    inputs = processor(image, text=prompt, return_tensors="pt").to(
        device, torch.float16)
    
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


def chatInference(image, prompt, history=[]):
    history.append(prompt)

    textInput = " ".join(history)
    output = imageCaptionWithText(image, textInput)
    history.append(output)
    if not history[-1][-1] in string.punctuation:
        history[-1] += "."

    chat = [(history[i], history[i + 1])
            for i in range(0, len(history) - 1, 2)]
    return {chatbot: chat, state: history}


with gr.Blocks(
    css="""
    .message.svelte-w6rprc.svelte-w6rprc.svelte-w6rprc {font-size: 20px; margin-top: 20px}
    #component-21 > div.wrap.svelte-w6rprc {height: 600px;}
    """
) as iface:
    state = gr.State([])
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", interactive=True)
            information_text = gr.Textbox(lines=1, label="Information")
            model_radio = gr.Radio(modelList, label="Model Select", default_value=modelName)
            model_radio.change(modelChange, [model_radio], [information_text])
        with gr.Column(scale=1.8):
            with gr.Column():
                caption_output = gr.Textbox(lines=1, label="Caption Output")
                caption_button = gr.Button(value="Caption it!", interactive=True, variant="primary")
                caption_button.click(imageCaption, [image_input], [caption_output])
                gr.Markdown("""Please use proper punctuation. For example: How many people are in the picture?""")
                with gr.Row():
                    with gr.Column(scale=1.5):
                        chatbot = gr.Chatbot(label="Chat Output")
                    with gr.Column(scale=1):
                        chat_input = gr.Textbox(lines=1, label="Chat Input")
                        chat_input.submit(chatInference, [image_input, chat_input, state], [chatbot, state])
                        with gr.Row():
                            clear_button = gr.Button(value="Clear", interactive=True)
                            clear_button.click(lambda: ("", [], []), [], [chat_input, chatbot, state], queue=False)
                            submit_button = gr.Button(value="Submit", interactive=True, variant="primary")
                            submit_button.click(chatInference, [image_input, chat_input, state], [chatbot, state])
                image_input.change(lambda: ("", "", []), [], [chatbot, caption_output, state], queue=False)
                
iface.queue(concurrency_count=1, api_open=False, max_size=10)
iface.launch(enable_queue=True)

