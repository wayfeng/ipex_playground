import gradio as gr
import open_clip
import intel_extension_for_pytorch as ipex
import torch
import torchvision.transforms as T

from scipy.special import softmax

import pandas as pd

#model_id = 'ViT-B-32'
#pretrained_path = './checkpoints/open_clip_vit_b_32.pth'
model_id = 'ViT-L-14'
pretrained_path = f"./checkpoints/open_clip_{model_id.replace('-', '_').lower()}.pth"

model, _, preprocess = open_clip.create_model_and_transforms(model_id, pretrained=pretrained_path)
_ = model.eval()
device = torch.device('xpu')
model = model.to(device)
model = ipex.optimize(model)

tokenizer = open_clip.get_tokenizer(model_id)

def classify(image_pil, text):
    image = preprocess(image_pil).unsqueeze(0).to(device)
    texts = text.split(',')
    token = tokenizer(texts).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(token)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
        probs = (100 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
    data = pd.DataFrame({"text": texts, "prob": [x for x in probs[0]]})
    print(data)
    return gr.BarPlot(
        data,
        x="text",
        y="prob",
        title="Results",
        tooltip=["text", "prob"],
        y_lim=[0.0, 1.0],
        width=400,
    )

demo = gr.Interface(
    classify,
    [
        gr.Image(label="Image", type="pil"),
        gr.Textbox(label="Labels", info="Comma-seperated list of class labels"),
    ],
    gr.BarPlot(),
    examples=[["./assets/cat_dog.jpeg", "cat,dog,deer"]]
)

try:
    demo.launch(debug=True)
except Exception:
    demo.launch(share=True, debug=True)
