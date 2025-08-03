# %%
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--state", default="DE", type=str)
parser.add_argument("--device", default=0, type=int)
args = parser.parse_args()

image_folder = f"/home/michael/project/data/Nodes_{args.state}"
output_path = f"./clip_image/{args.state}.npy"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()

all_embeddings = []

nodes = pd.read_csv(f"../data/Road_Networks/{args.state}/Road_Network_Nodes_{args.state}.csv")
for idx, node in tqdm(nodes.iterrows(), total=len(nodes)):
    node_id = int(node["node_id"])
    img_path = f"{image_folder}/{node_id}.png"
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)  # shape: (1, 512)
            features = features / features.norm(p=2, dim=-1, keepdim=True)  # normalize (optional)
            cls_embedding = features[0].cpu().numpy()
        all_embeddings.append(cls_embedding)
    except Exception as e:
        cls_embedding = np.zeros([512])
        all_embeddings.append(cls_embedding)
        print(f"[Warning] Failed to process {img_path}: {e}")

all_embeddings = np.stack(all_embeddings)  # shape: (N, 512)
np.save(output_path, all_embeddings)
print(f"Saved {all_embeddings.shape[0]} embeddings to {output_path}")
