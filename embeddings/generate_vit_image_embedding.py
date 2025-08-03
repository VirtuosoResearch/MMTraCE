# %%
import torch
from transformers import ViTFeatureExtractor, ViTModel
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
output_path = f"./vit_image/{args.state}.npy"
device = torch.device(f"cuda:{args.device}")

model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)
model.eval().to(device) 

all_embeddings = []

nodes = pd.read_csv(f"../data/Road_Networks/{args.state}/Road_Network_Nodes_{args.state}.csv")
for idx, node in tqdm(nodes.iterrows(), total=len(nodes)):
    id = int(node["node_id"])
    img_path = f"{image_folder}/{id}.png"
    # print(img_path)
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[0, 0].cpu().numpy() 
        all_embeddings.append(cls_embedding)
        # print(f"Done {img_path}")
    except Exception as e:
        cls_embedding = torch.zeros([768])
        all_embeddings.append(cls_embedding)
        print(f"[Warning] Failed to process {img_path}: {e}")

all_embeddings = np.stack(all_embeddings)  # shape: (N, 768)
np.save(output_path, all_embeddings)
print(f"Saved {all_embeddings.shape[0]} embeddings to {output_path}")

# model_name = "google/vit-base-patch16-224"
# feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
# model = ViTModel.from_pretrained(model_name)

# image_path = "/home/michael/project/ML4RoadSafety/ml_for_road_safety/image_sample_MD.png"
# image = Image.open(image_path).convert("RGB")

# inputs = feature_extractor(images=image, return_tensors="pt")

# # print("inputs.shape: ",inputs['pixel_values'].shape)

# with torch.no_grad():
#     outputs = model(**inputs)
#     last_hidden_state = outputs.last_hidden_state  # shape: (1, 197, 768)
#     cls_embedding = last_hidden_state[0, 0] # [CLS]
#     print(last_hidden_state.shape)
# print("CLS embedding shape:", cls_embedding.shape)
