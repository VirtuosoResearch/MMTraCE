import torch
from segment_anything import sam_model_registry
from PIL import Image
import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument("--state", default="DE", type=str)
parser.add_argument("--device", default=0, type=int)
args = parser.parse_args()

image_folder = f"/home/michael/project/data/Nodes_{args.state}"
output_path = f"./sam_image/{args.state}_embedding.npy"
device = torch.device(f"cuda:{args.device}")

checkpoint_path = "../data/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)
sam.eval()

transform = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

all_embeddings = []

nodes = pd.read_csv(f"../data/Road_Networks/{args.state}/Road_Network_Nodes_{args.state}.csv")
for idx, node in tqdm(nodes.iterrows(), total=len(nodes)):
    node_id = int(node["node_id"])
    img_path = f"{image_folder}/{node_id}.png"
    try:
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 1024, 1024)

        with torch.no_grad():
            image_embedding = sam.image_encoder(image_tensor)  # (1, 256, 64, 64)
            embedding = torch.mean(image_embedding.flatten(2), dim=-1)  # shape: (1, 256)
            embedding = embedding[0].cpu().numpy()
        all_embeddings.append(embedding)
    except Exception as e:
        print(f"[Warning] Failed to process {img_path}: {e}")
        all_embeddings.append(np.zeros(256))  # same dim as SAM embedding

all_embeddings = np.stack(all_embeddings)  # shape: (N, 256)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, all_embeddings)
print(f"Saved {all_embeddings.shape[0]} embeddings to {output_path}")
