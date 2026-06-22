import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import open_clip
import faiss

from sklearn.metrics.pairwise import cosine_similarity

CSV_PATH = "data/styles.csv"
IMAGE_FOLDER = "data/images"

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

print("="*50)
print("Dataset Loaded Successfully!")
print("="*50)

print("Total Products :", len(df))

print("\nLoading CLIP Model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model.to(device)

print("CLIP Loaded Successfully!")

print("\nGenerating Image Embeddings...")

embeddings = []
valid_products = []

LIMIT = 500

for _, row in df.head(LIMIT).iterrows():

    image_path = os.path.join(
        IMAGE_FOLDER,
        f"{row['id']}.jpg"
    )

    if not os.path.exists(image_path):
        continue

    image = preprocess(
        Image.open(image_path)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)

    embedding = embedding.cpu().numpy()[0]

    embeddings.append(embedding)

    valid_products.append(row)

print(f"Generated {len(embeddings)} embeddings.")

embeddings = np.array(embeddings).astype("float32")

np.save(
    "embeddings/image_embeddings.npy",
    embeddings
)

print("Embeddings Saved!")

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(embeddings)

print("FAISS Index Created!")


# task 1

print("\n" + "=" * 50)
print("TASK 1 : SMART PRODUCT RECOMMENDATION")
print("=" * 50)

recommendation_map = {

    "Shirts": ["Formal Pants", "Leather Belt", "Watch"],

    "Tshirts": ["Jeans", "Sneakers", "Cap"],

    "Jeans": ["T-shirt", "Sneakers", "Backpack"],

    "Running Shoes": ["Sports Socks", "Fitness Watch", "Water Bottle"],

    "Casual Shoes": ["Backpack", "Wallet", "Sunglasses"],

    "Sandals": ["Beach Bag", "Hat", "Sunglasses"],

    "Watches": ["Bracelet", "Wallet", "Sunglasses"]

}

sample_product = valid_products[0]

article = sample_product["articleType"]

print("\nSelected Product :")

print(sample_product["productDisplayName"])

print("\nRecommended Products :")

if article in recommendation_map:

    for item in recommendation_map[article]:

        print("✓", item)

else:

    print("✓ Wallet")
    print("✓ Sunglasses")
    print("✓ Backpack")

    print("\n" + "=" * 50)
print("TASK 2 : UNIQUE PRODUCT CATALOG")
print("=" * 50)

similarity = cosine_similarity(embeddings)

visited = set()

catalog = []

for i in range(len(valid_products)):

    if i in visited:
        continue

    group = [i]

    visited.add(i)

    for j in range(i + 1, len(valid_products)):

        if similarity[i][j] > 0.95:

            group.append(j)

            visited.add(j)

    catalog.append(group)

print("Total Products :", len(valid_products))

print("Unique Products :", len(catalog))

print("\nSample Unique Catalog\n")

for group in catalog[:10]:

    idx = group[0]

    print(valid_products[idx]["productDisplayName"])

