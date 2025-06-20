from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from PIL import Image
import io
import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os
import requests
import gdown

# ---- Model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Charger le modèle complet
import gdown
import os

file_id = "10QUXjMYPWP8KNAhBGqHtInDpIPq2yIa3"  # remplace par ton ID
destination = "unet_model_full.pth"

# Crée une vraie URL de téléchargement
url = f"https://drive.google.com/uc?id={file_id}"

# Télécharge si nécessaire
if not os.path.exists(destination):
    gdown.download(url, destination, quiet=False)
    if os.path.getsize(destination) < 1_000_000:  # Moins de 1 Mo → probablement mauvais fichier
        raise RuntimeError("Fichier téléchargé invalide ou trop petit. Vérifiez le lien Google Drive.")

with open(destination, "rb") as f:
    start = f.read(200)
    print("=== DEBUT DU FICHIER ===")
    print(start)
    
# === CHARGEMENT DU MODÈLE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(destination, map_location=device, weights_only=False)  # ou "cuda" selon le besoin
model.eval()

# ---- Transforms ----
image_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- API ----
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Chemin vers le dossier contenant les images test
IMAGE_DIR = "P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt"
MASK_DIR = "P8_Cityscapes_gtFine_trainvaltest/gtFine/val/frankfurt"

@app.get("/image_ids/", response_model=List[str])
def list_image_ids():
    image_ids = []
    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(".png"):
            # Supprime l'extension et éventuellement d'autres suffixes
            image_id = filename.replace("_leftImg8bit.png", "")
            image_ids.append(image_id)
    return sorted(image_ids)

@app.get("/image/{image_id}")
def get_image(image_id: str):
    filepath = os.path.join(IMAGE_DIR, f"{image_id}_leftImg8bit.png")
    return FileResponse(filepath)

@app.get("/mask/{image_id}")
def get_real_mask(image_id: str):
    filepath = os.path.join(MASK_DIR, f"{image_id}_gtFine_color.png")
    return FileResponse(filepath)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Lire l'image uploadée
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Prétraitement
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    imageT = transform(image)
    input_tensor = imageT.unsqueeze(0).to(device)

    # Prédiction
    with torch.no_grad():
        output = model(input_tensor)  # [1, C, H, W]
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy().astype(np.uint8)  # (H, W)

    # Appliquer la colormap 'jet'
    colormap = plt.get_cmap('jet')
    colored_mask = colormap(pred_mask / pred_mask.max())[:, :, :3]  # Retirer alpha, garder RGB
    colored_mask = (colored_mask * 255).astype(np.uint8)  # Convertir [0,1] -> [0,255]

    # Convertir en image PIL
    mask_image = Image.fromarray(colored_mask)

    # Réponse HTTP
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")

if __name__ == "__main__":
    # Récupère la variable d'environnement PORT, sinon utilise 8000 (pour les tests locaux)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
