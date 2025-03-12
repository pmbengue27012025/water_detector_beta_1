import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import logging

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialisation de FastAPI
app = FastAPI()

# Dossier pour stocker les images
IMAGE_FOLDER = "sentinel_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)


# Configuration du modèle U-Net
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.decoder1 = self.upconv_block(1024, 512)
        self.decoder2 = self.upconv_block(512, 256)
        self.decoder3 = self.upconv_block(256, 128)
        self.decoder4 = self.upconv_block(128, 64)

        # Final layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder
        d1 = self.decoder1(bottleneck)
        d1 = torch.cat([d1, e4], dim=1)
        d2 = self.decoder2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d3 = self.decoder3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d4 = self.decoder4(d3)
        d4 = torch.cat([d4, e1], dim=1)

        # Final layer
        out = self.final_conv(d4)
        return out


# Fonction pour charger et prétraiter les images Sentinel-2
def load_and_preprocess_image(file_path):
    with rasterio.open(file_path) as dataset:
        # Lire les bandes spectrales (rouge, vert, bleu, proche infrarouge)
        bands = [dataset.read(i) for i in range(1, 5)]  # Bandes 1 à 4
        image = np.stack(bands, axis=0)  # Shape: [4, height, width]
        image = image / 10000.0  # Normalisation
        return torch.tensor(image, dtype=torch.float32)


# Fonction pour créer des labels (simulé ici)
def create_labels(image):
    # Simuler des labels (eau, végétation, sol nu)
    labels = np.zeros((image.shape[1], image.shape[2]), dtype=np.int64)
    labels[image[3] > 0.2] = 1  # Eau (basé sur NDWI)
    labels[image[2] > 0.4] = 2  # Végétation (basé sur NDVI)
    return torch.tensor(labels, dtype=torch.int64)


# Entraînement du modèle
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


# Endpoint pour entraîner le modèle
@app.post("/train")
async def train():
    try:
        # Charger les données d'entraînement (simulé ici)
        image_paths = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith(".tiff")]
        X_train = torch.stack([load_and_preprocess_image(path) for path in image_paths])
        y_train = torch.stack([create_labels(img) for img in X_train])

        # Créer un DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        # Initialiser le modèle, la fonction de perte et l'optimiseur
        model = UNet(in_channels=4, out_channels=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Entraîner le modèle
        train_model(model, train_loader, criterion, optimizer, num_epochs=10)

        # Sauvegarder le modèle
        torch.save(model.state_dict(), "unet_model.pth")
        return JSONResponse(content={"message": "Modèle entraîné et sauvegardé avec succès."})
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint pour prédire sur une nouvelle image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Charger l'image
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Prétraitement de l'image
        image = load_and_preprocess_image(temp_file_path).unsqueeze(0)  # Ajouter une dimension de batch

        # Charger le modèle
        model = UNet(in_channels=4, out_channels=3)
        model.load_state_dict(torch.load("unet_model.pth"))
        model.eval()

        # Prédiction
        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Retourner la prédiction
        return JSONResponse(content={"prediction": predicted_class.tolist()})
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Supprimer le fichier temporaire
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# Démarrer l'application FastAPI
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)