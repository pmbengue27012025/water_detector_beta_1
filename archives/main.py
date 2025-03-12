from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from sentinelhub import SentinelHubRequest, DataCollection, MimeType, BBox, CRS, bbox_to_dimensions, SHConfig
import logging
from datetime import datetime, timedelta
import rasterio
from rasterio.transform import from_origin
import numpy as np
import os
import io
from matplotlib import pyplot as plt

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialisation de FastAPI
app = FastAPI()

# Configuration de Sentinel Hub
config = SHConfig()
config.sh_client_id = os.getenv("SENTINELHUB_CLIENT_ID", "2534ab91-7f44-4f8c-8645-b94ae9ec4b95")  # Utiliser une variable d'environnement
config.sh_client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET", "LPK1LXXhyTK7HnI2ZdlT7eRu0hIWc3bT")  # Utiliser une variable d'environnement

def calculate_ndvi(red_band, nir_band):
    """
    Calcule l'indice NDVI à partir des bandes rouge et proche infrarouge.

    Args:
        red_band (np.ndarray): Bande rouge.
        nir_band (np.ndarray): Bande proche infrarouge.

    Returns:
        np.ndarray: Indice NDVI.
    """
    try:
        logger.info("Calcul de l'indice NDVI")
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-10)  # Éviter la division par zéro
        return ndvi
    except Exception as e:
        logger.error(f"Erreur lors du calcul de l'NDVI : {e}")
        raise

def calculate_ndwi(green_band, nir_band):
    """
    Calcule l'indice NDWI à partir des bandes verte et proche infrarouge.

    Args:
        green_band (np.ndarray): Bande verte.
        nir_band (np.ndarray): Bande proche infrarouge.

    Returns:
        np.ndarray: Indice NDWI.
    """
    try:
        logger.info("Calcul de l'indice NDWI")
        green = green_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        ndwi = (green - nir) / (green + nir + 1e-10)  # Éviter la division par zéro
        return ndwi
    except Exception as e:
        logger.error(f"Erreur lors du calcul de l'NDWI : {e}")
        raise

def interpret_ndvi(ndvi_value):
    """
    Interprète la valeur NDVI.

    Args:
        ndvi_value (float): Valeur NDVI.

    Returns:
        str: Interprétation de la valeur NDVI.
    """
    if ndvi_value < -0.2:
        return "Eau ou surface nue"
    elif -0.2 <= ndvi_value < 0.2:
        return "Sol nu ou végétation très faible"
    elif 0.2 <= ndvi_value < 0.5:
        return "Végétation modérée"
    elif ndvi_value >= 0.5:
        return "Végétation dense"
    else:
        return "Valeur NDVI invalide"

def interpret_ndwi(ndwi_value):
    """
    Interprète la valeur NDWI.

    Args:
        ndwi_value (float): Valeur NDWI.

    Returns:
        str: Interprétation de la valeur NDWI.
    """
    if ndwi_value < -0.2:
        return "Sol sec ou surface nue"
    elif -0.2 <= ndwi_value < 0.2:
        return "Sol humide ou végétation"
    elif ndwi_value >= 0.2:
        return "Eau ou zone humide"
    else:
        return "Valeur NDWI invalide"

@app.post("/analyze-tiff")
async def analyze_tiff(file: UploadFile = File(...)):
    """
    Endpoint pour analyser un fichier TIFF multispectral et retourner les valeurs NDVI et NDWI.
    """
    try:
        # Lire le fichier TIFF téléchargé
        contents = await file.read()
        with rasterio.open(io.BytesIO(contents)) as dataset:
            # Vérifier le nombre de bandes
            if dataset.count < 4:
                raise HTTPException(
                    status_code=400,
                    detail=f"Le fichier TIFF doit contenir au moins 4 bandes spectrales. Bandes disponibles : {dataset.count}"
                )

            # Lire les bandes spectrales (supposons que les bandes sont dans l'ordre : Bleu, Vert, Rouge, Proche Infrarouge)
            blue_band = dataset.read(1)  # Bande 1 : Bleu
            green_band = dataset.read(2)  # Bande 2 : Vert
            red_band = dataset.read(3)   # Bande 3 : Rouge
            nir_band = dataset.read(4)   # Bande 4 : Proche Infrarouge

        # Calculer les indices NDVI et NDWI
        ndvi = calculate_ndvi(red_band, nir_band)
        ndwi = calculate_ndwi(green_band, nir_band)

        # Interpréter les valeurs NDVI et NDWI
        ndvi_mean = float(np.nanmean(ndvi))
        ndwi_mean = float(np.nanmean(ndwi))
        ndvi_interpretation = interpret_ndvi(ndvi_mean)
        ndwi_interpretation = interpret_ndwi(ndwi_mean)

        # Retourner les résultats sous forme de JSON
        return {
            "ndvi_mean": ndvi_mean,
            "ndvi_interpretation": ndvi_interpretation,
            "ndwi_mean": ndwi_mean,
            "ndwi_interpretation": ndwi_interpretation,
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse du fichier TIFF : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-sentinel-image")
async def get_sentinel_image_api(lat: float, lon: float, size_km: float = 10):
    """
    Récupère une image Sentinel-2 pour une zone géographique donnée.
    """
    try:
        # Définir la zone d'intérêt (BBox)
        bbox = BBox([lon - size_km / 2, lat - size_km / 2, lon + size_km / 2, lat + size_km / 2], crs=CRS.WGS84)

        # Définir la période d'historique (4 ans)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=4 * 365)
        time_interval = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        # Nom du fichier de sortie
        output_file = f"sentinel_image_{lat}_{lon}_{size_km}km.tiff"

        # Récupérer l'image
        get_sentinel_image(bbox, time_interval, output_file)

        # Retourner le fichier TIFF
        return FileResponse(output_file, media_type="image/tiff", filename=output_file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'image Sentinel-2 : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")