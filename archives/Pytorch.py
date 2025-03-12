from fastapi.encoders import jsonable_encoder
import asyncio
import os
import re
import logging
import numpy as np
import rasterio
import tempfile
import aiohttp
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sentinelhub import (
    SentinelHubRequest,
    DataCollection,
    MimeType,
    BBox,
    CRS,
    bbox_to_dimensions,
    SHConfig
)
from rasterio.transform import from_origin
from geojson import Polygon
import requests
import pandas as pd
import json
# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation de FastAPI
app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration Sentinel Hub
config = SHConfig()
config.sh_client_id = os.getenv("SENTINELHUB_CLIENT_ID", "2534ab91-7f44-4f8c-8645-b94ae9ec4b95")
config.sh_client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET", "LPK1LXXhyTK7HnI2ZdlT7eRu0hIWc3bT")

# Dossier pour stocker les images
IMAGE_FOLDER = "sentinel_images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

WATER_THRESHOLDS = {
    "NDWI": 0.2,
    "MNDWI": 0.4,
    "MIN_VALID_PIXELS": 0.1  # 10% de pixels valides
}

# Fonctions utilitaires
def format_sentinel_date(dt: datetime) -> str:
    """Formate une date pour l'API Sentinel Hub"""
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def parse_filename(filename: str):
    pattern = r"sentinel_image_([-\d.]+)_([-\d.]+)_([\d.]+)km_(\d{8})_(\d{6})\.tiff"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError("Format de fichier invalide")

    return (
        float(match.group(1)),  # Latitude
        float(match.group(2)),  # Longitude
        float(match.group(3)),  # Altitude (km)
        datetime.strptime(match.group(4), "%Y%m%d").replace(tzinfo=timezone.utc),  # Date
        datetime.strptime(match.group(5), "%H%M%S").time()  # Heure
    )


# Calcul des indices
def calculate_indices(red, nir, green, swir):
    ndvi = np.divide(nir - red, nir + red, out=np.zeros_like(nir), where=(nir + red) != 0)
    ndwi = (green - nir) / (green + nir + 1e-10)
    mndwi = (green - swir) / (green + swir + 1e-10)
    return ndvi, ndwi, mndwi


# Gestion de l'authentification Sentinel Hub
async def get_access_token():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://services.sentinel-hub.com/oauth/token",
                    auth=aiohttp.BasicAuth(config.sh_client_id, config.sh_client_secret),
                    data={"grant_type": "client_credentials"},
                    timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["access_token"]
    except Exception as e:
        logger.error(f"Erreur d'authentification: {str(e)}")
        raise HTTPException(status_code=503, detail="Service d'authentification indisponible")

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def fetch_sentinel_geodata(lon, lat):
    """Récupère les données géométriques (polygone) d'une image Sentinel-2 à une position donnée."""

    ACCESS_TOKEN = await get_access_token()
    if not ACCESS_TOKEN:
        print("Erreur : Impossible de récupérer le token d'accès.")
        return None

    offset = 0.002  # Définition de la boîte englobante (BBOX)
    bbox = [lon - offset, lat - offset, lon + offset, lat + offset]

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=30)
    datetime_range = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"

    url = " https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search"

    payload = {
        "bbox": bbox,
        "collections": ["sentinel-2-l2a"],
        "datetime": datetime_range,
        "limit": 1
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    features = data.get("features", [])

                    if features:
                        geometry = features[0]["geometry"]
                        print(json.dumps(geometry, indent=5))  # Affichage du GeoJSON
                        return geometry
                    else:
                        print("Aucune image Sentinel-2 trouvée pour ces coordonnées.")
                        return None
                else:
                    print(f"Erreur {response.status}: {await response.text()}")
                    return None
        except Exception as e:
            print(f"Erreur lors de la requête : {e}")
            return None


def create_stats_request(lon, lat, data, evalscript, resolution):
    utm_zone = int((lon + 180) / 6) + 1
    return {
        "input": {
            "bounds": {
                "geometry": data,
                "properties": {"crs": f"http://www.opengis.net/def/crs/OGC/1.3/CRS84{utm_zone:02d}"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "mosaickingOrder": "leastCC",
                    "resolution": f"{resolution}M"
                }
            }]
        },
        "aggregation": {
            "timeRange": {"from": "2020-01-01T00:00:00Z", "to": "2024-12-31T00:00:00Z"},
            "aggregationInterval": {"of": "P30D"},
            "evalscript": evalscript,
            "resx": resolution,
            "resy": resolution
        }
    }


async def fetch_water_index(lon, lat, evalscript_name):
    evalscripts = {
        "NDWI": """
        //VERSION=3
        function setup() {
            return {
                input: [{ bands: ["B03", "B08", "dataMask"], units: "REFLECTANCE" }],
                output: [{ id: "data", bands: 1 }, { id: "dataMask", bands: 1 }]
            }
        }
        function evaluatePixel(samples) {
            const eps = 1e-6;
            const denom = samples.B03 + samples.B08 + eps;
            return {
                data: [(samples.B03 - samples.B08) / denom],
                dataMask: [samples.dataMask * (denom > eps ? 1 : 0)]
            };
        }""",

        "MNDWI": """
        //VERSION=3
        function setup() {
            return {
                input: [{ bands: ["B03", "B11", "dataMask"], units: "REFLECTANCE" }],
                output: [{ id: "data", bands: 1 }, { id: "dataMask", bands: 1 }]
            }
        }
        function evaluatePixel(samples) {
            const eps = 1e-6;
            const denom = samples.B03 + samples.B11 + eps;
            return {
                data: [(samples.B03 - samples.B11) / denom],
                dataMask: [samples.dataMask * (denom > eps ? 1 : 0)]
            };
        }"""
    }

    data = await fetch_sentinel_geodata(lon, lat)
    if not data:
        return None

    stats_request = create_stats_request(
        lon, lat, data,
        evalscripts[evalscript_name],
        resolution=10 if evalscript_name == "NDWI" else 20
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(
                "https://services.sentinel-hub.com/api/v1/statistics",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {await get_access_token()}"
                },
                json=stats_request
        ) as response:
            if response.status == 200:
                return await response.json()
    return None


def analyze_water(ndwi_mean, mndwi_mean, valid_pixels_ratio):
    detection = {
        "ndwi_detection": bool(ndwi_mean > WATER_THRESHOLDS["NDWI"]),
        "mndwi_detection": bool(mndwi_mean > WATER_THRESHOLDS["MNDWI"]),
        "valid_data": bool(valid_pixels_ratio > WATER_THRESHOLDS["MIN_VALID_PIXELS"])
    }

    detection["water_detected"] = bool(
        detection["ndwi_detection"] and
        detection["mndwi_detection"] and
        detection["valid_data"]
    )

    # Conversion numérique explicite
    detection["confidence"] = float(min(
        (float(ndwi_mean) / WATER_THRESHOLDS["NDWI"]),
        (float(mndwi_mean) / WATER_THRESHOLDS["MNDWI"])
    )) if detection["water_detected"] else 0.0

    return detection

async def fetch_sentinel_data(lon,lat):
    data= await fetch_sentinel_geodata(lon,lat)
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{
          bands: [
            "B04",
            "B08",
            "SCL",
            "dataMask"
          ]
        }],
        output: [
          {
            id: "data",
            bands: 1
          },
          {
            id: "dataMask",
            bands: 1
          }]
      }
    }

    function evaluatePixel(samples) {
        let ndvi = (samples.B08 - samples.B04)/(samples.B08 + samples.B04)

        var validNDVIMask = 1
        if (samples.B08 + samples.B04 == 0 ){
            validNDVIMask = 0
        }

        var noWaterMask = 1
        if (samples.SCL == 6 ){
            noWaterMask = 0
        }

        return {
            data: [ndvi],
            // Exclude nodata pixels, pixels where ndvi is not defined and water pixels from statistics:
            dataMask: [samples.dataMask * validNDVIMask * noWaterMask]
        }
    }
    """

    stats_request = {
        "input": {
            "bounds": {
                "geometry": data,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/32633"
                }
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "mosaickingOrder": "leastCC",
                    "maxCloudCoverage": 40
                }
            }]
        },
        "aggregation": {
            "timeRange": {
                "from": "2020-01-01T00:00:00Z",
                "to": "2024-12-31T00:00:00Z"
            },
            "aggregationInterval": {
                "of": "P30D"
            },
            "evalscript": evalscript,
            "resx": 10,
            "resy": 10
        }
    }

    # Obtenir le token d'accès
    access_token = await get_access_token()
    if not access_token:
        logger.error("Impossible d'obtenir un token d'accès. Abandon de la requête.")
        return None

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    url = "https://services.sentinel-hub.com/api/v1/statistics"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=stats_request) as response:
                print(url)
                if response.status != 200:
                    error_msg = await response.text()
                    logger.error(f"Erreur API {response.status} : {error_msg}")
                    return None

                sh_statistics = await response.json()
                logger.info("Données ndvi reçues avec succès depuis Sentinel Hub API.")

                # Vérification du contenu
                if "data" not in sh_statistics or not sh_statistics["data"]:
                    logger.warning("Réponse API vide ou invalide.")
                    return None

                return sh_statistics
    except Exception as e:
        logger.error(f"Erreur lors de la requête Sentinel Hub : {str(e)}", exc_info=True)
        return None
async def ndwi_fetch_sentinel_data(lon, lat):
    data = await fetch_sentinel_geodata(lon, lat)
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{
          bands: ["B03", "B08", "dataMask"]  // Supprimé SCL car inutile pour NDWI
        }],
        output: [
          { id: "data", bands: 1 },
          { id: "dataMask", bands: 1 }
        ]
      }
    }

    function evaluatePixel(samples) {
        let ndwi = (samples.B03 - samples.B08) / (samples.B08 + samples.B03 + 1e-6); // Éviter la division par zéro
        var validndwiMask = (samples.B08 + samples.B03) > 0 ? 1 : 0;
        return {
            data: [ndwi],
            dataMask: [samples.dataMask * validndwiMask]  // Pas de masque SCL
        };
    }
    """
    stats_request = {
        "input": {
            "bounds": {
                "geometry": data,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/32633"
                }
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "mosaickingOrder": "leastCC",
                    "maxCloudCoverage": 40
                }
            }]
        },
        "aggregation": {
            "timeRange": {
                "from": "2020-01-01T00:00:00Z",
                "to": "2024-12-31T00:00:00Z"
            },
            "aggregationInterval": { "of": "P30D" },
            "evalscript": evalscript,
            "resx": 10,
            "resy": 10
        }
    }

    access_token = await get_access_token()
    if not access_token:
        logger.error("Token d'accès manquant.")
        return None

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://services.sentinel-hub.com/api/v1/statistics",
                headers=headers,
                json=stats_request
            ) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    logger.error(f"Erreur {response.status}: {error_msg}")
                    return None

                sh_statistics = await response.json()
                logger.info(f"Réponse API NDWI: {json.dumps(sh_statistics, indent=2)}")  # Log détaillé

                if not sh_statistics.get("data"):
                    logger.warning("Aucune donnée NDWI trouvée.")
                    return None

                return sh_statistics
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        return None
async def mndwi_fetch_sentinel_data(lon,lat):
    data = await fetch_sentinel_geodata(lon, lat)
    evalscript = """
    //VERSION=3
    function setup() {
      return {
        input: [{
          bands: [
            "B03",
            "B11",
            "SCL",
            "dataMask"
          ]
        }],
        output: [
          {
            id: "data",
            bands: 1
          },
          {
            id: "dataMask",
            bands: 1
          }]
      }
    }

    function evaluatePixel(samples) {
        let mndwi = (samples.B03 - samples.B11)/(samples.B11 + samples.B03)

        var validmndwiMask = 1
        if (samples.B11 + samples.B03 == 0 ){
            validmndwiMask = 0
        }

        var noWaterMask = 1
        if (samples.SCL == 6 ){
            noWaterMask = 0
        }

        return {
            data: [mndwi],
            // Exclude nodata pixels, pixels where mndwi is not defined and water pixels from statistics:
            dataMask: [samples.dataMask * validmndwiMask * noWaterMask]
        }
    }
    """

    stats_request = {
        "input": {
            "bounds": {
                "geometry":data,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/32633"
                }
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "mosaickingOrder": "leastCC"
                    }
                }
            ]
        },
        "aggregation": {
            "timeRange": {
                "from": "2020-01-01T00:00:00Z",
                "to": "2024-12-31T00:00:00Z"
            },
            "aggregationInterval": {
                "of": "P30D"
            },
            "evalscript": evalscript,
            "resx": 10,
            "resy": 10
        }
    }
	  # Obtenir le token d'accès
    access_token = await get_access_token()
    if not access_token:
        logger.error("Impossible d'obtenir un token d'accès. Abandon de la requête.")
        return None

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    url = "https://services.sentinel-hub.com/api/v1/statistics"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=stats_request) as response:
                print(url)
                if response.status != 200:
                    error_msg = await response.text()
                    logger.error(f"Erreur API {response.status} : {error_msg}")
                    return None

                sh_statistics = await response.json()
                logger.info("Données mndwi reçues avec succès depuis Sentinel Hub API.")

                # Vérification du contenu
                if "data" not in sh_statistics or not sh_statistics["data"]:
                    logger.warning("Réponse API vide ou invalide.")
                    return None

                return sh_statistics
    except Exception as e:
        logger.error(f"Erreur lors de la requête Sentinel Hub : {str(e)}", exc_info=True)
        return None

def process_stats(data, index):
    """Extrait les statistiques de la réponse"""
    try:
        stats = data["data"][0]["stats"]["default"][index]
        return {
            "mean": stats.get("mean"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "stdev": stats.get("stdev")
        }
    except KeyError as e:
        logger.error(f"Données statistiques manquantes: {str(e)}")
        return None


# Récupération des données météo
async def get_weather_data(lat: float, lon: float, start_date: datetime, end_date: datetime):
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                return await response.json()

    except Exception as e:
        logger.error(f"Erreur météo: {str(e)}")
        raise HTTPException(status_code=503, detail="Service météo indisponible")


# Endpoint principal
@app.post("/analyze-tiff")
async def analyze_tiff(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        # Validation du fichier
        if not file.filename.lower().endswith((".tiff", ".tif")):
            raise HTTPException(status_code=400, detail="Format de fichier non supporté")

        # Extraction des métadonnées
        lat, lon, altitude, date, time = parse_filename(file.filename)
        start_date = (date - timedelta(days=5 * 365)).replace(tzinfo=timezone.utc)
        end_date = date.replace(tzinfo=timezone.utc)
        print(date,start_date, end_date)
        print(lat, lon, altitude, date, time)

        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tiff") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Lecture des bandes
        with rasterio.open(temp_file_path) as src:
            if src.count < 4:
                raise HTTPException(status_code=400, detail="Fichier TIFF incomplet (4 bandes requises)")
            # Ajustez les indices selon l'ordre réel des bandes
            red = src.read(3).astype(np.float32)  # B04 (rouge)
            nir = src.read(2).astype(np.float32)  # B08 (NIR)
            green = src.read(1).astype(np.float32)  # B03 (vert)
            swir = src.read(4).astype(np.float32)  # B11 (SWIR)
            # Recalcul des indices avec les bonnes bandes
            ndvi, ndwi, mndwi = calculate_indices(red, nir, green, swir)
            # Log des valeurs pour débogage
            logger.info(f"NDWI mean: {np.nanmean(ndwi)}, MNDWI mean: {np.nanmean(mndwi)}")
            # Calcul des statistiques
            stats = {
                "ndvi": float(np.nanmean(ndvi)),
                "ndwi": float(np.nanmean(ndwi)),
                "mndwi": float(np.nanmean(mndwi)),
                "valid_pixels_ratio": np.mean(np.isfinite(ndwi) & np.isfinite(mndwi))
            }
            # Analyse de l'eau
            water_analysis = analyze_water(
                stats["ndwi"],
                stats["mndwi"],
                stats["valid_pixels_ratio"]
            )
        # Récupération données externes
        geometry = {
            "type": "Point",
            "coordinates": [lat, lon]  # Longitude, Latitude
        }

        time_range = {
            "from": start_date,
            "to": end_date
        }

        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B04", "B08", "B03", "SCL"],
            output: [
              { id: "ndvi", bands: 1 },
              { id: "ndwi", bands: 1 }
            ]
          };
        }

        function evaluatePixel(sample) {
          let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
          let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
          return [ndvi, ndwi];
        }
        """

        water_historical_data = {
            "ndwi": await fetch_water_index(lon, lat, "NDWI"),
            "mndwi": await fetch_water_index(lon, lat, "MNDWI")
        }
        historical_data = await fetch_sentinel_data(lon, lat)
        ndwihistorical_data = await ndwi_fetch_sentinel_data(lon,lat)
        mndwihistorical_data = await mndwi_fetch_sentinel_data(lon, lat)
        weather_data = await get_weather_data(lat, lon, start_date, end_date)

        return JSONResponse(
            content=jsonable_encoder({
                "metadata": {
                    "coordinates": {"lat": float(lat), "lon": float(lon)},
                    "altitude_km": float(altitude),
                    "date": date.strftime("%Y-%m-%d"),
                    "time": time.strftime("%H:%M:%S")
                },
                "water_historical_data":water_historical_data,
                "water_analysis": {
                    **water_analysis,
                    "thresholds": WATER_THRESHOLDS,
                    "current_values": {
                        "ndvi": float(stats["ndvi"]),
                        "ndwi": float(stats["ndwi"]),
                        "mndwi": float(stats["mndwi"]),
                        "valid_pixels_ratio": float(stats["valid_pixels_ratio"])
                    }
                },
                "indices": {
                    "ndvi": float(np.nanmean(ndvi)),
                    "ndwi": float(np.nanmean(ndwi)),
                    "mndwi": float(np.nanmean(mndwi)),
                    "historical_trend": historical_data,
                    "mndwihistorical_data": mndwihistorical_data,
                    "ndwihistorical_data": ndwihistorical_data
                },
                "weather": weather_data.get("daily", {})
            }),
            status_code=200
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Erreur non gérée: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Échec suppression fichier temporaire: {str(e)}")


# Serveur de fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")