# Water Detector

## Description

Water Detector est un projet destiné à mesurer et détecter la présence d'eau via l'analyse de données satellitaires. Conçu comme une API REST avec FastAPI, il permet une intégration facile dans des systèmes de surveillance d'inondations, de gestion de bassins hydriques ou de suivi environnemental.

Ce projet utilise des données Sentinel-2 traitées en temps quasi-réel et propose des fonctionnalités asynchrones pour un traitement optimisé.

## Fonctionnalités

- Détection d'eau par analyse NDWI (Normalized Difference Water Index)
- API REST avec endpoints dédiés (téléchargement de données, visualisation, historique)
- Intégration native avec SentinelHub pour l'accès aux données satellitaires
- Traitement asynchrone des requêtes
- Génération de cartes d'eau au format GeoTIFF
- Historique des requêtes avec stockage local
- Prise en charge des fichiers GeoJSON pour la délimitation de zones
- Alertes configurables 

---

## Prérequis

### Matériel requis

Aucun matériel spécifique nécessaire (solution cloud-based)
Accès internet pour l'interaction avec SentinelHub
### Logiciel requis
- Python 3.11
- Bibliothèques nécessaires : 
     fastapi>=0.85.0
     numpy>=1.23.0
     rasterio>=1.3.0
     sentinelhub>=3.10.0
     aiohttp>=3.8.0
     pandas>=1.5.0
     python-dotenv>=0.21.0
     uvicorn>=0.19.0
---

## Installation

1. Clonez ce dépôt sur votre machine locale :  
   ```bash
   git clone https://github.com/votre-utilisateur/water_detector.git
   cd water_detector
   ```
2. Créer et activer un environnement virtuel :
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   
3. Installez les dépendances nécessaires :  
   ```bash
   pip install -r requirements.txt
   ```
4. Configurer les credentials SentinelHub :
   echo "SH_CLIENT_ID=votre_client_id" > .env
   echo "SH_CLIENT_SECRET=votre_client_secret" >> .env


## Utilisation

1. Lancez le programme principal :  
   ```bash
   uvicorn src.main:app --reload
   ```

## Structure du projet

```plaintext
water_detector/
├── src/                   # Code source principal
│   ├── main.py            # Point d'entrée FastAPI
│   └── processors/        # Modules de traitement des données
├── data/                  # Résultats et fichiers temporaires
├── config/                # Configuration SentinelHub
├── tests/                 # Tests automatisés
├── requirements.txt       # Dépendances Python
└── README.md              # Documentation
```

---

## Contribution

Les contributions sont les bienvenues !  
Veuillez suivre ces étapes pour proposer des changements :

1. Forkez ce dépôt.
2. Créez une branche avec une description claire de vos modifications :  
   ```bash
   git checkout -b feature/nom-de-la-feature
   ```

3. Faites vos modifications et validez-les :  
   ```bash
   git commit -m "Ajout d'une nouvelle fonctionnalité"
   ```

4. Poussez sur votre fork et soumettez une pull request.

---

## Auteurs
