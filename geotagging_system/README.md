# Geotagging System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Description
A robust backend system for processing marine sonar data (Side Scan Sonar) and accurately geolocating detected targets (e.g., shipwrecks, UXO, debris) from pixel coordinates to real-world geographic coordinates (WGS84). This system accounts for vessel position, heading, towfish layback, and sonar geometry.

## Architecture Overview
- **`src.models`**: Pydantic data structures
- **`src.parsers`**: Ingest CSV, JSON, NMEA formats
- **`src.transforms`**: Coordinate projections and layback calculations
- **`src.pipeline`**: Geotagging execution logic
- **`src.reporting`**: Output generators (JSON, CSV, GeoJSON) and QA/QC validation
- **`src.api`**: FastAPI REST endpoints
- **`tests`**: Pytest test suite

## Prerequisites
- Python 3.11+
- GDAL and PROJ libraries installed on the system (required for spatial transformations)

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
### Python API
```python
from src.pipeline.geotagging_engine import GeotaggingEngine
from src.models import MissionMetadata, SonarImageInfo

engine = GeotaggingEngine("config.yaml")

# 1. Load data
engine.load_sonar_metadata("pings.csv")

# 2. Process detections
mission = MissionMetadata(survey_id="SURVEY_001", date="2024-01-15", vessel="RV Explorer")
image_info = SonarImageInfo(width_pixels=4096, height_pixels=10000, samples_per_ping=2048, range_m=150.0)

results = engine.process_detections(detections_list, image_info)

# 3. Generate report
report = engine.generate_report(mission)
```

### Starting the REST API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check API health status |
| `/upload_sonar_data` | POST | Upload CSV/JSON sonar log files |
| `/task_status/{task_id}` | GET | Get processing task status |
| `/get_report/{task_id}` | GET | Retrieve final report (JSON) |

## Configuration
See `config.yaml` to adjust the sonar parameters and QA/QC thresholds.

## Docker Usage
```bash
docker build -t geotagging-system .
docker run -p 8000:8000 geotagging-system
```

## Testing
```bash
pytest tests/
```

## License
MIT License.
