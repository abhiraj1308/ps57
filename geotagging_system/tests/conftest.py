import pytest
import json
import csv
import math
import os
from datetime import datetime, timezone, timedelta
from typing import List
from pathlib import Path

from src.models import (
    PingMetadata, Detection, SonarImageInfo, SonarSystemConfig,
    MissionMetadata, GeolocatedDetection, Geolocation, Dimensions,
    QualityFlag
)

@pytest.fixture
def sample_sonar_config():
    """Klein 3000 sonar configuration."""
    return SonarSystemConfig(
        name="Klein System 3000",
        frequency_khz=445,
        max_range_m=150,
        beam_width_deg=0.2,
        towfish_depth_offset_m=5.0,
        towpoint_offset_stern_m=15.0,
        catenary_factor=0.90,
    )

@pytest.fixture
def sample_image_info():
    """Standard sonar image dimensions."""
    return SonarImageInfo(
        width_pixels=4096,
        height_pixels=10000,
        samples_per_ping=2048,  # per side (port/stbd)
        range_m=150.0
    )

@pytest.fixture
def synthetic_pings():
    """
    Generate 10,000 pings along a realistic survey track.
    Survey area: Tampa Bay, FL (27.8°N, -82.5°W)
    Vessel speed: ~4 knots, heading NE then turning SE
    Ping rate: 10 Hz (0.1s interval)
    Altitude: ~15m with slight variations
    Depth: ~2m (towfish)
    
    Track: Start at (27.80, -82.50), head NE at 045° for 5000 pings,
    then turn to 135° SE for 5000 pings.
    Speed: 4 knots = ~2.06 m/s, ping interval 0.1s = ~0.206m between pings
    """
    pings = []
    base_lat, base_lon = 27.80, -82.50
    base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    speed_ms = 2.06  # 4 knots in m/s
    ping_interval = 0.1  # seconds
    distance_per_ping = speed_ms * ping_interval
    
    lat, lon = base_lat, base_lon
    heading = 45.0  # NE
    
    for i in range(10000):
        if i == 5000:  # Turn at halfway
            heading = 135.0  # SE
        
        # Simple flat-earth approximation for test data
        dlat = distance_per_ping * math.cos(math.radians(heading)) / 111320.0
        dlon = distance_per_ping * math.sin(math.radians(heading)) / (111320.0 * math.cos(math.radians(lat)))
        lat += dlat
        lon += dlon
        
        # Add slight noise
        import random
        random.seed(i)
        alt_noise = random.gauss(0, 0.3)
        
        ping = PingMetadata(
            ping_number=i,
            timestamp=base_time + timedelta(seconds=i * ping_interval),
            latitude=lat,
            longitude=lon,
            heading_deg=heading + random.gauss(0, 0.5) % 360,
            speed_knots=4.0 + random.gauss(0, 0.2),
            altitude_m=max(1.0, 15.0 + alt_noise),
            depth_m=2.0 + random.gauss(0, 0.1),
            slant_range_m=150.0,
            swath_width_m=300.0,
            cable_out_m=50.0,
            num_samples=2048,
            quality_flags=[QualityFlag.GPS_VALID, QualityFlag.ALTITUDE_VALID]
        )
        pings.append(ping)
    
    return pings

@pytest.fixture
def synthetic_detections():
    """
    Generate 50 synthetic detections at known positions.
    Mix of ghost_net, pipe, wreck, tire, debris classes.
    Spread across the ping range with varying confidences.
    """
    detections = []
    import random
    random.seed(42)
    classes = ['ghost_net', 'pipe', 'wreck', 'tire', 'debris']
    
    for i in range(50):
        ping_start = random.randint(100, 9800)
        ping_end = ping_start + random.randint(10, 50)
        x_center = random.randint(200, 3896)
        y_center = (ping_start + ping_end) // 2
        width = random.randint(20, 200)
        height = random.randint(10, 100)
        
        det = Detection(
            detection_id=f"DET_{i:03d}",
            bbox_pixels=[
                x_center - width // 2,
                y_center - height // 2,
                x_center + width // 2,
                y_center + height // 2
            ],
            confidence=round(random.uniform(0.4, 0.99), 2),
            class_label=random.choice(classes),
            ping_range=[ping_start, ping_end]
        )
        detections.append(det)
    
    return detections

@pytest.fixture
def sample_mission_metadata():
    return MissionMetadata(
        survey_id="SSS_2024_001",
        date="2024-01-15",
        vessel="RV_Surveyor",
        sonar_system="Klein_System_3000",
        coordinate_system="WGS84"
    )

@pytest.fixture
def sample_csv_file(tmp_path, synthetic_pings):
    """Create a sample CSV sonar log file from synthetic pings."""
    csv_path = tmp_path / "test_sonar_log.csv"
    # Write first 100 pings to CSV for quick testing
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'ping_number', 'timestamp', 'latitude', 'longitude',
            'heading', 'speed_knots', 'altitude', 'depth',
            'slant_range', 'cable_out'
        ])
        writer.writeheader()
        for ping in synthetic_pings[:100]:
            writer.writerow({
                'ping_number': ping.ping_number,
                'timestamp': ping.timestamp.isoformat(),
                'latitude': ping.latitude,
                'longitude': ping.longitude,
                'heading': ping.heading_deg,
                'speed_knots': ping.speed_knots,
                'altitude': ping.altitude_m,
                'depth': ping.depth_m,
                'slant_range': ping.slant_range_m,
                'cable_out': ping.cable_out_m,
            })
    return csv_path

@pytest.fixture
def sample_json_file(tmp_path, synthetic_pings):
    """Create a sample JSON metadata file."""
    json_path = tmp_path / "test_metadata.json"
    data = {
        "metadata": {
            "survey_id": "SSS_2024_001",
            "sonar_system": "Klein 3000"
        },
        "pings": [
            {
                "ping_number": p.ping_number,
                "timestamp": p.timestamp.isoformat(),
                "latitude": p.latitude,
                "longitude": p.longitude,
                "heading": p.heading_deg,
                "speed": p.speed_knots,
                "altitude": p.altitude_m,
                "depth": p.depth_m,
                "slant_range": p.slant_range_m,
                "cable_out": p.cable_out_m,
            }
            for p in synthetic_pings[:100]
        ]
    }
    json_path.write_text(json.dumps(data, indent=2))
    return json_path

@pytest.fixture
def sample_nmea_file(tmp_path):
    """Create a sample NMEA log file with GGA and RMC sentences."""
    nmea_path = tmp_path / "test_nav.nmea"
    lines = []
    base_lat = 27.80
    base_lon = -82.50
    for i in range(20):
        time_str = f"{10:02d}{i:02d}{0:02d}.00"
        lat_nmea = _decimal_to_nmea_lat(base_lat + i * 0.0001)
        lon_nmea = _decimal_to_nmea_lon(base_lon + i * 0.0001)
        
        gga = f"$GPGGA,{time_str},{lat_nmea},N,{lon_nmea},W,1,08,1.0,15.0,M,0.0,M,,"
        gga += f"*{_nmea_checksum(gga[1:])}"
        lines.append(gga)
        
        rmc = f"$GPRMC,{time_str},A,{lat_nmea},N,{lon_nmea},W,4.0,045.0,150124,,,"
        rmc += f"*{_nmea_checksum(rmc[1:])}"
        lines.append(rmc)
    
    nmea_path.write_text('\n'.join(lines) + '\n')
    return nmea_path

def _decimal_to_nmea_lat(decimal_deg):
    d = int(abs(decimal_deg))
    m = (abs(decimal_deg) - d) * 60
    return f"{d:02d}{m:07.4f}"

def _decimal_to_nmea_lon(decimal_deg):
    d = int(abs(decimal_deg))
    m = (abs(decimal_deg) - d) * 60
    return f"{d:03d}{m:07.4f}"

def _nmea_checksum(sentence):
    cs = 0
    for char in sentence:
        if char == '*':
            break
        cs ^= ord(char)
    return f"{cs:02X}"
