import React from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

export default function MapViewer({ geoData }: { geoData: any }) {
    if (!geoData) {
        return (
            <div className="text-gray-400 p-4 border border-gray-700 rounded mt-4">
                Loading geospatial data...
            </div>
        );
    }

    return (
        <div className="w-full border border-gray-600 rounded overflow-hidden z-0 relative" style={{ height: '400px' }}>
            <MapContainer
                center={[22.5726, 88.3639]}
                zoom={12}
                style={{ height: '100%', width: '100%' }}
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <GeoJSON
                    data={geoData}
                    onEachFeature={(feature, layer) => {
                        if (feature.properties) {
                            layer.bindPopup(
                                `<div style="font-family: sans-serif;">
                                    <strong style="color: #d97706; font-size: 14px;">${feature.properties.type}</strong><br/>
                                    <b>Confidence:</b> ${feature.properties.confidence}%<br/>
                                    <b>Priority:</b> ${feature.properties.priority}
                                </div>`
                            );
                        }
                    }}
                />
            </MapContainer>
        </div>
    );
}