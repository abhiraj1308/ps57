import React, { useState, useEffect } from 'react';
import MapViewer from './MapViewer';

export default function Dashboard() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [surveyId, setSurveyId] = useState<string | null>(null);
    const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'complete'>('idle');
    const [progress, setProgress] = useState<number>(0);

    // Mock GeoJSON output matching your Geospatial Engineer's expected format
    const mockGeoData = {
        type: "FeatureCollection",
        features: [
            {
                type: "Feature",
                properties: { type: "Ghost Net", priority: "HIGH", confidence: 94.1 },
                geometry: { type: "Point", coordinates: [88.365812, 22.573421] }
            },
            {
                type: "Feature",
                properties: { type: "Cylinder/Pipe", priority: "MEDIUM", confidence: 87.4 },
                geometry: { type: "Point", coordinates: [88.375812, 22.583421] }
            }
        ]
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files.length > 0) {
            setSelectedFile(event.target.files[0]);
        }
    };

    const handleUpload = () => {
        if (!selectedFile) {
            alert("Please select a raw Sonar (XTF) file first.");
            return;
        }
        setStatus('uploading');

        // Simulate network upload time
        setTimeout(() => {
            setSurveyId("ps57-survey-1308");
            setStatus('processing');
        }, 1500);
    };

    // Pipeline progression simulation
    useEffect(() => {
        let interval: number; // Fixed TypeScript error here
        if (status === 'processing') {
            interval = window.setInterval(() => {
                setProgress((prev) => {
                    if (prev >= 100) {
                        setStatus('complete');
                        window.clearInterval(interval);
                        return 100;
                    }
                    return prev + 20; // 5 stages, 20% each
                });
            }, 1500);
        }
        return () => {
            if (interval) window.clearInterval(interval);
        };
    }, [status]);

    return (
        <div className="p-8 bg-gray-900 text-white min-h-screen font-sans">
            <h1 className="text-3xl font-bold mb-8 border-b border-gray-700 pb-4">PS57 MARINE INTELLIGENCE</h1>

            {status === 'idle' && (
                <div className="bg-gray-800 p-8 rounded-lg border border-gray-700 text-center">
                    <h2 className="text-xl mb-4 font-semibold">Upload New Survey Data</h2>
                    <div className="border-2 border-dashed border-gray-500 rounded-lg p-10 mb-6 bg-gray-900">
                        <input
                            type="file"
                            accept=".xtf,.png,.jpg,.jpeg"
                            onChange={handleFileChange}
                            className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-500 cursor-pointer"
                        />
                        {selectedFile && (
                            <p className="mt-4 text-green-400 font-medium">Selected: {selectedFile.name}</p>
                        )}
                    </div>
                    <button
                        onClick={handleUpload}
                        disabled={!selectedFile}
                        className={`px-8 py-3 rounded font-bold transition ${selectedFile ? 'bg-blue-600 hover:bg-blue-500' : 'bg-gray-600 cursor-not-allowed'}`}
                    >
                        Upload & Process Data
                    </button>
                </div>
            )}

            {status === 'uploading' && (
                <div className="bg-gray-800 p-8 rounded-lg border border-gray-700 text-center">
                    <h2 className="text-xl font-bold text-blue-400 animate-pulse">UPLOADING SECURELY TO BACKEND...</h2>
                </div>
            )}

            {status === 'processing' && (
                <div className="bg-gray-800 p-8 rounded-lg border border-gray-700">
                    <h2 className="text-xl font-bold mb-6 text-blue-400">AUTOMATED PROCESSING PIPELINE</h2>
                    <ul className="space-y-4 mb-6 font-medium text-gray-300">
                        <li className={`flex justify-between ${progress >= 20 ? 'text-green-400' : ''}`}><span>1. XTF INGESTION & PARSING</span> {progress >= 20 ? '✓' : '○'}</li>
                        <li className={`flex justify-between ${progress >= 40 ? 'text-green-400' : ''}`}><span>2. SONAR IMAGE PRE-PROCESSING</span> {progress >= 40 ? '✓' : '○'}</li>
                        <li className={`flex justify-between ${progress >= 60 ? 'text-green-400' : (progress === 40 ? 'text-yellow-400 animate-pulse' : '')}`}><span>3. AI/ML ANOMALY DETECTION</span> {progress >= 60 ? '✓' : '○'}</li>
                        <li className={`flex justify-between ${progress >= 80 ? 'text-green-400' : (progress === 60 ? 'text-yellow-400 animate-pulse' : '')}`}><span>4. DIE CONFIDENCE FILTERING</span> {progress >= 80 ? '✓' : '○'}</li>
                        <li className={`flex justify-between ${progress >= 100 ? 'text-green-400' : (progress === 80 ? 'text-yellow-400 animate-pulse' : '')}`}><span>5. GEOSPATIAL MAPPING</span> {progress >= 100 ? '✓' : '○'}</li>
                    </ul>
                    <div className="w-full bg-gray-700 rounded-full h-3">
                        <div className="bg-blue-500 h-3 rounded-full transition-all duration-500" style={{ width: `${progress}%` }}></div>
                    </div>
                </div>
            )}

            {status === 'complete' && (
                <div className="bg-gray-800 p-8 rounded-lg border border-gray-700">
                    <h2 className="font-bold text-2xl mb-6 text-green-400">SURVEY ANALYSIS COMPLETE</h2>

                    <div className="grid grid-cols-3 gap-4 text-center mb-6 bg-gray-900 p-4 rounded border border-gray-700">
                        <div>Total Anomalies: <span className="block text-2xl font-bold text-blue-400">27</span></div>
                        <div>Critical Hazards: <span className="block text-2xl font-bold text-red-500">8</span></div>
                        <div>System Confidence: <span className="block text-2xl font-bold text-green-400">91.4%</span></div>
                    </div>

                    <MapViewer geoData={mockGeoData} />

                    <div className="mt-8 flex gap-4">
                        <button className="bg-blue-600 px-6 py-2 rounded hover:bg-blue-500 font-bold shadow-lg">View Sonar Overlays</button>
                        <button className="bg-green-600 px-6 py-2 rounded hover:bg-green-500 font-bold shadow-lg">Export JSON Report</button>
                    </div>
                </div>
            )}
        </div>
    );
}