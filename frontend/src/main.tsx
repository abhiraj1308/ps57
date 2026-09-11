import { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./style.css";

const LeafletMapContainer = MapContainer as any;
const LeafletTileLayer = TileLayer as any;
const LeafletMarker = Marker as any;
const LeafletPopup = Popup as any;

type Detection = {
  id: number;
  class_name: string;
  confidence: number;
  latitude: number;
  longitude: number;
  width: number;
  height: number;
  status: string;
  priority: string;
};

function App() {
  // Original State
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiOnline, setApiOnline] = useState(false);
  const [lastUpdated, setLastUpdated] = useState("");

  // New Upload Pipeline State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [surveyId, setSurveyId] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'complete'>('idle');
  const [progress, setProgress] = useState<number>(0);

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  // Health and Data Fetching
  const fetchData = async () => {
    try {
      const healthResponse = await fetch(`${API_URL}/health`);
      if (!healthResponse.ok) throw new Error("API offline");
      setApiOnline(true);

      const response = await fetch(`${API_URL}/detections`);
      if (!response.ok) throw new Error("Could not fetch detections");

      const data: Detection[] = await response.json();
      setDetections(data);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      console.error("API error:", error);
      setApiOnline(false);
    } finally {
      setLoading(false);
    }
  };

  // Poll for health status
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Handle File Upload & Pipeline Simulation
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setUploadStatus('uploading');

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      setUploadStatus('processing');
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");
      
      const data = await response.json();
      console.log(data);
      
      setSurveyId(data.filename);
      // Wait a moment before completing
      setTimeout(() => {
        setUploadStatus('complete');
        fetchData();
      }, 500);

    } catch (error) {
      console.error(error);
      setUploadStatus('idle');
    }
  };

  // Progress Bar Simulation
  useEffect(() => {
    let interval: number;
    if (uploadStatus === 'processing') {
      // Fake progress up to 90%
      interval = window.setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return 90;
          return prev + 10;
        });
      }, 500);
    } else if (uploadStatus === 'complete') {
      setProgress(100);
    } else if (uploadStatus === 'idle') {
      setProgress(0);
    }
    return () => { if (interval) window.clearInterval(interval); };
  }, [uploadStatus]);

  // Calculations
  const totalDetections = detections.length;
  const highPriority = detections.filter((d) => d.priority.toLowerCase() === "high").length;
  const newDetections = detections.filter((d) => d.status.toLowerCase() === "new").length;
  const averageConfidence = detections.length > 0
    ? detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length
    : 0;
  const mapCenter: [number, number] = detections.length > 0
    ? [detections[0].latitude, detections[0].longitude]
    : [20.5937, 78.9629];

  return (
    <div className="app">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="logo">
          <div className="logoMark">PS</div>
          <div className="logoText">
            <h1>PS57</h1>
            <span>MARINE AI</span>
          </div>
        </div>
        <nav className="navigation">
          <div className="navItem active"><span className="navIcon">◉</span><span>Dashboard</span></div>
          <div className="navItem"><span className="navIcon">⌁</span><span>Detections</span></div>
          <div className="navItem"><span className="navIcon">⌖</span><span>Geospatial</span></div>
          <div className="navItem"><span className="navIcon">◈</span><span>Sonar Analysis</span></div>
          <div className="navItem"><span className="navIcon">▣</span><span>Reports</span></div>
        </nav>
        <div className="sidebarBottom">
          <div className="systemLabel">SYSTEM STATUS</div>
          <div className="systemStatus">
            <span className={apiOnline ? "statusDot online" : "statusDot offline"} />
            <span>{apiOnline ? "All systems operational" : "API disconnected"}</span>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main">
        {/* HEADER */}
        <header className="header">
          <div className="headerText">
            <p className="eyebrow">AUTONOMOUS MARINE INTELLIGENCE</p>
            <h2>Detection Dashboard</h2>
            <p className="subtitle">AI-powered underwater debris and anomaly monitoring</p>
          </div>
          <div className="headerRight">
            <div className={apiOnline ? "apiBadge onlineBadge" : "apiBadge offlineBadge"}>
              <span className={apiOnline ? "statusDot online" : "statusDot offline"} />
              {apiOnline ? "API ONLINE" : "API OFFLINE"}
            </div>
            <button className="refreshButton" onClick={fetchData}>↻ Refresh</button>
          </div>
        </header>

        {/* UPLOAD PIPELINE ZONE */}
        {uploadStatus === 'idle' && (
          <section className="panel mb-6 p-6 border border-gray-700 rounded bg-gray-800 text-center">
            <h3 className="text-lg font-bold text-white mb-4">Initialize New Sonar Survey</h3>
            <div className="border-2 border-dashed border-gray-500 rounded-lg p-8 mb-4">
              <input type="file" accept=".xtf,.png" onChange={handleFileChange} className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-blue-600 file:text-white cursor-pointer" />
            </div>
            <button onClick={handleUpload} disabled={!selectedFile} className={`px-6 py-2 rounded font-bold ${selectedFile ? 'bg-blue-600 text-white hover:bg-blue-500' : 'bg-gray-600 text-gray-400 cursor-not-allowed'}`}>
              Upload & Process XTF Data
            </button>
          </section>
        )}

        {uploadStatus === 'processing' && (
          <section className="panel mb-6 p-6 border border-gray-700 rounded bg-gray-800">
            <h3 className="text-lg font-bold text-blue-400 mb-4">PROCESSING PIPELINE</h3>
            <ul className="space-y-3 mb-4 text-gray-300">
              <li className="flex justify-between"><span>1. XTF Ingestion</span> <span>{progress >= 20 ? '✓' : '○'}</span></li>
              <li className="flex justify-between"><span>2. Sonar Pre-processing</span> <span>{progress >= 40 ? '✓' : '○'}</span></li>
              <li className="flex justify-between"><span>3. AI Anomaly Detection</span> <span>{progress >= 60 ? '✓' : '○'}</span></li>
              <li className="flex justify-between"><span>4. Confidence Filtering</span> <span>{progress >= 80 ? '✓' : '○'}</span></li>
              <li className="flex justify-between"><span>5. Geospatial Mapping</span> <span>{progress >= 100 ? '✓' : '○'}</span></li>
            </ul>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div className="bg-blue-500 h-2 rounded-full transition-all duration-500" style={{ width: `${progress}%` }}></div>
            </div>
          </section>
        )}

        {/* RENDER STATS, MAP, AND TABLE ONLY WHEN COMPLETE OR IF DATA ALREADY EXISTS */}
        {(uploadStatus === 'complete' || detections.length > 0) && (
          <>
            {/* STATISTICS */}
            <section className="statsGrid">
              <div className="statCard">
                <div className="statTop"><span>Total Detections</span><span className="statIcon">◎</span></div>
                <strong>{totalDetections}</strong>
                <p>Objects detected</p>
              </div>
              <div className="statCard dangerCard">
                <div className="statTop"><span>High Priority</span><span className="statIcon">!</span></div>
                <strong>{highPriority}</strong>
                <p>Requires attention</p>
              </div>
              <div className="statCard warningCard">
                <div className="statTop"><span>New Detections</span><span className="statIcon">✦</span></div>
                <strong>{newDetections}</strong>
                <p>Awaiting validation</p>
              </div>
              <div className="statCard">
                <div className="statTop"><span>Avg. Confidence</span><span className="statIcon">◉</span></div>
                <strong>{(averageConfidence * 100).toFixed(1)}%</strong>
                <p>AI model confidence</p>
              </div>
            </section>

            {/* MAP + AI */}
            <section className="contentGrid">
              <div className="panel mapPanel">
                <div className="panelHeader">
                  <div>
                    <h3>Detection Map</h3>
                    <p>Geographic distribution of detected objects</p>
                  </div>
                  <span className="liveBadge">● LIVE</span>
                </div>
                <div className="map">
                  <LeafletMapContainer center={mapCenter} zoom={5} scrollWheelZoom={true} style={{ width: "100%", height: "100%" }}>
                    <LeafletTileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                    {detections.map((detection) => (
                      <LeafletMarker key={detection.id} position={[detection.latitude, detection.longitude]}>
                        <LeafletPopup>
                          <div className="popupContent">
                            <strong>{detection.class_name}</strong><br />
                            Confidence: {(detection.confidence * 100).toFixed(1)}%<br />
                            Priority: {detection.priority}<br />
                            Status: {detection.status}<br />
                            Location: {detection.latitude.toFixed(5)}, {detection.longitude.toFixed(5)}<br />
                            Size: {detection.width} × {detection.height}
                          </div>
                        </LeafletPopup>
                      </LeafletMarker>
                    ))}
                  </LeafletMapContainer>
                </div>
              </div>

              {/* AI INTELLIGENCE */}
              <div className="panel intelligencePanel">
                <div className="panelHeader">
                  <div>
                    <h3>AI Intelligence</h3>
                    <p>Current detection overview</p>
                  </div>
                </div>
                <div className="intelligenceContent">
                  {detections.length === 0 ? (
                    <div className="emptyState">
                      <div className="emptyIcon">◌</div>
                      <h4>No detections</h4>
                      <p>The system has not detected any objects yet.</p>
                    </div>
                  ) : (
                    <>
                      <div className="bigNumber">{detections.length}</div>
                      <p className="bigNumberLabel">Active detection records</p>
                      <div className="progressBlock">
                        <div className="progressLabel">
                          <span>Average confidence</span>
                          <span>{(averageConfidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="progress">
                          <div className="progressFill" style={{ width: `${Math.min(averageConfidence * 100, 100)}%` }} />
                        </div>
                      </div>
                      <div className="prioritySummary">
                        <div>
                          <span className="priorityDot high" />
                          <span>High priority</span>
                          <strong>{highPriority}</strong>
                        </div>
                        <div>
                          <span className="priorityDot normal" />
                          <span>Other</span>
                          <strong>{totalDetections - highPriority}</strong>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </section>

            {/* DETECTION TABLE */}
            <section className="panel detectionsPanel">
              <div className="panelHeader">
                <div>
                  <h3>Recent Detections</h3>
                  <p>Latest objects identified by PS57 intelligence</p>
                </div>
                <span className="updated">Last updated: {lastUpdated || "—"}</span>
              </div>
              {loading ? (
                <div className="loading">Loading detection data...</div>
              ) : detections.length === 0 ? (
                <div className="emptyTable">No detection records available.</div>
              ) : (
                <div className="tableWrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>ID</th><th>CLASS</th><th>CONFIDENCE</th><th>LOCATION</th><th>SIZE</th><th>STATUS</th><th>PRIORITY</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detections.map((detection) => (
                        <tr key={detection.id}>
                          <td><span className="id">#{detection.id}</span></td>
                          <td><strong>{detection.class_name}</strong></td>
                          <td>
                            <div className="confidence">
                              <span>{(detection.confidence * 100).toFixed(1)}%</span>
                              <div className="miniProgress">
                                <div style={{ width: `${Math.min(detection.confidence * 100, 100)}%` }} />
                              </div>
                            </div>
                          </td>
                          <td>
                            <span className="coordinates">{detection.latitude.toFixed(5)}<br />{detection.longitude.toFixed(5)}</span>
                          </td>
                          <td>{detection.width} × {detection.height}</td>
                          <td><span className="statusBadge">{detection.status}</span></td>
                          <td>
                            <span className={detection.priority.toLowerCase() === "high" ? "priorityBadge highPriority" : "priorityBadge"}>
                              {detection.priority}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </>
        )}

        {/* FOOTER */}
        <footer>
          <span>PS57 Marine Intelligence Platform</span>
          <span>AI Detection Engine • PostgreSQL • FastAPI</span>
        </footer>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);