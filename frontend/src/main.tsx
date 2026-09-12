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
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiOnline, setApiOnline] = useState(false);
  const [lastUpdated, setLastUpdated] = useState("");

  const [currentView, setCurrentView] = useState<'dashboard' | 'detections' | 'geospatial' | 'sonar' | 'reports'>('dashboard');
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [surveyId, setSurveyId] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'complete'>('idle');
  const [progress, setProgress] = useState<number>(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [inspectDetection, setInspectDetection] = useState<Detection | null>(null);

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  const downloadCSV = () => {
    if (detections.length === 0) return alert("No data to export");
    const headers = ["ID", "Class", "Confidence", "Latitude", "Longitude", "Status", "Priority"];
    const rows = detections.map(d => [d.id, d.class_name, (d.confidence * 100).toFixed(1) + "%", d.latitude.toFixed(5), d.longitude.toFixed(5), d.status, d.priority].join(","));
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows].join("\\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "ps57_detections_report.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

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

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFiles(event.target.files);
      setUploadError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFiles) return;
    setUploadStatus('uploading');
    setUploadError(null);

    const formData = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
        formData.append("files", selectedFiles[i]);
    }

    try {
      setUploadStatus('processing');
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      
      if (data.status === "error") {
          setUploadError(data.message);
          setUploadStatus('idle');
          return;
      }
      
      if (!response.ok) throw new Error("Upload failed");
      
      setSurveyId(data.filename);
      setTimeout(() => {
        setUploadStatus('complete');
        fetchData();
        setCurrentView('dashboard');
      }, 500);

    } catch (error) {
      console.error(error);
      setUploadError("A network error occurred. Is the API sleeping?");
      setUploadStatus('idle');
    }
  };

  useEffect(() => {
    let interval: number;
    if (uploadStatus === 'processing') {
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

  const totalDetections = detections.length;
  const highPriority = detections.filter(d => d.priority.toLowerCase() === 'high').length;
  const newDetections = detections.filter(d => d.status.toLowerCase() === 'new').length;
  const averageConfidence = totalDetections > 0 
    ? detections.reduce((acc, curr) => acc + curr.confidence, 0) / totalDetections 
    : 0;

  const mapCenter: [number, number] = detections.length > 0
    ? [detections[0].latitude, detections[0].longitude]
    : [27.80, -82.50];

  const DetectionsTable = () => (
    <section className="panel detectionsPanel">
      <div className="panelHeader">
        <div>
          <h3>Detection Records</h3>
          <p>Objects identified by PS57 intelligence</p>
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
                <th>ID</th><th>CLASS</th><th>CONFIDENCE</th><th>LOCATION</th><th>SIZE</th><th>STATUS</th><th>PRIORITY</th><th>ACTION</th>
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
                  <td>{detection.width.toFixed(2)}m × {detection.height.toFixed(2)}m</td>
                  <td><span className="statusBadge">{detection.status}</span></td>
                  <td>
                    <span className={detection.priority.toLowerCase() === "high" ? "priorityBadge highPriority" : "priorityBadge"}>
                      {detection.priority}
                    </span>
                  </td>
                  <td>
                    <button onClick={() => setInspectDetection(detection)} style={{background: '#3b82f6', color: 'white', border: 'none', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '10px'}}>
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );

  const GeospatialMap = () => (
    <section className="panel mapPanel" style={{height: "600px"}}>
      <div className="panelHeader">
        <div>
          <h3>Geospatial Map</h3>
          <p>Geographic distribution of detected objects</p>
        </div>
        <span className="liveBadge">● LIVE</span>
      </div>
      <div className="map" style={{height: "500px"}}>
        <LeafletMapContainer center={mapCenter} zoom={6} scrollWheelZoom={true} style={{ width: "100%", height: "100%" }}>
          <LeafletTileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {detections.map((detection) => (
            <LeafletMarker key={detection.id} position={[detection.latitude, detection.longitude]}>
              <LeafletPopup>
                <div className="popupContent">
                  <strong>{detection.class_name}</strong><br />
                  Confidence: {(detection.confidence * 100).toFixed(1)}%<br />
                  Priority: {detection.priority}<br />
                  Status: {detection.status}<br />
                  Location: {detection.latitude.toFixed(5)}, {detection.longitude.toFixed(5)}
                </div>
              </LeafletPopup>
            </LeafletMarker>
          ))}
        </LeafletMapContainer>
      </div>
    </section>
  );

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logoContainer">
          <div className="logoMark">PS</div>
          <h1>PS57 Analytics</h1>
        </div>
        
        <ul className="navigation">
          <li className={currentView === 'dashboard' ? 'active' : ''} onClick={() => setCurrentView('dashboard')}>
            <span className="navIcon">⊞</span> Dashboard
          </li>
          <li className={currentView === 'detections' ? 'active' : ''} onClick={() => setCurrentView('detections')}>
            <span className="navIcon">⌖</span> Detections
          </li>
          <li className={currentView === 'geospatial' ? 'active' : ''} onClick={() => setCurrentView('geospatial')}>
            <span className="navIcon">📍</span> Geospatial
          </li>
          <li className={currentView === 'sonar' ? 'active' : ''} onClick={() => setCurrentView('sonar')}>
            <span className="navIcon">🌊</span> Sonar Analysis
          </li>
          <li className={currentView === 'reports' ? 'active' : ''} onClick={() => setCurrentView('reports')}>
            <span className="navIcon">📄</span> Reports
          </li>
        </ul>

        <div className="sidebarBottom">
          <div className="systemLabel">SYSTEM STATUS</div>
          <div className="systemStatus">
            <span className={apiOnline ? "statusDot online" : "statusDot offline"} />
            <span style={{fontSize: "0.85rem"}}>{apiOnline ? "All systems operational" : "Server Sleeping (Upload to wake)"}</span>
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="header">
          <div>
            <h2>{currentView.charAt(0).toUpperCase() + currentView.slice(1)} Overview</h2>
            <p className="subtitle">AI-Powered Underwater Anomaly Detection</p>
          </div>
          
          <div className="headerRight">
            <div className={apiOnline ? "apiBadge onlineBadge" : "apiBadge offlineBadge"}>
              <span className={apiOnline ? "statusDot online" : "statusDot offline"} />
              {apiOnline ? "API ONLINE" : "SERVER SLEEPING"}
            </div>
            <button className="refreshButton" onClick={fetchData}>↻ Refresh</button>
            <button className="primaryButton" onClick={() => setCurrentView('sonar')}>+ New Upload</button>
          </div>
        </header>

        {currentView === 'sonar' && (
          <section className="uploadSection">
            <div className="uploadCard">
              <h3>Run New Sonar Analysis</h3>
              <p>Upload a batch of XTF or TIFF sonar logs to process them through the pipeline.</p>
              
              {uploadError && (
                  <div className="errorMessage">
                      {uploadError}
                  </div>
              )}

              <div className="uploadControls" style={{display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap', marginTop: '10px'}}>
                {/* File Input */}
                <input 
                  type="file" 
                  id="sonarFiles" 
                  multiple 
                  onChange={handleFileChange} 
                  style={{display: 'none'}}
                />
                <label htmlFor="sonarFiles" className="primaryButton" style={{cursor: 'pointer', background: '#3b82f6', display: 'inline-block', padding: '8px 12px'}}>
                  + Select Files
                </label>

                {/* Folder Input */}
                <input 
                  type="file" 
                  id="sonarFolder" 
                  multiple 
                  // @ts-ignore
                  webkitdirectory="" 
                  onChange={handleFileChange} 
                  style={{display: 'none'}}
                />
                <label htmlFor="sonarFolder" className="primaryButton" style={{cursor: 'pointer', background: '#6366f1', display: 'inline-block', padding: '8px 12px'}}>
                  + Select Folder
                </label>

                <span style={{color: '#94a3b8', fontSize: '13px', marginLeft: '10px'}}>
                  {selectedFiles 
                    ? `${selectedFiles.length} file(s) selected` 
                    : "No files chosen"}
                </span>

                <button 
                  className="uploadButton" 
                  onClick={handleUpload}
                  style={{marginLeft: 'auto', padding: '8px 16px'}}
                  disabled={!selectedFiles || uploadStatus === 'uploading' || uploadStatus === 'processing'}
                >
                  {uploadStatus === 'idle' || uploadStatus === 'complete' ? 'Upload & Process' : 'Processing...'}
                </button>
              </div>

              {uploadStatus !== 'idle' && (
                <div className="pipelineStatus">
                  <h4>Processing Pipeline</h4>
                  <div className="progressBarContainer">
                    <div className="progressBar" style={{ width: `${progress}%` }}></div>
                  </div>
                  <ul className="pipelineSteps">
                    <li className={progress >= 20 ? 'active' : ''}>1. File Upload</li>
                    <li className={progress >= 40 ? 'active' : ''}>2. Waterfall Generation</li>
                    <li className={progress >= 60 ? 'active' : ''}>3. AI Inference</li>
                    <li className={progress >= 80 ? 'active' : ''}>4. DIE Verification</li>
                    <li className={progress >= 90 ? 'active' : ''}>5. Geospatial Mapping</li>
                  </ul>
                  <p className="statusNote">Note: Render Free Tier may take up to 60 seconds to wake up the server during this step.</p>
                </div>
              )}
            </div>
          </section>
        )}

        {currentView === 'dashboard' && (
          <>
            <section className="statsGrid">
              <div className="statCard">
                <div className="statTop"><span>Total Detections</span><span className="statIcon">📊</span></div>
                <strong>{totalDetections}</strong>
                <p>Objects detected</p>
              </div>
              <div className="statCard dangerCard">
                <div className="statTop"><span>High Priority</span><span className="statIcon">🚨</span></div>
                <strong>{highPriority}</strong>
                <p>Requires attention</p>
              </div>
              <div className="statCard warningCard">
                <div className="statTop"><span>New Detections</span><span className="statIcon">⭐</span></div>
                <strong>{newDetections}</strong>
                <p>Awaiting validation</p>
              </div>
              <div className="statCard">
                <div className="statTop"><span>Avg. Confidence</span><span className="statIcon">💯</span></div>
                <strong>{(averageConfidence * 100).toFixed(1)}%</strong>
                <p>AI model confidence</p>
              </div>
            </section>

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
                            Size: {detection.width.toFixed(2)}m × {detection.height.toFixed(2)}m
                          </div>
                        </LeafletPopup>
                      </LeafletMarker>
                    ))}
                  </LeafletMapContainer>
                </div>
              </div>

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
                      <div className="emptyIcon">⭕</div>
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
            
            <DetectionsTable />
          </>
        )}

        {currentView === 'detections' && <DetectionsTable />}

        {currentView === 'geospatial' && <GeospatialMap />}

        {currentView === 'reports' && (
            <section className="panel">
                <div className="panelHeader">
                    <div>
                        <h3>Generate Reports</h3>
                        <p>Export analytics and detection records</p>
                    </div>
                </div>
                <div className="emptyState" style={{minHeight: "300px"}}>
                    <div className="emptyIcon">📄</div>
                    <h4>Export Data</h4>
                    <p>Download your latest sonar intelligence analysis.</p>
                    <button className="primaryButton" style={{marginTop: "20px"}} onClick={downloadCSV}>Download CSV</button>
                </div>
            </section>
        )}

        <footer>
          <span>PS57 Marine Intelligence Platform</span>
          <span>AI Detection Engine | PostgreSQL | FastAPI</span>
        </footer>
      </main>

      {/* INSPECT MODAL */}
      {inspectDetection && (
        <div style={{position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999}}>
          <div style={{background: '#1e293b', padding: '20px', borderRadius: '12px', width: '80%', maxWidth: '800px', border: '1px solid #334155'}}>
            <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '15px'}}>
              <h3 style={{margin: 0, color: '#f8fafc'}}>Sonar Imagery Inspect: #{inspectDetection.id}</h3>
              <button onClick={() => setInspectDetection(null)} style={{background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '20px'}}>&times;</button>
            </div>
            
            <div style={{background: '#0f172a', height: '400px', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed #475569', position: 'relative', overflow: 'hidden'}}>
               {/* Dummy Sonar Image Simulation */}
               <div style={{position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, background: 'repeating-linear-gradient(0deg, #1e293b, #1e293b 2px, #0f172a 2px, #0f172a 4px)', opacity: 0.5}}></div>
               <div style={{position: 'absolute', top: '40%', left: '40%', width: '20%', height: '20%', border: '2px solid #ef4444', background: 'rgba(239,68,68,0.2)'}}></div>
               <span style={{position: 'absolute', top: '35%', left: '40%', color: '#ef4444', fontSize: '12px', fontWeight: 'bold'}}>{inspectDetection.class_name} ({(inspectDetection.confidence*100).toFixed(1)}%)</span>
            </div>
            
            <div style={{marginTop: '15px', color: '#cbd5e1', fontSize: '13px', display: 'flex', justifyContent: 'space-between'}}>
              <span><strong>Location:</strong> {inspectDetection.latitude.toFixed(5)}, {inspectDetection.longitude.toFixed(5)}</span>
              <span><strong>Size:</strong> {inspectDetection.width.toFixed(2)}m x {inspectDetection.height.toFixed(2)}m</span>
              <span><strong>Priority:</strong> {inspectDetection.priority}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);