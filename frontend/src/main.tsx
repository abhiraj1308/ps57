import { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";

import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./style.css";

/*
 * React-Leaflet v5 + TypeScript compatibility
 */
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

  const API_URL = "http://localhost:8000";

  const fetchData = async () => {
    try {
      const healthResponse = await fetch(`${API_URL}/health`);

      if (!healthResponse.ok) {
        throw new Error("API offline");
      }

      setApiOnline(true);

      const response = await fetch(`${API_URL}/detections`);

      if (!response.ok) {
        throw new Error("Could not fetch detections");
      }

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

  const totalDetections = detections.length;

  const highPriority = detections.filter(
    (detection) =>
      detection.priority.toLowerCase() === "high"
  ).length;

  const newDetections = detections.filter(
    (detection) =>
      detection.status.toLowerCase() === "new"
  ).length;

  const averageConfidence =
    detections.length > 0
      ? detections.reduce(
          (sum, detection) =>
            sum + detection.confidence,
          0
        ) / detections.length
      : 0;

  const mapCenter: [number, number] =
    detections.length > 0
      ? [
          detections[0].latitude,
          detections[0].longitude,
        ]
      : [20.5937, 78.9629];

  return (
    <div className="app">

      {/* SIDEBAR */}
      <aside className="sidebar">

        <div className="logo">
          <div className="logoMark">
            PS
          </div>

          <div className="logoText">
            <h1>PS57</h1>
            <span>MARINE AI</span>
          </div>
        </div>

        <nav className="navigation">

          <div className="navItem active">
            <span className="navIcon">◉</span>
            <span>Dashboard</span>
          </div>

          <div className="navItem">
            <span className="navIcon">⌁</span>
            <span>Detections</span>
          </div>

          <div className="navItem">
            <span className="navIcon">⌖</span>
            <span>Geospatial</span>
          </div>

          <div className="navItem">
            <span className="navIcon">◈</span>
            <span>Sonar Analysis</span>
          </div>

          <div className="navItem">
            <span className="navIcon">▣</span>
            <span>Reports</span>
          </div>

        </nav>

        <div className="sidebarBottom">

          <div className="systemLabel">
            SYSTEM STATUS
          </div>

          <div className="systemStatus">

            <span
              className={
                apiOnline
                  ? "statusDot online"
                  : "statusDot offline"
              }
            />

            <span>
              {apiOnline
                ? "All systems operational"
                : "API disconnected"}
            </span>

          </div>

        </div>

      </aside>

      {/* MAIN */}
      <main className="main">

        {/* HEADER */}
        <header className="header">

          <div className="headerText">

            <p className="eyebrow">
              AUTONOMOUS MARINE INTELLIGENCE
            </p>

            <h2>
              Detection Dashboard
            </h2>

            <p className="subtitle">
              AI-powered underwater debris and
              anomaly monitoring
            </p>

          </div>

          <div className="headerRight">

            <div
              className={
                apiOnline
                  ? "apiBadge onlineBadge"
                  : "apiBadge offlineBadge"
              }
            >

              <span
                className={
                  apiOnline
                    ? "statusDot online"
                    : "statusDot offline"
                }
              />

              {apiOnline
                ? "API ONLINE"
                : "API OFFLINE"}

            </div>

            <button
              className="refreshButton"
              onClick={fetchData}
            >
              ↻ Refresh
            </button>

          </div>

        </header>

        {/* STATISTICS */}
        <section className="statsGrid">

          <div className="statCard">

            <div className="statTop">
              <span>Total Detections</span>

              <span className="statIcon">
                ◎
              </span>
            </div>

            <strong>
              {totalDetections}
            </strong>

            <p>
              Objects detected
            </p>

          </div>

          <div className="statCard dangerCard">

            <div className="statTop">
              <span>High Priority</span>

              <span className="statIcon">
                !
              </span>
            </div>

            <strong>
              {highPriority}
            </strong>

            <p>
              Requires attention
            </p>

          </div>

          <div className="statCard warningCard">

            <div className="statTop">
              <span>New Detections</span>

              <span className="statIcon">
                ✦
              </span>
            </div>

            <strong>
              {newDetections}
            </strong>

            <p>
              Awaiting validation
            </p>

          </div>

          <div className="statCard">

            <div className="statTop">
              <span>Avg. Confidence</span>

              <span className="statIcon">
                ◉
              </span>
            </div>

            <strong>
              {(averageConfidence * 100).toFixed(1)}%
            </strong>

            <p>
              AI model confidence
            </p>

          </div>

        </section>

        {/* MAP + AI */}
        <section className="contentGrid">

          {/* MAP */}
          <div className="panel mapPanel">

            <div className="panelHeader">

              <div>
                <h3>
                  Detection Map
                </h3>

                <p>
                  Geographic distribution of
                  detected objects
                </p>
              </div>

              <span className="liveBadge">
                ● LIVE
              </span>

            </div>

            <div className="map">

              <LeafletMapContainer
                center={mapCenter}
                zoom={5}
                scrollWheelZoom={true}
                style={{
                  width: "100%",
                  height: "100%",
                }}
              >

                <LeafletTileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {detections.map(
                  (detection) => (

                    <LeafletMarker
                      key={detection.id}
                      position={[
                        detection.latitude,
                        detection.longitude,
                      ]}
                    >

                      <LeafletPopup>

                        <div className="popupContent">

                          <strong>
                            {detection.class_name}
                          </strong>

                          <br />

                          Confidence:{" "}
                          {(
                            detection.confidence * 100
                          ).toFixed(1)}
                          %

                          <br />

                          Priority:{" "}
                          {detection.priority}

                          <br />

                          Status:{" "}
                          {detection.status}

                          <br />

                          Location:{" "}
                          {detection.latitude.toFixed(5)}
                          ,{" "}
                          {detection.longitude.toFixed(5)}

                          <br />

                          Size:{" "}
                          {detection.width} ×{" "}
                          {detection.height}

                        </div>

                      </LeafletPopup>

                    </LeafletMarker>

                  )
                )}

              </LeafletMapContainer>

            </div>

          </div>

          {/* AI INTELLIGENCE */}
          <div className="panel intelligencePanel">

            <div className="panelHeader">

              <div>

                <h3>
                  AI Intelligence
                </h3>

                <p>
                  Current detection overview
                </p>

              </div>

            </div>

            <div className="intelligenceContent">

              {detections.length === 0 ? (

                <div className="emptyState">

                  <div className="emptyIcon">
                    ◌
                  </div>

                  <h4>
                    No detections
                  </h4>

                  <p>
                    The system has not detected
                    any objects yet.
                  </p>

                </div>

              ) : (

                <>

                  <div className="bigNumber">
                    {detections.length}
                  </div>

                  <p className="bigNumberLabel">
                    Active detection records
                  </p>

                  <div className="progressBlock">

                    <div className="progressLabel">

                      <span>
                        Average confidence
                      </span>

                      <span>
                        {(averageConfidence * 100).toFixed(1)}%
                      </span>

                    </div>

                    <div className="progress">

                      <div
                        className="progressFill"
                        style={{
                          width: `${Math.min(
                            averageConfidence * 100,
                            100
                          )}%`,
                        }}
                      />

                    </div>

                  </div>

                  <div className="prioritySummary">

                    <div>

                      <span className="priorityDot high" />

                      <span>
                        High priority
                      </span>

                      <strong>
                        {highPriority}
                      </strong>

                    </div>

                    <div>

                      <span className="priorityDot normal" />

                      <span>
                        Other
                      </span>

                      <strong>
                        {totalDetections - highPriority}
                      </strong>

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

              <h3>
                Recent Detections
              </h3>

              <p>
                Latest objects identified by
                PS57 intelligence
              </p>

            </div>

            <span className="updated">
              Last updated:{" "}
              {lastUpdated || "—"}
            </span>

          </div>

          {loading ? (

            <div className="loading">
              Loading detection data...
            </div>

          ) : detections.length === 0 ? (

            <div className="emptyTable">
              No detection records available.
            </div>

          ) : (

            <div className="tableWrapper">

              <table>

                <thead>

                  <tr>
                    <th>ID</th>
                    <th>CLASS</th>
                    <th>CONFIDENCE</th>
                    <th>LOCATION</th>
                    <th>SIZE</th>
                    <th>STATUS</th>
                    <th>PRIORITY</th>
                  </tr>

                </thead>

                <tbody>

                  {detections.map(
                    (detection) => (

                      <tr key={detection.id}>

                        <td>
                          <span className="id">
                            #{detection.id}
                          </span>
                        </td>

                        <td>
                          <strong>
                            {detection.class_name}
                          </strong>
                        </td>

                        <td>

                          <div className="confidence">

                            <span>
                              {(
                                detection.confidence * 100
                              ).toFixed(1)}
                              %
                            </span>

                            <div className="miniProgress">

                              <div
                                style={{
                                  width: `${Math.min(
                                    detection.confidence *
                                      100,
                                    100
                                  )}%`,
                                }}
                              />

                            </div>

                          </div>

                        </td>

                        <td>

                          <span className="coordinates">

                            {detection.latitude.toFixed(5)}

                            <br />

                            {detection.longitude.toFixed(5)}

                          </span>

                        </td>

                        <td>
                          {detection.width} ×{" "}
                          {detection.height}
                        </td>

                        <td>

                          <span className="statusBadge">
                            {detection.status}
                          </span>

                        </td>

                        <td>

                          <span
                            className={
                              detection.priority.toLowerCase() ===
                              "high"
                                ? "priorityBadge highPriority"
                                : "priorityBadge"
                            }
                          >
                            {detection.priority}
                          </span>

                        </td>

                      </tr>

                    )
                  )}

                </tbody>

              </table>

            </div>

          )}

        </section>

        {/* FOOTER */}
        <footer>

          <span>
            PS57 Marine Intelligence Platform
          </span>

          <span>
            AI Detection Engine • PostgreSQL • FastAPI
          </span>

        </footer>

      </main>

    </div>
  );
}

ReactDOM.createRoot(
  document.getElementById("root")!
).render(
  <App />
);