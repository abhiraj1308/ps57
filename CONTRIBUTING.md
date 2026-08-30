# PS57 Contribution Guide

## 1. Project Overview

PS57 is an AI-powered automated underwater marine debris and anomaly detection system using Side-Scan Sonar (SSS) imagery.

The overall system pipeline is:

```text
Side-Scan Sonar
      ↓
Sonar Image Processing
      ↓
AI/ML Detection
      ↓
Detection Intelligence
      ↓
Geospatial Processing
      ↓
PostgreSQL
      ↓
FastAPI
      ↓
React Dashboard
      ↓
Actionable Reports