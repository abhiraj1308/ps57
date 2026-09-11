"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

class TestAPI:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
    
    def test_upload_sonar_data(self, client, sample_csv_file):
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/upload_sonar_data",
                files={"file": ("test.csv", f, "text/csv")},
                data={"survey_id": "TEST_001", "sonar_system": "klein_3000"}
            )
        assert response.status_code == 202
        data = response.json()
        assert 'task_id' in data
    
    def test_get_report_not_found(self, client):
        response = client.get("/get_report/NONEXISTENT")
        assert response.status_code == 404
    
    def test_task_status_not_found(self, client):
        response = client.get("/task_status/nonexistent-id")
        assert response.status_code == 404
