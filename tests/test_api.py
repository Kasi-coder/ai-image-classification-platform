from fastapi.testclient import TestClient
from api.main import app

def test_health():
    c=TestClient(app)
    r=c.get("/health")
    assert r.status_code==200
    assert r.json()["status"]=="ok"

def test_classes():
    c=TestClient(app)
    r=c.get("/classes")
    assert r.status_code==200
    assert len(r.json()["classes"])==10
