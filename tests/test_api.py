from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_rejects_unsupported_extension():
    response = client.post("/upload", files={"file": ("logs.exe", b"data")})
    assert response.status_code == 415


def test_upload_uses_block_pipeline(monkeypatch):
    expected = {
        "total_lines": 3,
        "lines_without_block_id": 0,
        "unmatched_event_lines": 0,
        "matched_event_rate": 1.0,
        "total_blocks": 1,
        "low_context_blocks": 0,
        "insufficient_context": False,
        "confidence_warning": None,
        "model_name": "random_forest",
        "model_version": "hdfs-v1-rf-1",
        "anomaly_block_count": 1,
        "anomaly_rate": 1.0,
        "blocks": [{"block_id": "blk_1", "n_lines": 3, "anomaly": 1, "anomaly_score": 0.9}],
    }
    monkeypatch.setattr(main, "predict_blocks_from_lines", lambda lines, root: expected)
    response = client.post(
        "/upload",
        files={"file": ("sample.log", b"one\ntwo\nthree\n", "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "random_forest"
    assert body["total_blocks"] == 1
    assert body["anomaly_count"] == 1
