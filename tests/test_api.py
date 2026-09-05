"""Tests para la API de inferencia."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from facial_attributes.api.dependencies import (
    ATTRIBUTE_DISPLAY_NAMES,
    CELEBA_ATTRIBUTE_NAMES,
)
from facial_attributes.api.main import app


@pytest.fixture()
def client() -> TestClient:
    """Cliente de prueba con lifespan deshabilitado."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """Imagen de ejemplo en bytes JPEG."""
    img = Image.new("RGB", (100, 100), color=(200, 180, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestHealthEndpoint:
    """Tests para GET /api/health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient) -> None:
        response = client.get("/api/health")
        data = response.json()
        assert "status" in data
        assert "device" in data
        assert "model_loaded" in data
        assert "num_attributes" in data
        assert "face_detector_available" in data

    def test_health_status_ok(self, client: TestClient) -> None:
        response = client.get("/api/health")
        data = response.json()
        assert data["status"] == "ok"
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["num_attributes"], int)


class TestAttributesEndpoint:
    """Tests para GET /api/attributes."""

    def test_attributes_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/attributes")
        assert response.status_code == 200

    def test_attributes_count(self, client: TestClient) -> None:
        response = client.get("/api/attributes")
        data = response.json()
        assert data["num_attributes"] == 40
        assert len(data["attributes"]) == 40

    def test_attributes_structure(self, client: TestClient) -> None:
        response = client.get("/api/attributes")
        data = response.json()
        for attr in data["attributes"]:
            assert "name" in attr
            assert "display_name" in attr
            assert "index" in attr

    def test_attributes_names_match_celeba(self, client: TestClient) -> None:
        response = client.get("/api/attributes")
        data = response.json()
        names = [a["name"] for a in data["attributes"]]
        assert names == CELEBA_ATTRIBUTE_NAMES

    def test_attributes_have_display_names(self, client: TestClient) -> None:
        response = client.get("/api/attributes")
        data = response.json()
        for attr in data["attributes"]:
            assert attr["name"] in ATTRIBUTE_DISPLAY_NAMES


class TestPredictEndpoint:
    """Tests para POST /api/predict."""

    def test_predict_returns_200(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        response = client.post(
            "/api/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_predict_response_structure(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        response = client.post(
            "/api/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert "faces" in data
        assert "num_faces_detected" in data
        assert "num_faces_with_predictions" in data
        assert "inference_time_ms" in data
        assert "image_size" in data
        assert "error" in data

    def test_predict_without_model_returns_error(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        response = client.post(
            "/api/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        data = response.json()
        assert data["error"] is not None

    def test_predict_invalid_content_type(self, client: TestClient) -> None:
        response = client.post(
            "/api/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_rejects_non_image_content_type(self, client: TestClient) -> None:
        response = client.post(
            "/api/predict",
            files={"file": ("data.csv", b"a,b,c", "text/csv")},
        )
        assert response.status_code == 400


class TestSchemas:
    """Tests para los modelos Pydantic."""

    def test_attribute_info_creation(self) -> None:
        from facial_attributes.api.schemas import AttributeInfo

        attr = AttributeInfo(
            name="Smiling",
            display_name="Sonriente",
            index=0,
        )
        assert attr.name == "Smiling"
        assert attr.index == 0

    def test_health_response_defaults(self) -> None:
        from facial_attributes.api.schemas import HealthResponse

        resp = HealthResponse(
            status="ok",
            device="cpu",
            model_loaded=False,
            num_attributes=40,
            face_detector_available=False,
        )
        assert resp.status == "ok"
        assert resp.model_loaded is False


class TestDependencies:
    """Tests para el módulo de dependencias."""

    def test_celeba_attributes_count(self) -> None:
        assert len(CELEBA_ATTRIBUTE_NAMES) == 40

    def test_display_names_cover_all_attributes(self) -> None:
        for name in CELEBA_ATTRIBUTE_NAMES:
            assert name in ATTRIBUTE_DISPLAY_NAMES

    def test_get_pipeline_without_init_raises(self) -> None:
        import facial_attributes.api.dependencies as deps
        from facial_attributes.api.dependencies import get_pipeline

        original = deps._PIPELINE
        deps._PIPELINE = None
        try:
            with pytest.raises(RuntimeError, match="Pipeline no inicializado"):
                get_pipeline()
        finally:
            deps._PIPELINE = original
