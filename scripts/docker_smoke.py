"""Smoke test de integración Docker (T-14.5).

Verifica que ``docker compose build && docker compose up`` levanta los
servicios de backend y frontend y que el health check de la API responde.

Requiere el Docker CLI corriendo localmente o en CI.

Uso:
    uv run python scripts/docker_smoke.py
    uv run python scripts/docker_smoke.py --skip-build
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"

PROJECT_NAME = "facial-attributes-smoke"
BACKEND_HEALTH_URL = "http://localhost:8000/api/health"
FRONTEND_URL = "http://localhost:3000/"


def _run(cmd: list[str], *, timeout: int | None = None) -> int:
    """Ejecutar comando heredando stdout/stderr."""
    print(f"==> {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, timeout=timeout)
    return result.returncode


def _show_logs(service: str) -> None:
    """Mostrar logs de un servicio para diagnóstico."""
    _run(["docker", "compose", "--project-name", PROJECT_NAME, "logs", service])


def _fetch_json(url: str, timeout: float) -> dict | None:
    """Consultar un endpoint JSON, devolviendo None si falla."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return dict(json.loads(resp.read().decode("utf-8")))
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def _wait_for_health(url: str, timeout: float) -> dict:
    """Esperar hasta que el health check responda con status=ok."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = _fetch_json(url, timeout=5.0)
        if data is not None and data.get("status") == "ok":
            return data
        time.sleep(5)
    raise TimeoutError(
        f"El backend no respondió status=ok en {url} tras {timeout:.0f}s"
    )


def _check_frontend(url: str) -> None:
    """Verificar que el frontend responde HTTP 200."""
    try:
        with urllib.request.urlopen(url, timeout=10.0) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    if status != 200:
        raise RuntimeError(f"El frontend respondió HTTP {status} en {url}")


def run_smoke_test(skip_build: bool, build_timeout: int, health_timeout: int) -> None:
    """Ejecutar el smoke test completo de los servicios Docker."""
    if not COMPOSE_FILE.exists():
        raise RuntimeError(f"No se encontró {COMPOSE_FILE}")

    _run(["docker", "compose", "version"])

    if not skip_build:
        if (
            _run(
                ["docker", "compose", "--project-name", PROJECT_NAME, "build"],
                timeout=build_timeout,
            )
            != 0
        ):
            raise RuntimeError("Fallo: docker compose build")
    else:
        print("==> (skip-build) omitiendo construcción de imágenes")

    if (
        _run(
            ["docker", "compose", "--project-name", PROJECT_NAME, "up", "-d"],
            timeout=build_timeout,
        )
        != 0
    ):
        raise RuntimeError("Fallo: docker compose up")

    try:
        health = _wait_for_health(BACKEND_HEALTH_URL, health_timeout)
        print(
            "==> Health check OK: "
            f"status={health.get('status')} "
            f"model_loaded={health.get('model_loaded')} "
            f"face_detector_available={health.get('face_detector_available')}"
        )

        _check_frontend(FRONTEND_URL)
        print("==> Frontend OK (HTTP 200)")
    except TimeoutError:
        _show_logs("backend")
        raise
    except (RuntimeError, OSError):
        _show_logs("frontend")
        raise

    print("==> Smoke test de integración Docker completado con éxito.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test de integración Docker (T-14.5)."
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="No construir las imágenes (asume que ya existen).",
    )
    parser.add_argument(
        "--build-timeout",
        type=int,
        default=1800,
        help="Timeout de build/up en segundos (default: 1800).",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=180,
        help="Timeout del health check en segundos (default: 180).",
    )
    return parser.parse_args(argv)


def _cleanup() -> None:
    """Detener servicios al terminar."""
    _run(["docker", "compose", "--project-name", PROJECT_NAME, "down"])


def main() -> int:
    args = _parse_args()
    try:
        run_smoke_test(args.skip_build, args.build_timeout, args.health_timeout)
        return 0
    except (RuntimeError, TimeoutError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        _cleanup()


if __name__ == "__main__":
    sys.exit(main())
