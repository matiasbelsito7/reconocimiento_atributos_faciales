"""Logger de predicciones para trazabilidad."""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PredictionRecord:
    """Registro de predicción."""

    prediction_id: str
    timestamp: float
    image_id: str
    num_faces: int
    faces: list[dict[str, object]]
    latency_ms: float
    model_version: str
    metadata: dict[str, str] = field(default_factory=dict)


class PredictionLogger:
    """Logger de predicciones para trazabilidad."""

    def __init__(
        self,
        log_dir: str | Path = "logs/predictions",
        max_file_size_mb: int = 100,
        model_version: str = "1.0.0",
    ) -> None:
        """Inicializar logger de predicciones.

        Args:
            log_dir: Directorio de logs de predicciones.
            max_file_size_mb: Tamaño máximo de archivo en MB.
            model_version: Versión del modelo.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_mb = max_file_size_mb
        self.model_version = model_version
        self._current_file: Path | None = None
        self._current_size: int = 0

    def log_prediction(
        self,
        image_id: str,
        faces: list[dict[str, object]],
        latency_ms: float,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Registrar una predicción.

        Args:
            image_id: ID de la imagen procesada.
            faces: Lista de predicciones por rostro.
            latency_ms: Latencia de la predicción.
            metadata: Metadata adicional.

        Returns:
            ID de la predicción registrada.
        """
        prediction_id = str(uuid.uuid4())

        record = PredictionRecord(
            prediction_id=prediction_id,
            timestamp=time.time(),
            image_id=image_id,
            num_faces=len(faces),
            faces=faces,
            latency_ms=latency_ms,
            model_version=self.model_version,
            metadata=metadata or {},
        )

        self._write_record(record)
        return prediction_id

    def _write_record(self, record: PredictionRecord) -> None:
        """Escribir registro a archivo.

        Args:
            record: Registro a escribir.
        """
        if (
            self._current_file is None
            or self._current_size >= self.max_file_size_mb * 1024 * 1024
        ):
            self._rotate_file()

        line = json.dumps(asdict(record)) + "\n"

        with open(self._current_file, "a") as f:
            f.write(line)

        self._current_size += len(line.encode("utf-8"))

    def _rotate_file(self) -> None:
        """Rotar archivo de log."""
        timestamp = int(time.time())
        self._current_file = self.log_dir / f"predictions_{timestamp}.jsonl"
        self._current_size = 0

    def get_predictions(
        self,
        limit: int | None = None,
        model_version: str | None = None,
    ) -> list[PredictionRecord]:
        """Obtener predicciones registradas.

        Args:
            limit: Límite de predicciones a retornar.
            model_version: Filtrar por versión de modelo.

        Returns:
            Lista de registros de predicción.
        """
        records: list[PredictionRecord] = []

        log_files = sorted(self.log_dir.glob("predictions_*.jsonl"), reverse=True)

        for log_file in log_files:
            with open(log_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        record = PredictionRecord(**data)

                        if model_version and record.model_version != model_version:
                            continue

                        records.append(record)

                        if limit and len(records) >= limit:
                            return records

        return records

    def get_prediction_count(self, model_version: str | None = None) -> int:
        """Obtener conteo de predicciones.

        Args:
            model_version: Filtrar por versión de modelo.

        Returns:
            Número total de predicciones.
        """
        count = 0
        log_files = self.log_dir.glob("predictions_*.jsonl")

        for log_file in log_files:
            with open(log_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if model_version and data.get("model_version") != model_version:
                            continue
                        count += 1

        return count

    def clear_logs(self) -> int:
        """Limpiar logs de predicciones.

        Returns:
            Número de archivos eliminados.
        """
        count = 0
        for log_file in self.log_dir.glob("predictions_*.jsonl"):
            log_file.unlink()
            count += 1
        return count
