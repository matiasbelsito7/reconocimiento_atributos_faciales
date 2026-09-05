"""Entrenar modelo de atributos faciales en un subconjunto para CPU."""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from facial_attributes.model.classifier import FacialAttributeClassifier, ModelConfig
from facial_attributes.model.losses import MultilabelLoss
from facial_attributes.training.checkpoint import CheckpointManager
from facial_attributes.training.config import set_seed
from facial_attributes.training.dataset import (
    CachedAttributeDataset,
    FacialAttributeDataset,
)


class SubsetTrainer:
    """Entrenador ligero para subconjuntos en CPU."""

    def __init__(self, sample_ids: list[int] | None = None) -> None:
        self.sample_ids = sample_ids
        self.device = torch.device("cpu")
        set_seed(42)

    def train(
        self,
        annotations_file: Path,
        images_dir: Path,
        num_epochs: int = 15,
        batch_size: int = 16,
        learning_rate: float = 3e-4,
        num_workers: int = 0,
        checkpoint_dir: str = "checkpoints",
        resume: bool = False,
        cache_dir: Path | None = None,
    ) -> dict[str, list[float]]:
        """Ejecutar entrenamiento.

        Args:
            annotations_file: Ruta al CSV de anotaciones.
            images_dir: Directorio de imágenes.
            num_epochs: Número de épocas.
            batch_size: Tamaño de lote.
            learning_rate: Tasa de aprendizaje.
            num_workers: Trabajadores del DataLoader (0 en Windows).
            checkpoint_dir: Directorio de checkpoints.
            resume: Continuar desde el último checkpoint si existe.
            cache_dir: Directorio de cache de .npy (acelera entrenamiento).

        Returns:
            Historial de entrenamiento.
        """
        if cache_dir is not None and (cache_dir / "npy").exists():
            ds: FacialAttributeDataset | CachedAttributeDataset = (
                CachedAttributeDataset(
                    annotations_file=annotations_file,
                    cache_dir=cache_dir,
                )
            )
            print(
                f"Usando cache con {len(ds)} instancias desde {cache_dir}", flush=True
            )
        else:
            ds = FacialAttributeDataset(
                annotations_file=annotations_file,
                images_dir=images_dir,
                transform=self._build_transform(),
            )
            print(f"Imágenes cargadas desde {images_dir}", flush=True)

        n = len(ds)
        val_size = int(0.15 * n)
        test_size = int(0.075 * n)
        train_size = n - val_size - test_size

        train_ds, val_ds, test_ds = random_split(
            ds,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        num_attributes = ds.get_num_attributes()
        print(
            f"Atributos: {num_attributes}, Train: {train_size}, Val: {val_size}, Test: {test_size}",
            flush=True,
        )

        model_config = ModelConfig(
            num_attributes=num_attributes,
            backbone="resnet18",
            pretrained=True,
            dropout_rate=0.4,
        )
        model = FacialAttributeClassifier(model_config).to(self.device)

        loss_fn = MultilabelLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        cm = CheckpointManager(checkpoint_dir)
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        start_epoch = 0

        if resume:
            latest = cm.get_latest_checkpoint()
            if latest is not None:
                info = cm.load_checkpoint(model, optimizer, latest)
                start_epoch = info["epoch"] + 1
                best_val_loss = info["best_val_loss"]
                print(
                    f"Reanudando desde checkpoint {latest.name}, "
                    f"época {info['epoch']}, val_loss {info['best_val_loss']:.4f}",
                    flush=True,
                )

        for epoch in range(start_epoch, num_epochs):
            start = time.time()
            model.train()
            total_loss, n_batch = 0.0, 0
            for images, attributes in train_loader:
                images, attributes = images.to(self.device), attributes.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, attributes)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batch += 1

            model.eval()
            val_loss, val_preds, val_targets = 0.0, [], []
            with torch.no_grad():
                for images, attributes in val_loader:
                    images, attributes = images.to(self.device), attributes.to(
                        self.device
                    )
                    outputs = model(images)
                    val_loss += loss_fn(outputs, attributes).item()
                    val_preds.append(torch.sigmoid(outputs).cpu())
                    val_targets.append(attributes.cpu())

            results = self._evaluate(val_preds, val_targets)

            train_loss = total_loss / max(n_batch, 1)
            val_loss_avg = val_loss / max(len(val_loader), 1)

            is_best = val_loss_avg < best_val_loss
            if is_best:
                best_val_loss = val_loss_avg

            cm.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss_avg,
                config=asdict(model_config),
                is_best=is_best,
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss_avg)
            history["val_f1"].append(results["f1"])

            elapsed = time.time() - start
            print(
                f"epoch {epoch+1}/{num_epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss_avg:.4f} "
                f"val_acc={results['accuracy']:.3f} val_f1={results['f1']:.3f} "
                f"({elapsed:.1f}s) {'[BEST]' if is_best else ''}",
                flush=True,
            )

        print("\nEntrenamiento completado.", flush=True)
        print(
            f"Mejor modelo guardado en: {Path(checkpoint_dir) / 'best_model.pt'}",
            flush=True,
        )
        return history

    def _build_transform(self):
        """Construir transformación coherente con la inferencia."""
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def _evaluate(
        self, preds: list[torch.Tensor], targets: list[torch.Tensor]
    ) -> dict[str, float]:
        """Calcular métricas en validación."""
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        binary = (preds > 0.5).float()

        correct = (binary == targets).sum().item()
        total = targets.numel()
        accuracy = correct / max(total, 1)

        tp = ((binary == 1) & (targets == 1)).sum().item()
        fp = ((binary == 1) & (targets == 0)).sum().item()
        fn = ((binary == 0) & (targets == 1)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


if __name__ == "__main__":
    trainer = SubsetTrainer()
    trainer.train(
        annotations_file=Path("data/processed/celeba_subset_40000.csv"),
        images_dir=Path("data/raw/images"),
        num_epochs=10,
        batch_size=32,
        learning_rate=3e-4,
        num_workers=0,
        checkpoint_dir="checkpoints_40k",
        resume=False,
        cache_dir=Path(
            os.environ.get("CELEBA_CACHE_DIR", "data/processed/cache_40000")
        ),
    )
