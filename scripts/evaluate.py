"""Evaluar modelo de atributos faciales en el conjunto de test.

Genera reporte per-attribute con métricas de Precision, Recall, F1, PR-AUC,
ROC-AUC, análisis de errores y optimización de thresholds.

Uso:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --checkpoint checkpoints_40k/best_model.pt
    uv run python scripts/evaluate.py --output evaluation_results
"""

from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from facial_attributes.data.dataset import DatasetManager
from facial_attributes.evaluation.evaluator import Evaluator
from facial_attributes.evaluation.metrics import EvaluationMetrics
from facial_attributes.evaluation.thresholds import ThresholdOptimizer
from facial_attributes.model.classifier import FacialAttributeClassifier, ModelConfig
from facial_attributes.training.config import set_seed
from facial_attributes.training.dataset import FacialAttributeDataset


def _resolve_observable_columns(annotations_file: Path) -> list[str]:
    """Resolver las columnas observables reales del CSV de anotaciones."""
    import pandas as pd

    df = pd.read_csv(annotations_file)
    manager = DatasetManager(annotations_file.parent)
    return [col for col in manager.get_observable_attribute_columns(df)]


def _build_transform(normalize: bool) -> object:
    """Construir transformación de inferencia.

    Coincide con la usada en entrenamiento (Resize + ToTensor).
    Si ``normalize`` es verdadero, se agrega normalización ImageNet.
    """
    from torchvision import transforms

    ops = [transforms.Resize((224, 224)), transforms.ToTensor()]
    if normalize:
        ops.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return transforms.Compose(ops)


def load_test_split(
    annotations_file: Path,
    images_dir: Path,
    train_ratio: float = 0.775,
    val_ratio: float = 0.15,
    seed: int = 42,
    normalize: bool = False,
    attribute_columns: list[str] | None = None,
) -> tuple[torch.utils.data.Dataset, list[str]]:
    """Cargar dataset y extraer el split de test (idéntico al entrenamiento).

    Reproduce la separación de train_subset.py:
    train=77.5%, val=15%, test=7.5% con seed=42.

    Args:
        annotations_file: Ruta al CSV de anotaciones.
        images_dir: Directorio de imágenes.
        train_ratio: Proporción de entrenamiento.
        val_ratio: Proporción de validación.
        seed: Semilla de la división.
        normalize: Aplicar normalización ImageNet.
        attribute_columns: Subconjunto de atributos (todos si None).

    Returns:
        Tupla de (dataset_test, attribute_columns).
    """
    ds = FacialAttributeDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        transform=_build_transform(normalize),
        attribute_columns=attribute_columns,
    )
    attribute_columns = ds.get_attribute_columns()

    n = len(ds)
    val_size = int(val_ratio * n)
    test_size = int((1 - train_ratio - val_ratio) * n)
    train_size = n - val_size - test_size

    _, _, test_ds = torch.utils.data.random_split(
        ds,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    print(
        f"Dataset: {n} imágenes, test split: {len(test_ds)} muestras "
        f"(train={train_size}, val={val_size}, test={test_size})"
    )
    return test_ds, attribute_columns


def load_model(
    checkpoint_path: Path, num_attributes: int = 40
) -> FacialAttributeClassifier:
    """Cargar modelo desde checkpoint."""
    config = ModelConfig(
        num_attributes=num_attributes,
        backbone="resnet18",
        pretrained=False,
        dropout_rate=0.4,
    )
    model = FacialAttributeClassifier(config)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("best_val_loss", "?")
    print(f"Modelo cargado: checkpoint época {epoch}, val_loss={val_loss}")
    return model


def run_inference(
    model: FacialAttributeClassifier,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ejecutar inferencia en el dataset de test.

    Returns:
        Tupla de (predictions_probabilities, targets) como numpy arrays.
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    start = time.time()
    with torch.no_grad():
        for images, attributes in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu()
            all_preds.append(probs)
            all_targets.append(attributes)

    elapsed = time.time() - start
    predictions = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    print(f"Inferencia completada: {len(predictions)} muestras en {elapsed:.1f}s")
    return predictions, targets


def evaluate_and_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    attribute_names: list[str],
    output_dir: Path,
    num_error_samples: int = 50,
) -> None:
    """Evaluar predicciones, optimizar thresholds y guardar reportes."""
    evaluator = Evaluator(attribute_names=attribute_names)

    # Evaluación con threshold fijo (0.5)
    print("\n" + "=" * 70)
    print("EVALUACIÓN CON THRESHOLD FIJO (0.5)")
    print("=" * 70)
    report = evaluator.evaluate(
        predictions, targets, threshold=0.5, num_error_samples=num_error_samples
    )

    _print_metrics(report.metrics)

    # Optimización de thresholds
    print("\n" + "=" * 70)
    print("OPTIMIZACIÓN DE THRESHOLDS POR ATRIBUTO")
    print("=" * 70)
    optimizer = ThresholdOptimizer()
    threshold_results = optimizer.optimize(predictions, targets, attribute_names)

    # Comparar F1 con threshold fijo vs optimizado
    print(
        f"\n{'Atributo':<25} {'F1 (t=0.5)':<12} {'F1 (opt)':<12} "
        f"{'Threshold':<10} {'Mejora'}"
    )
    print("-" * 75)
    improvements = 0
    for i, attr_name in enumerate(attribute_names):
        pred_fixed = (predictions[:, i] > 0.5).astype(int)
        f1_fixed = f1_score(targets[:, i], pred_fixed, zero_division=0)
        f1_opt = threshold_results[i].f1_score
        t_opt = threshold_results[i].threshold
        delta = f1_opt - f1_fixed
        if delta > 0.001:
            improvements += 1
        print(
            f"{attr_name:<25} {f1_fixed:<12.4f} {f1_opt:<12.4f} "
            f"{t_opt:<10.2f} {delta:+.4f}"
        )

    print(
        f"\n{improvements}/{len(attribute_names)} atributos mejoran "
        f"con threshold optimizado"
    )

    # Evaluar con thresholds optimizados
    pred_optimized = optimizer.apply_thresholds(predictions, threshold_results)
    print("\n" + "=" * 70)
    print("EVALUACIÓN CON THRESHOLDS OPTIMIZADOS")
    print("=" * 70)
    report_opt = evaluator.evaluate(
        pred_optimized.astype(float),
        targets,
        threshold=0.5,
        num_error_samples=num_error_samples,
    )
    _print_metrics(report_opt.metrics)

    # Estimar márgenes de incerteza
    print("\n" + "=" * 70)
    print("MÁRGENES DE INCERTEZA")
    print("=" * 70)
    margins = optimizer.estimate_margins(predictions, targets, threshold_results)
    print(f"\n{'Atributo':<25} {'Threshold':<10} {'Margin':<10} {'Rango incerteza'}")
    print("-" * 65)
    for m in margins:
        low = max(0.0, m.threshold - m.margin)
        high = min(1.0, m.threshold + m.margin)
        print(
            f"{m.attribute:<25} {m.threshold:<10.2f} {m.margin:<10.2f} "
            f"[{low:.2f}, {high:.2f}]"
        )

    # Guardar reportes
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator.save_report(report, output_dir)

    # Guardar thresholds optimizados
    thresholds_data = {
        "per_attribute": [
            {"attribute": r.attribute, "threshold": r.threshold, "f1_score": r.f1_score}
            for r in threshold_results
        ]
    }
    with open(output_dir / "thresholds.json", "w") as f:
        json.dump(thresholds_data, f, indent=2)

    # Guardar márgenes
    margins_data = {
        "per_attribute": [
            {
                "attribute": m.attribute,
                "threshold": m.threshold,
                "margin": m.margin,
                "range": [
                    max(0.0, m.threshold - m.margin),
                    min(1.0, m.threshold + m.margin),
                ],
            }
            for m in margins
        ]
    }
    with open(output_dir / "margins.json", "w") as f:
        json.dump(margins_data, f, indent=2)

    # Guardar resumen de peores/mejores atributos
    summary = {
        "worst_attributes": report.metrics.worst_attributes,
        "best_attributes": report.metrics.best_attributes,
        "global_metrics": {
            "accuracy": report.metrics.accuracy,
            "precision": report.metrics.precision,
            "recall": report.metrics.recall,
            "f1": report.metrics.f1,
            "macro_f1": report.metrics.macro_f1,
            "hamming_loss": report.metrics.hamming,
            "average_precision": report.metrics.average_precision,
            "macro_roc_auc": report.metrics.macro_roc_auc,
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nReportes guardados en: {output_dir}/")
    print("  - metrics.json")
    print("  - attribute_summary.csv")
    print("  - error_analysis.json")
    print("  - thresholds.json")
    print("  - margins.json")
    print("  - summary.json")


def _print_metrics(metrics: EvaluationMetrics) -> None:
    """Imprimir métricas globales y Top-5 peores/mejores."""
    print(f"\n  Accuracy:          {metrics.accuracy:.4f}")
    print(f"  Precision (micro): {metrics.precision:.4f}")
    print(f"  Recall (micro):    {metrics.recall:.4f}")
    print(f"  F1 (micro):        {metrics.f1:.4f}")
    print(f"  Macro F1:          {metrics.macro_f1:.4f}")
    print(f"  Hamming Loss:      {metrics.hamming:.4f}")
    print(f"  Avg Precision:     {metrics.average_precision:.4f}")
    print(f"  Macro ROC-AUC:     {metrics.macro_roc_auc:.4f}")

    print("\n  Top-5 MEJORES atributos (por F1):")
    for attr in metrics.best_attributes:
        am = next(a for a in metrics.per_attribute if a.name == attr)
        print(
            f"    {attr:<25} F1={am.f1:.4f}  P={am.precision:.4f}  "
            f"R={am.recall:.4f}  support={am.support}"
        )

    print("\n  Top-5 PEORES atributos (por F1):")
    for attr in metrics.worst_attributes:
        am = next(a for a in metrics.per_attribute if a.name == attr)
        print(
            f"    {attr:<25} F1={am.f1:.4f}  P={am.precision:.4f}  "
            f"R={am.recall:.4f}  support={am.support}"
        )


def main() -> None:
    parser = ArgumentParser(description="Evaluar modelo de atributos faciales")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints_40k/best_model.pt"),
        help="Ruta al checkpoint del modelo",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/processed/celeba_subset_40000.csv"),
        help="Ruta al CSV de anotaciones",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/raw/images"),
        help="Directorio de imágenes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results"),
        help="Directorio de salida para reportes",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-error-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="No aplicar normalización ImageNet (solo para modelos legacy sin normalizar)",
    )
    parser.add_argument(
        "--observable-only",
        action="store_true",
        help="Evaluar solo los 24 atributos visualmente observables",
    )
    parser.add_argument(
        "--attributes",
        default=None,
        help="Subconjunto de atributos (columnas Atr_* separadas por coma)",
    )
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    if args.observable_only and args.attributes:
        parser.error("--observable-only y --attributes son excluyentes")

    set_seed(args.seed)

    print("=" * 70)
    print("EVALUACIÓN DE MODELO - Atributos Faciales")
    print("=" * 70)

    # 1. Resolver subconjunto de atributos
    attribute_subset: list[str] | None = None
    if args.attributes is not None:
        attribute_subset = [c.strip() for c in args.attributes.split(",") if c.strip()]
    elif args.observable_only:
        attribute_subset = _resolve_observable_columns(args.annotations)
        print(
            f"Evaluando solo 24 atributos observables: "
            f"{len(attribute_subset)} columnas"
        )

    # 2. Cargar test split
    test_ds, attribute_columns = load_test_split(
        annotations_file=args.annotations,
        images_dir=args.images_dir,
        seed=args.seed,
        normalize=args.normalize,
        attribute_columns=attribute_subset,
    )

    # 2. Cargar modelo
    model = load_model(args.checkpoint, num_attributes=len(attribute_columns))

    # 3. Inferencia
    predictions, targets = run_inference(model, test_ds, batch_size=args.batch_size)

    # 4. Evaluar y generar reportes
    evaluate_and_report(
        predictions,
        targets,
        attribute_columns,
        output_dir=args.output,
        num_error_samples=args.num_error_samples,
    )


if __name__ == "__main__":
    main()
