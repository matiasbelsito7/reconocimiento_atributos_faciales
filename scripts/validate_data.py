"""Script de validación de datasets."""

import argparse
from pathlib import Path

from facial_attributes.data.validation import DataValidator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validar dataset de atributos faciales"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/raw/images"),
        help="Directorio de imágenes",
    )
    parser.add_argument(
        "--annotations-file",
        type=Path,
        default=Path("data/raw/annotations/celeba_attributes.csv"),
        help="Archivo de anotaciones CSV",
    )
    args = parser.parse_args()

    validator = DataValidator(args.images_dir, args.annotations_file)
    report = validator.validate()

    print("=" * 60)
    print("REPORTE DE VALIDACIÓN DE DATASET")
    print("=" * 60)

    print(f"\nEstado: {report['status']}")

    if report["stats"]:
        print("\nEstadísticas:")
        print(f"  - Total de imágenes: {report['stats']['total_images']}")
        print(f"  - Total de atributos: {report['stats']['total_attributes']}")

    if report["errors"]:
        print(f"\nErrores ({len(report['errors'])}):")
        for error in report["errors"]:
            print(f"  [ERROR] {error}")

    if report["warnings"]:
        print(f"\nAdvertencias ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"  [WARN] {warning}")

    if not report["errors"] and not report["warnings"]:
        print("\n[OK] Validacion completada sin errores")

    print("=" * 60)


if __name__ == "__main__":
    main()
