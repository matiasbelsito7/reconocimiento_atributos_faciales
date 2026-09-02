"""Script para descargar el dataset CelebA."""

import argparse
import subprocess
import sys
from pathlib import Path


def download_from_kaggle(output_dir: Path) -> None:
    """Descargar CelebA desde Kaggle."""
    try:
        import kaggle
    except ImportError:
        print("Instalando kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        import kaggle

    print("Descargando CelebA desde Kaggle...")
    kaggle.api.dataset_download_files(
        dataset="jessicali9530/celeba-dataset",
        path=str(output_dir),
        unzip=True,
    )
    print("Descarga completada.")


def download_from_url(output_dir: Path) -> None:
    """Descargar CelebA desde URL directa."""
    import urllib.request

    # URLs de CelebA (pueden cambiar)
    urls = {
        "list_attr_celeba.txt": "https://drive.google.com/uc?id=0B7EVK8CluIwkTXRfTEN2dGJUNFE&export=download",
        "identity_CelebA.txt": "https://drive.google.com/uc?id=0B7EVK8CluIwkdTVJZVBqMkxEdFZ4&export=download",
    }

    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in urls.items():
        output_file = annotations_dir / filename
        if output_file.exists():
            print(f"{filename} ya existe, saltando...")
            continue

        print(f"Descargando {filename}...")
        try:
            urllib.request.urlretrieve(url, output_file)
            print(f"Descargado: {filename}")
        except Exception as e:
            print(f"Error descargando {filename}: {e}")
            print(f"Por favor, descarga manualmente desde: {url}")


def create_sample_dataset(output_dir: Path, num_samples: int = 100) -> None:
    """Crear dataset de ejemplo para testing."""
    from PIL import Image

    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creando dataset de ejemplo con {num_samples} imágenes...")

    # Crear imágenes de ejemplo
    for i in range(num_samples):
        img = Image.new(
            "RGB", (178, 218), color=(i * 2 % 256, i * 3 % 256, i * 5 % 256)
        )
        img.save(images_dir / f"{i:06d}.jpg")

    # Crear anotaciones de ejemplo
    import random

    random.seed(42)

    attributes = [
        "Atr_5_o_clock_shadow",
        "Atr_arched_eyebrows",
        "Atr_bags_under_eyes",
        "Atr_bald",
        "Atr_bangs",
        "Atr_big_lips",
        "Atr_big_nose",
        "Atr_black_hair",
        "Atr_blond_hair",
        "Atr_blurry",
        "Atr_brown_hair",
        "Atr_bushy_eyebrows",
        "Atr_chubby",
        "Atr_double_chin",
        "Atr_eyeglasses",
        "Atr_goatee",
        "Atr_gray_hair",
        "Atr_heavy_makeup",
        "Atr_high_cheekbones",
        "Atr_male",
        "Atr_mouth_slightly_open",
        "Atr_mustache",
        "Atr_narrow_eyes",
        "Atr_no_beard",
        "Atr_oval_face",
        "Atr_pale_skin",
        "Atr_pointy_nose",
        "Atr_receding_hairline",
        "Atr_rosy_cheeks",
        "Atr_sideburns",
        "Atr_smiling",
        "Atr_straight_hair",
        "Atr_wavy_hair",
        "Atr_wearing_earrings",
        "Atr_wearing_hat",
        "Atr_wearing_lipstick",
        "Atr_wearing_necklace",
        "Atr_wearing_necktie",
    ]

    with open(annotations_dir / "celeba_attributes.csv", "w") as f:
        f.write("image_id," + ",".join(attributes) + "\n")
        for i in range(num_samples):
            row = [str(i).zfill(6)]
            for _ in attributes:
                row.append(str(random.randint(0, 1)))
            f.write(",".join(row) + "\n")

    print(f"Dataset de ejemplo creado en: {output_dir}")
    print(f"  - Imágenes: {images_dir}")
    print(f"  - Anotaciones: {annotations_dir / 'celeba_attributes.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Descargar dataset CelebA")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directorio de salida",
    )
    parser.add_argument(
        "--method",
        choices=["kaggle", "url", "sample"],
        default="sample",
        help="Método de descarga (default: sample)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Número de muestras para dataset de ejemplo",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "kaggle":
        download_from_kaggle(args.output_dir)
    elif args.method == "url":
        download_from_url(args.output_dir)
    else:
        create_sample_dataset(args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()
