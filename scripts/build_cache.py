"""Preprocesar y cachear imagenes resized para acelerar el entrenamiento.

Cada worker escribe un .npy por imagen a disco (evita transferir tensores
por la cola de multiprocessing, que falla en Windows por limites de
memoria compartida). Luego el proceso principal apila los .npy en un
tensor unico.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


def _resize_one(args: tuple[int, str, str, str]) -> None:
    """Redimensionar una imagen y guardarla como .npy.

    Args:
        args: (indice, id_raw, directorio_imagenes, directorio_salida)
    """
    idx, raw_id, img_dir, out_dir = args
    img = Image.open(f"{img_dir}/{raw_id}.jpg").convert("RGB")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    np.save(f"{out_dir}/{idx:06d}.npy", arr)


def _collect(annotations_file: Path, images_dir: Path, output_dir: Path) -> None:
    """Escribir .npy por imagen usando workers."""
    df = pd.read_csv(annotations_file)
    raw_ids = df["image_id"].astype(str).str.zfill(6).tolist()
    n = len(raw_ids)

    out_dir = output_dir / "npy"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.npy")))
    print(f"Ya existen {existing}/{n} .npy", flush=True)

    tasks = [(i, rid, str(images_dir), str(out_dir)) for i, rid in enumerate(raw_ids)]

    print(f"Cacheando {n} imagenes con 8 workers en {out_dir}...", flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(_resize_one, tasks)):
            if i % 2000 == 0 and i > 0:
                print(f"  {i}/{n}...", flush=True)

    print("Todas las imagenes cacheadas como .npy", flush=True)


def build_cache(
    annotations_file: Path,
    images_dir: Path,
    output_dir: Path,
) -> None:
    """Construir cache de imagenes resized.

    Args:
        annotations_file: CSV del subconjunto.
        images_dir: Directorio de imagenes.
        output_dir: Directorio de salida.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_dir = output_dir / "npy"
    existing = len(list(npy_dir.glob("*.npy"))) if npy_dir.exists() else 0
    df = pd.read_csv(annotations_file)

    if existing == len(df) and (output_dir / "attributes.pt").exists():
        print(f"Cache ya existe y es valido: {existing} instancias", flush=True)
        return

    _collect(annotations_file, images_dir, output_dir)

    # Guardar atributos como tensor
    df = pd.read_csv(annotations_file)
    attr_cols = [c for c in df.columns if c.startswith("Atr_")]
    attr_tensor = torch.tensor(
        df[attr_cols].values.astype("float32"), dtype=torch.float32
    )
    torch.save(attr_tensor, output_dir / "attributes.pt")
    print(
        f"Atributos guardados en {output_dir / 'attributes.pt'}: {attr_tensor.shape}",
        flush=True,
    )


if __name__ == "__main__":
    import os

    annotations = Path("data/processed/celeba_subset_40000.csv")
    images = Path("data/raw/images")
    default_out = Path(os.environ.get("CELEBA_CACHE_DIR", "data/processed/cache_40000"))
    build_cache(
        annotations_file=annotations,
        images_dir=images,
        output_dir=default_out,
    )
