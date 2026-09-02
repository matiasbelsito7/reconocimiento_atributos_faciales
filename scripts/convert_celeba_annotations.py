"""Convert CelebA list_attr_celeba.txt to CSV format."""

from pathlib import Path

import pandas as pd


def convert_celeba_attributes(
    input_file: Path,
    output_file: Path,
) -> None:
    """Convert CelebA attribute file to CSV."""
    with open(input_file) as f:
        lines = f.readlines()

    n_images = int(lines[0].strip())
    attribute_names = lines[1].strip().split()

    print(f"Total images: {n_images}")
    print(f"Attributes: {len(attribute_names)}")

    data = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) == len(attribute_names) + 1:
            image_id = parts[0].replace(".jpg", "")
            values = [1 if v == "1" else 0 for v in parts[1:]]
            data.append([image_id] + values)

    columns = ["image_id"] + ["Atr_" + attr for attr in attribute_names]
    df = pd.DataFrame(data, columns=columns)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"CSV saved: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    input_file = Path(
        r"C:\Users\Manaus\AppData\Local\Temp\kaggle\extracted\celeba\celeba\list_attr_celeba.txt"
    )
    output_file = Path(
        r"C:\Users\Manaus\PROYECTOS\reconocimiento_atributos_faciales\data\raw\annotations\celeba_attributes.csv"
    )
    convert_celeba_attributes(input_file, output_file)
