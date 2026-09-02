"""Script para versionado de datos con DVC."""

import subprocess
import sys
from pathlib import Path


def run_dvc_command(args: list[str], cwd: Path | None = None) -> int:
    """Ejecutar comando DVC."""
    cmd = ["uv", "run", "dvc"] + args
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    return result.returncode


def init_dvc(project_dir: Path) -> None:
    """Inicializar DVC en el proyecto."""
    print("Inicializando DVC...")
    run_dvc_command(["init"], cwd=project_dir)
    print("DVC inicializado correctamente.")


def add_data(data_path: Path, project_dir: Path) -> None:
    """Agregar datos al seguimiento de DVC."""
    print(f"Agregando {data_path} a DVC...")
    run_dvc_command(["add", str(data_path)], cwd=project_dir)
    print(f"{data_path} agregado correctamente.")


def tag_version(tag_name: str, project_dir: Path) -> None:
    """Crear tag de versión en DVC."""
    print(f"Creando tag {tag_name}...")
    run_dvc_command(["tag", tag_name], cwd=project_dir)
    print(f"Tag {tag_name} creado correctamente.")


def main() -> None:
    """Función principal."""
    project_dir = Path(__file__).parent.parent.parent

    if len(sys.argv) < 2:
        print("Uso: python manage_dvc.py <comando> [args]")
        print("Comandos:")
        print("  init          - Inicializar DVC")
        print("  add <path>    - Agregar datos a DVC")
        print("  tag <name>    - Crear tag de versión")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init":
        init_dvc(project_dir)
    elif command == "add":
        if len(sys.argv) < 3:
            print("Error: Se requiere una ruta para 'add'")
            sys.exit(1)
        data_path = Path(sys.argv[2])
        add_data(data_path, project_dir)
    elif command == "tag":
        if len(sys.argv) < 3:
            print("Error: Se requiere un nombre para 'tag'")
            sys.exit(1)
        tag_version(sys.argv[2], project_dir)
    else:
        print(f"Comando desconocido: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
