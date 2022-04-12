from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


def mkdir_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
