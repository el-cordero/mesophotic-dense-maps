"""Reading tile images and tile point files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class PointRecord:
    x: int
    y: int
    class_id: int


def iter_images(images_root: Path, split: str | None = None) -> list[Path]:
    roots = [images_root / split] if split else [p for p in images_root.iterdir() if p.is_dir()]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(
            sorted(
                p
                for p in root.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
                and not p.name.startswith(".")
                and not p.name.startswith("._")
            )
        )
    return files


def read_points_txt(path: Path) -> list[PointRecord]:
    """Read a tile point file with one 'x y class_id' record per line (1-based pixel coordinates)."""
    if not path.exists():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    records: list[PointRecord] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_no}: expected 3 fields, got {len(parts)}")
        x, y, class_id = (int(float(v)) for v in parts)
        records.append(PointRecord(x=x, y=y, class_id=class_id))
    return records
