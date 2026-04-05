from dataclasses import dataclass
from typing import Sequence


@dataclass
class OCRResult:
    """Single detected text region."""
    text: str
    score: float
    box: BBox


@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int

    @staticmethod
    def surrounding(others: Sequence[BBox]) -> BBox:
        x1 = min(b.x for b in others)
        y1 = min(b.y for b in others)
        x2 = max(b.x2 for b in others)
        y2 = max(b.y2 for b in others)
        return BBox(x1, y1, x2 - x1, y2 - y1)

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def contains(self, other: BBox) -> bool:
        return (
            self.x <= other.x and self.x2 >= other.x2 and
            self.y <= other.y and self.y2 >= other.y2
        )

    def overlaps_with(self, other: BBox) -> bool:
        return not (
            self.x > other.x2 or self.x2 < other.x or
            self.y > other.y2 or self.y2 < other.y
        )

    def expand(self, padding: int) -> "BBox":
        return BBox(
            max(0, self.x - padding),
            max(0, self.y - padding),
            self.width + 2 * padding,
            self.height + 2 * padding,
        )


