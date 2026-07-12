from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


@dataclass
class ColorRange:
    name: ClassVar[str]
    bits: int = 8

    @property
    def min(self) -> NDArray[np.float32]:
        raise NotImplementedError('min not implemented for this color range.')

    @property
    def max(self) -> NDArray[np.float32]:
        raise NotImplementedError('max not implemented for this color range.')

    @property
    def scale(self) -> NDArray[np.float32]:
        return self.max - self.min

    @property
    def offset(self) -> NDArray[np.float32]:
        return np.array(
            [
                self.min[0],
                (self.min[1] + self.max[1]) * 0.5,
                (self.min[2] + self.max[2]) * 0.5,
            ],
            np.float32,
        )


@dataclass
class FullRange(ColorRange):
    name: ClassVar[str] = 'Full Range'

    @property
    def min(self) -> NDArray[np.float32]:
        return np.zeros(3, np.float32)

    @property
    def max(self) -> NDArray[np.float32]:
        return np.ones(3, np.float32)


@dataclass
class LimitedRange(ColorRange):
    name: ClassVar[str] = 'Limited Range'

    @property
    def min(self) -> NDArray[np.float32]:
        return np.array(
            [
                16 << (self.bits - 8),
                16 << (self.bits - 8),
                16 << (self.bits - 8),
            ],
            np.float32,
        ) / np.float32(2**self.bits - 1)

    @property
    def max(self) -> NDArray[np.float32]:
        return np.array(
            [
                235 << (self.bits - 8),
                240 << (self.bits - 8),
                240 << (self.bits - 8),
            ],
            np.float32,
        ) / np.float32(2**self.bits - 1)
