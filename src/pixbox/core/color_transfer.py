from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


@dataclass
class ColorTransfer:
    name: ClassVar[str]

    def eotf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        raise NotImplementedError(
            'EOTF not implemented for this color transfer function.'
        )

    def oetf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        raise NotImplementedError(
            'OETF not implemented for this color transfer function.'
        )

    def rgb2lin(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.eotf(value)

    def lin2rgb(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.oetf(value)


@dataclass
class LinearTransfer(ColorTransfer):
    name: ClassVar[str] = 'Linear'

    def eotf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        return value

    def oetf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        return value


@dataclass
class SrgbTransfer(ColorTransfer):
    name: ClassVar[str] = 'sRGB'

    def eotf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.where(
            value <= 0.04045,
            value / 12.92,
            ((value + 0.055) / 1.055) ** 2.4,
        )

    def oetf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.where(
            value <= 0.0031308,
            12.92 * value,
            1.055 * (value ** (1 / 2.4)) - 0.055,
        )


@dataclass
class BT709Transfer(ColorTransfer):
    name: ClassVar[str] = 'BT.709'

    def eotf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.where(
            value < 0.081,
            value / 4.5,
            ((value + 0.099) / 1.099) ** (1 / 0.45),
        )

    def oetf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.where(
            value < 0.018,
            4.5 * value,
            1.099 * (value**0.45) - 0.099,
        )


@dataclass
class BT2020Transfer(ColorTransfer):
    name: ClassVar[str] = 'BT.2020'

    def eotf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.where(
            value < 0.08145,
            value / 4.5,
            ((value + 0.0993) / 1.0993) ** (1 / 0.45),
        )

    def oetf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.where(
            value < 0.0181,
            4.5 * value,
            1.0993 * (value**0.45) - 0.0993,
        )
