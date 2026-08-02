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
        with np.errstate(all='ignore'):
            linear = value / 12.92
            nonlinear = (np.maximum(value + 0.055, 0.0) / 1.055) ** 2.4
        return np.where(value <= 0.04045, linear, nonlinear)

    def oetf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        with np.errstate(all='ignore'):
            linear = 12.92 * value
            nonlinear = 1.055 * ((np.maximum(value, 0.0)) ** (1 / 2.4)) - 0.055
        return np.where(value <= 0.0031308, linear, nonlinear)


@dataclass
class BT709Transfer(ColorTransfer):
    name: ClassVar[str] = 'BT.709'

    def eotf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        with np.errstate(all='ignore'):
            linear = value / 4.5
            nonlinear = (np.maximum(value + 0.099, 0.0) / 1.099) ** (1 / 0.45)
        return np.where(value < 0.081, linear, nonlinear)

    def oetf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        with np.errstate(all='ignore'):
            linear = 4.5 * value
            nonlinear = 1.099 * (np.maximum(value, 0.0) ** 0.45) - 0.099
        return np.where(value < 0.018, linear, nonlinear)


@dataclass
class BT2020Transfer(ColorTransfer):
    name: ClassVar[str] = 'BT.2020'

    def eotf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        with np.errstate(all='ignore'):
            linear = value / 4.5
            nonlinear = (np.maximum(value + 0.0993, 0.0) / 1.0993) ** (1 / 0.45)
        return np.where(value < 0.08145, linear, nonlinear)

    def oetf(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        with np.errstate(all='ignore'):
            linear = 4.5 * value
            nonlinear = 1.0993 * (np.maximum(value, 0.0) ** 0.45) - 0.0993
        return np.where(value < 0.0181, linear, nonlinear)
