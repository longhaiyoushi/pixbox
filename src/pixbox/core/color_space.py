from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from pixbox.core.color_primaries import (
    BT709Primaries,
    BT2020Primaries,
    ColorPrimaries,
    SrgbPrimaries,
)
from pixbox.core.color_range import ColorRange, FullRange
from pixbox.core.color_transfer import (
    BT709Transfer,
    BT2020Transfer,
    ColorTransfer,
    SrgbTransfer,
)
from pixbox.core.pixel_format import PixelFormat, XYZFormat, YUVFormat


@dataclass
class ColorSpace:
    name: ClassVar[str]

    pixel_format: PixelFormat
    color_range: ColorRange = field(
        default_factory=lambda: FullRange(),
    )
    color_transfer: ColorTransfer = field(
        default_factory=lambda: SrgbTransfer(),
    )
    color_primaries: ColorPrimaries = field(
        default_factory=lambda: SrgbPrimaries(),
    )

    def rgb2xyz(self, rgb: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.color_primaries.rgb2xyz(rgb, self.color_transfer)

    def xyz2rgb(self, xyz: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.color_primaries.xyz2rgb(xyz, self.color_transfer)

    def rgb2yuv(self, rgb: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.color_primaries.rgb2yuv(rgb, self.color_range)

    def yuv2rgb(self, yuv: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.color_primaries.yuv2rgb(yuv, self.color_range)

    def data2value(self, value: NDArray[np.uint8]) -> NDArray[Any]:
        return self.pixel_format.data2value(value)

    def value2key(self, value: NDArray[Any]) -> NDArray[np.float32]:
        return self.pixel_format.value2key(value)

    def data2key(self, value: NDArray[np.uint8]) -> NDArray[np.float32]:
        return self.pixel_format.data2key(value)

    def key2value(self, value: NDArray[np.float32]) -> NDArray[Any]:
        return self.pixel_format.key2value(value)

    def value2data(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        return self.pixel_format.value2data(value)

    def key2data(self, value: NDArray[np.float32]) -> NDArray[np.uint8]:
        return self.pixel_format.key2data(value)

    def key2rgb(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        if isinstance(self.pixel_format, YUVFormat):
            value = self.yuv2rgb(value)
        elif isinstance(self.pixel_format, XYZFormat):
            value = self.xyz2rgb(value)
        return value


@dataclass
class SrgbColorSpace(ColorSpace):
    name: ClassVar[str] = 'sRGB'

    pixel_format: PixelFormat
    color_range: ColorRange = field(
        default_factory=lambda: FullRange(),
    )
    color_transfer: ColorTransfer = field(
        default_factory=lambda: SrgbTransfer(),
    )
    color_primaries: ColorPrimaries = field(
        default_factory=lambda: SrgbPrimaries(),
    )


@dataclass
class BT709ColorSpace(ColorSpace):
    name: ClassVar[str] = 'BT.709'

    pixel_format: PixelFormat
    color_range: ColorRange = field(
        default_factory=lambda: FullRange(),
    )
    color_transfer: ColorTransfer = field(
        default_factory=lambda: BT709Transfer(),
    )
    color_primaries: ColorPrimaries = field(
        default_factory=lambda: BT709Primaries(),
    )


@dataclass
class BT2020ColorSpace(ColorSpace):
    name: ClassVar[str] = 'BT.2020'

    pixel_format: PixelFormat
    color_range: ColorRange = field(
        default_factory=lambda: FullRange(),
    )
    color_transfer: ColorTransfer = field(
        default_factory=lambda: BT2020Transfer(),
    )
    color_primaries: ColorPrimaries = field(
        default_factory=lambda: BT2020Primaries(),
    )
