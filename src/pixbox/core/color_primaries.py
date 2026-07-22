# https://www.itu.int/rec/R-REC-BT/en
# https://registry.color.org/rgb-registry/

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from pixbox.core.color_range import ColorRange
from pixbox.core.color_transfer import ColorTransfer


@dataclass
class ColorPrimaries:
    name: ClassVar[str]
    red: ClassVar[NDArray[np.float32]]
    green: ClassVar[NDArray[np.float32]]
    blue: ClassVar[NDArray[np.float32]]
    white: ClassVar[NDArray[np.float32]]

    _matrix_rgb2xyz: NDArray[np.float32] | None = None
    _matrix_xyz2rgb: NDArray[np.float32] | None = None
    _matrix_rgb2yuv: NDArray[np.float32] | None = None
    _matrix_yuv2rgb: NDArray[np.float32] | None = None

    @property
    def matrix_rgb2xyz(self) -> NDArray[np.float32]:
        if self._matrix_rgb2xyz is None:
            primaries = np.array(
                [self.red, self.green, self.blue], np.float32
            ).T
            white_xyz = self.white / self.white[1]
            scale = np.linalg.solve(primaries, white_xyz)
            self._matrix_rgb2xyz = primaries @ np.diag(scale)
        return self._matrix_rgb2xyz

    @property
    def matrix_xyz2rgb(self) -> NDArray[np.float32]:
        if self._matrix_xyz2rgb is None:
            self._matrix_xyz2rgb = np.linalg.inv(self.matrix_rgb2xyz)
        return self._matrix_xyz2rgb

    @property
    def matrix_rgb2yuv(self) -> NDArray[np.float32]:
        if self._matrix_rgb2yuv is None:
            kr, kg, kb = self.matrix_rgb2xyz[1, :]
            self._matrix_rgb2yuv = np.array(
                [
                    [kr, kg, kb],
                    [-0.5 * kr / (1.0 - kb), -0.5 * kg / (1.0 - kb), 0.5],
                    [0.5, -0.5 * kg / (1.0 - kr), -0.5 * kb / (1.0 - kr)],
                ],
                np.float32,
            )
        return self._matrix_rgb2yuv

    @property
    def matrix_yuv2rgb(self) -> NDArray[np.float32]:
        if self._matrix_yuv2rgb is None:
            self._matrix_yuv2rgb = np.linalg.inv(self.matrix_rgb2yuv)
        return self._matrix_yuv2rgb

    def rgb2xyz(
        self,
        rgb: NDArray[np.float32],
        color_transfer: ColorTransfer | None = None,
    ) -> NDArray[np.float32]:
        if color_transfer is not None:
            rgb = color_transfer.rgb2lin(rgb)
        xyz = self.matrix_rgb2xyz @ rgb
        return xyz

    def xyz2rgb(
        self,
        xyz: NDArray[np.float32],
        color_transfer: ColorTransfer | None = None,
    ) -> NDArray[np.float32]:
        rgb = self.matrix_xyz2rgb @ xyz
        if color_transfer is not None:
            rgb = color_transfer.lin2rgb(rgb)
        return rgb

    def rgb2yuv(
        self, rgb: NDArray[np.float32], color_range: ColorRange | None = None
    ) -> NDArray[np.float32]:
        yuv = self.matrix_rgb2yuv @ rgb
        if color_range is not None:
            yuv = yuv * color_range.scale + color_range.offset
        return yuv

    def yuv2rgb(
        self, yuv: NDArray[np.float32], color_range: ColorRange | None = None
    ) -> NDArray[np.float32]:
        if color_range is not None:
            yuv = (yuv - color_range.offset) / color_range.scale
        rgb = self.matrix_yuv2rgb @ yuv
        return rgb


@dataclass
class SrgbPrimaries(ColorPrimaries):
    name: ClassVar[str] = 'sRGB'
    red: ClassVar[NDArray[np.float32]] = np.array(
        [0.64, 0.33, 0.03], np.float32
    )
    green: ClassVar[NDArray[np.float32]] = np.array(
        [0.30, 0.60, 0.10], np.float32
    )
    blue: ClassVar[NDArray[np.float32]] = np.array(
        [0.15, 0.06, 0.79], np.float32
    )
    white: ClassVar[NDArray[np.float32]] = np.array(
        [0.3127, 0.3290, 0.3583], np.float32
    )  # D65


@dataclass
class BT709Primaries(ColorPrimaries):
    name: ClassVar[str] = 'BT.709'
    red: ClassVar[NDArray[np.float32]] = np.array(
        [0.64, 0.33, 0.03], np.float32
    )
    green: ClassVar[NDArray[np.float32]] = np.array(
        [0.30, 0.60, 0.10], np.float32
    )
    blue: ClassVar[NDArray[np.float32]] = np.array(
        [0.15, 0.06, 0.79], np.float32
    )
    white: ClassVar[NDArray[np.float32]] = np.array(
        [0.3127, 0.3290, 0.3583], np.float32
    )  # D65


@dataclass
class BT2020Primaries(ColorPrimaries):
    name: ClassVar[str] = 'BT.2020'
    red: ClassVar[NDArray[np.float32]] = np.array(
        [0.708, 0.292, 0.000], np.float32
    )
    green: ClassVar[NDArray[np.float32]] = np.array(
        [0.170, 0.797, 0.033], np.float32
    )
    blue: ClassVar[NDArray[np.float32]] = np.array(
        [0.131, 0.046, 0.823], np.float32
    )
    white: ClassVar[NDArray[np.float32]] = np.array(
        [0.3127, 0.3290, 0.3583], np.float32
    )  # D65
