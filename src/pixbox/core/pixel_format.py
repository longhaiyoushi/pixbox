import functools
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray


@dataclass
class PixelFormat:
    name: ClassVar[str]
    height: int
    width: int
    stride: int = 0

    def __post_init__(self) -> None:
        self.stride = max(self.stride, self.pitch)

    @property
    def pitch(self) -> int:
        raise NotImplementedError(
            'pitch not implemented for this pixel format.'
        )

    @functools.cached_property
    def padding(self) -> int:
        return self.stride - self.pitch

    @property
    def bits_per_pixel(self) -> int:  # bpp
        raise NotImplementedError(
            'bits_per_pixel not implemented for this pixel format.'
        )

    @property
    def bytes_per_frame(self) -> int:
        raise NotImplementedError(
            'bytes_per_frame not implemented for this pixel format.'
        )

    def data2key(self, value8: NDArray[np.uint8]) -> NDArray[np.float32]:
        raise NotImplementedError(
            'data2key not implemented for this pixel format.'
        )

    def key2data(self, value: NDArray[np.float32]) -> NDArray[np.uint8]:
        raise NotImplementedError(
            'key2data not implemented for this pixel format.'
        )


class ChromaOrder(Enum):
    UV = 'uv'
    VU = 'vu'


class PackedOrder(Enum):
    YUV = 'yuv'
    UVY = 'uvy'


@dataclass
class YUVFormat(PixelFormat):
    is_integer: ClassVar[bool]
    chroma_order: ClassVar[ChromaOrder]
    bits: int = 8
    expanded: bool = True

    @functools.cached_property
    def pitch(self) -> int:
        return int(self.width * self.item_size)

    @functools.cached_property
    def item_size(self) -> float:
        size = self.bits / 8
        if self.expanded:
            size = math.ceil(size)
        return size

    @functools.cached_property
    def dtype(self) -> type[np.uint8 | np.uint16 | np.float16 | np.float32]:
        if self.is_integer:
            if self.bits <= 8:
                return np.uint8
            elif self.bits <= 16:
                return np.uint16
        else:
            if self.bits == 16:
                return np.float16
            elif self.bits == 32:
                return np.float32
        raise ValueError(f'Unsupported bit depth: {self.bits}.')

    @functools.cached_property
    def fill_value(self) -> float:
        if self.is_integer:
            return 1 << (self.bits - 1)
        else:
            return 128.0 / 255.0

    def unpack10(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        value8 = value8.view(np.uint8).reshape((-1, 5))
        value = np.empty((value8.shape[0], 4), self.dtype)
        value[:, 0] = ((value8[:, 0] & 0xFF) >> 0) | (
            (value8[:, 1] & 0x03) << 8
        )
        value[:, 1] = ((value8[:, 1] & 0xFC) >> 2) | (
            (value8[:, 2] & 0x0F) << 6
        )
        value[:, 2] = ((value8[:, 2] & 0xF0) >> 4) | (
            (value8[:, 3] & 0x3F) << 4
        )
        value[:, 3] = ((value8[:, 3] & 0xC0) >> 6) | (
            (value8[:, 4] & 0xFF) << 2
        )
        return value

    def unpack12(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        value8 = value8.view(np.uint8).reshape((-1, 3))
        value = np.empty((value8.shape[0], 2), self.dtype)
        value[:, 0] = ((value8[:, 0] & 0xFF) >> 0) | (
            (value8[:, 1] & 0x0F) << 8
        )
        value[:, 1] = ((value8[:, 1] & 0xF0) >> 4) | (
            (value8[:, 2] & 0xFF) << 4
        )
        return value

    def unpack(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        if self.expanded:
            value = value8.view(self.dtype)
        else:
            match self.bits:
                case 10:
                    value = self.unpack10(value8)
                case 12:
                    value = self.unpack12(value8)
                case _:
                    value = value8.view(self.dtype)
        return value

    def pack10(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        value = value.view(self.dtype).reshape((-1, 4))
        value8 = np.empty((value.shape[0], 5), np.uint8)
        value8[:, 0] = (value[:, 0] & 0x03FC) >> 2  # type: ignore[operator]
        value8[:, 1] = ((value[:, 0] & 0x0003) << 6) | (  # type: ignore[operator]
            (value[:, 1] & 0x03F0) >> 4  # type: ignore[operator]
        )
        value8[:, 2] = ((value[:, 1] & 0x000F) << 4) | (  # type: ignore[operator]
            (value[:, 2] & 0x03C0) >> 6  # type: ignore[operator]
        )
        value8[:, 3] = ((value[:, 2] & 0x003F) << 2) | (  # type: ignore[operator]
            (value[:, 3] & 0x0300) >> 8  # type: ignore[operator]
        )
        value8[:, 4] = (value[:, 3] & 0x00FF) << 0  # type: ignore[operator]
        return value8

    def pack12(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        value = value.view(self.dtype).reshape((-1, 2))
        value8 = np.empty((value.shape[0], 3), np.uint8)
        value8[:, 0] = (value[:, 0] & 0x0FF0) >> 4  # type: ignore[operator]
        value8[:, 1] = ((value[:, 0] & 0x000F) << 4) | (  # type: ignore[operator]
            (value[:, 1] & 0x0F00) >> 8  # type: ignore[operator]
        )
        value8[:, 2] = (value[:, 1] & 0x00FF) << 0  # type: ignore[operator]
        return value8

    def pack(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        if self.expanded:
            value8 = value.view(np.uint8)
        else:
            match self.bits:
                case 10:
                    value8 = self.pack10(value)
                case 12:
                    value8 = self.pack12(value)
                case _:
                    value8 = value.view(np.uint8)
        return value8

    def data2key(self, value8: NDArray[np.uint8]) -> NDArray[np.float32]:
        value = self.to_yuv(value8)
        if self.is_integer:
            value = value / (2**self.bits - 1)
        value = np.astype(value, np.float32)
        return value

    def key2data(self, value: NDArray[np.float32]) -> NDArray[np.uint8]:
        if self.is_integer:
            value = value * (2**self.bits - 1)
        value = np.astype(value, self.dtype)
        value8 = self.from_yuv(value)
        return value8

    def to_yuv(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        raise NotImplementedError(
            'to_yuv not implemented for this pixel format.'
        )

    def from_yuv(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        raise NotImplementedError(
            'from_yuv not implemented for this pixel format.'
        )


@dataclass
class Y(YUVFormat):
    @functools.cached_property
    def bits_per_pixel(self) -> int:
        return int(self.item_size * 8)

    @functools.cached_property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride

    def to_yuv(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        value8 = value8.view(np.uint8).reshape((self.height, self.stride))
        y8 = value8[: self.height, : self.pitch]
        y = self.unpack(y8)
        y = y.view(self.dtype).reshape((self.height, self.width))
        u = v = np.full_like(y, self.fill_value)
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = self.pack(y)
        y8 = y8.view(np.uint8).reshape((self.height, self.pitch))
        y8 = np.pad(y8, ({1: (0, self.padding)}), constant_values=0)
        value8 = y8.view(np.uint8).reshape((self.height, self.stride))
        return value8


@dataclass
class GRAY(Y):
    name: ClassVar[str] = 'GRAY'
    is_integer: ClassVar[bool] = True


@dataclass
class GRAYF(Y):
    name: ClassVar[str] = 'GRAYF'
    is_integer: ClassVar[bool] = False


@dataclass
class YUV420(YUVFormat):
    is_integer: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.height % 2 != 0 or self.width % 2 != 0:
            raise ValueError('YUV420 formats require even height and width.')
        super().__post_init__()

    @functools.cached_property
    def bits_per_pixel(self) -> int:
        return int(self.item_size * 12)

    @functools.cached_property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride * 3 // 2


@dataclass
class YUV420Planar(YUV420):
    def to_yuv(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        value8 = value8.view(np.uint8).reshape(
            (self.height * 3 // 2, self.stride)
        )
        y8 = value8[: self.height, : self.stride]
        y8 = y8.reshape((self.height, self.stride))[: self.height, : self.pitch]
        y = self.unpack(y8)
        y = y.view(self.dtype).reshape((self.height, self.width))
        uv8 = value8[self.height :, : self.stride]
        uv8 = uv8.reshape((2, self.height // 2, self.stride // 2))[
            :, : self.height // 2, : self.pitch // 2
        ]
        uv = self.unpack(uv8)
        uv = uv.view(self.dtype).reshape((2, self.height // 2, self.width // 2))
        uv = uv.repeat(2, axis=1).repeat(2, axis=2).transpose((1, 2, 0))
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[..., ::-1]
        value = np.dstack((y, uv))
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, uv = np.dsplit(value, [1])
        y8 = self.pack(y)
        y8 = y8.view(np.uint8).reshape((self.height, self.pitch))
        y8 = np.pad(y8, {1: (0, self.padding)}, constant_values=0)
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[::-1, ...]
        uv = (
            uv.transpose((2, 0, 1))
            .reshape((2, self.height // 2, 2, self.width // 2, 2))
            .mean(axis=(2, 4))
            + 0.5
        ).astype(self.dtype)
        uv8 = self.pack(np.ascontiguousarray(uv))
        uv8 = uv8.view(np.uint8).reshape((2, self.height // 2, self.pitch // 2))
        uv8 = np.pad(
            uv8, {2: (0, self.padding // 2)}, constant_values=self.fill_value
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value8 = value8.view(np.uint8).reshape(
            (self.height * 3 // 2, self.stride)
        )
        return value8


@dataclass
class YUV420_I420(YUV420Planar):
    name: ClassVar[str] = 'YUV420_I420'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.UV


@dataclass
class YUV420_YV12(YUV420Planar):
    name: ClassVar[str] = 'YUV420_YV12'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.VU


@dataclass
class YUV420SemiPlanar(YUV420):
    def to_yuv(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        value8 = value8.view(np.uint8).reshape(
            (self.height * 3 // 2, self.stride)
        )
        y8 = value8[: self.height, : self.stride]
        y8 = y8.reshape((self.height, self.stride))[: self.height, : self.pitch]
        y = self.unpack(y8)
        y = y.view(self.dtype).reshape((self.height, self.width))
        uv8 = value8[self.height :, : self.stride]
        uv8 = uv8.reshape((self.height // 2, self.stride))[
            : self.height // 2, : self.pitch
        ]
        uv = self.unpack(uv8)
        uv = uv.view(self.dtype).reshape((self.height // 2, self.width // 2, 2))
        uv = uv.repeat(2, axis=0).repeat(2, axis=1)
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[..., ::-1]
        value = np.dstack((y, uv))
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, uv = np.dsplit(value, [1])
        y8 = self.pack(y)
        y8 = y8.view(np.uint8).reshape((self.height, self.pitch))
        y8 = np.pad(y8, {1: (0, self.padding)}, constant_values=0)
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[..., ::-1]
        uv = (
            uv.reshape((self.height // 2, 2, self.width // 2, 2, 2)).mean(
                axis=(1, 3)
            )
            + 0.5
        ).astype(self.dtype)
        uv8 = self.pack(uv)
        uv8 = uv8.view(np.uint8).reshape((self.height // 2, self.pitch))
        uv8 = np.pad(
            uv8, {1: (0, self.padding)}, constant_values=self.fill_value
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value8 = value8.view(np.uint8).reshape(
            (self.height * 3 // 2, self.stride)
        )
        return value8


@dataclass
class YUV420_NV12(YUV420SemiPlanar):
    name: ClassVar[str] = 'YUV420_NV12'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.UV


@dataclass
class YUV420_NV21(YUV420SemiPlanar):
    name: ClassVar[str] = 'YUV420_NV21'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.VU


@dataclass
class YUV422(YUVFormat):
    is_integer: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.height % 2 != 0 or self.width % 2 != 0:
            raise ValueError('YUV422 formats require even height and width.')
        super().__post_init__()

    @functools.cached_property
    def bits_per_pixel(self) -> int:
        return int(self.item_size * 16)

    @functools.cached_property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride * 2


@dataclass
class YUV422Planar(YUV422):
    def to_yuv(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        value8 = value8.view(np.uint8).reshape((self.height * 2, self.stride))
        y8 = value8[: self.height, : self.stride]
        y8 = y8.reshape((self.height, self.stride))[: self.height, : self.pitch]
        y = self.unpack(y8)
        y = y.view(self.dtype).reshape((self.height, self.width))
        uv8 = value8[self.height :, : self.stride]
        uv8 = uv8.reshape((2, self.height, self.stride // 2))[
            :, : self.height, : self.pitch // 2
        ]
        uv = self.unpack(uv8)
        uv = uv.view(self.dtype).reshape((2, self.height, self.width // 2))
        uv = uv.repeat(2, axis=2).transpose((1, 2, 0))
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[..., ::-1]
        value = np.dstack((y, uv))
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, uv = np.dsplit(value, [1])
        y8 = self.pack(y)
        y8 = y8.view(np.uint8).reshape((self.height, self.pitch))
        y8 = np.pad(y8, {1: (0, self.padding)}, constant_values=0)
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[::-1, ...]
        uv = (
            uv.transpose((2, 0, 1))
            .reshape((2, self.height, self.width // 2, 2))
            .mean(axis=3)
            + 0.5
        ).astype(self.dtype)
        uv8 = self.pack(np.ascontiguousarray(uv))
        uv8 = uv8.view(np.uint8).reshape((2, self.height, self.pitch // 2))
        uv8 = np.pad(
            uv8, {2: (0, self.padding // 2)}, constant_values=self.fill_value
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value8 = value8.view(np.uint8).reshape((self.height * 2, self.stride))
        return value8


@dataclass
class YUV422_I422(YUV422Planar):
    name: ClassVar[str] = 'YUV422_I422'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.UV


@dataclass
class YUV422_YV16(YUV422Planar):
    name: ClassVar[str] = 'YUV422_YV16'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.VU


@dataclass
class YUV422SemiPlanar(YUV422):
    def to_yuv(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        value8 = value8.view(np.uint8).reshape((self.height * 2, self.stride))
        y8 = value8[: self.height, : self.stride]
        y8 = y8.reshape((self.height, self.stride))[: self.height, : self.pitch]
        y = self.unpack(y8)
        y = y.view(self.dtype).reshape((self.height, self.width))
        uv8 = value8[self.height :, : self.stride]
        uv8 = uv8.reshape((self.height, self.stride))[
            : self.height, : self.pitch
        ]
        uv = self.unpack(uv8)
        uv = uv.view(self.dtype).reshape((self.height, self.width // 2, 2))
        uv = uv.repeat(2, axis=1)
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[..., ::-1]
        value = np.dstack((y, uv))
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, uv = np.dsplit(value, [1])
        y8 = self.pack(y)
        y8 = y8.view(np.uint8).reshape((self.height, self.pitch))
        y8 = np.pad(y8, {1: (0, self.padding)}, constant_values=0)
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[..., ::-1]
        uv = (
            uv.reshape((self.height, self.width // 2, 2, 2)).mean(axis=2) + 0.5
        ).astype(self.dtype)
        uv8 = self.pack(uv)
        uv8 = uv8.view(np.uint8).reshape((self.height, self.pitch))
        uv8 = np.pad(
            uv8, {1: (0, self.padding)}, constant_values=self.fill_value
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value8 = value8.view(np.uint8).reshape((self.height * 2, self.stride))
        return value8


@dataclass
class YUV422_NV16(YUV422SemiPlanar):
    name: ClassVar[str] = 'YUV422_NV16'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.UV


@dataclass
class YUV422_NV61(YUV422SemiPlanar):
    name: ClassVar[str] = 'YUV422_NV61'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.VU


@dataclass
class YUV422Packed(YUV422):
    packed_order: ClassVar[PackedOrder]

    @functools.cached_property
    def pitch(self) -> int:
        return int(self.width * self.item_size * 2)

    @functools.cached_property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride

    def to_yuv(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        value8 = value8.view(np.uint8).reshape((self.height, self.stride))
        value8 = value8[: self.height, : self.pitch]
        value = self.unpack(value8)
        value = value.view(self.dtype).reshape((self.height, self.width, 2))
        if self.packed_order == PackedOrder.YUV:
            y, uv = np.split(value, [1], axis=-1)
        else:
            uv, y = np.split(value, [1], axis=-1)
        uv = uv.view(self.dtype).reshape((self.height, self.width // 2, 2))
        uv = uv.repeat(2, axis=1)
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[..., ::-1]
        value = np.dstack((y, uv))
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, uv = np.dsplit(value, [1])
        y = y.view(self.dtype).reshape((self.height, self.width // 2, 2))
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[..., ::-1]
        uv = (
            uv.reshape((self.height, self.width // 2, 2, 2)).mean(axis=2) + 0.5
        ).astype(self.dtype)
        if self.packed_order == PackedOrder.YUV:
            value = np.stack((y, uv), axis=-1)
        else:
            value = np.stack((uv, y), axis=-1)
        value8 = self.pack(value)
        value8 = value8.view(np.uint8).reshape((self.height, self.pitch))
        value8 = np.pad(value8, {1: (0, self.padding)}, constant_values=0)
        value8 = value8.view(np.uint8).reshape((self.height, self.stride))
        return value8


@dataclass
class YUV422_YUYV(YUV422Packed):
    name: ClassVar[str] = 'YUV422_YUYV'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.UV
    packed_order: ClassVar[PackedOrder] = PackedOrder.YUV


@dataclass
class YUV422_YVYU(YUV422Packed):
    name: ClassVar[str] = 'YUV422_YVYU'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.VU
    packed_order: ClassVar[PackedOrder] = PackedOrder.YUV


@dataclass
class YUV422_UYVY(YUV422Packed):
    name: ClassVar[str] = 'YUV422_UYVY'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.UV
    packed_order: ClassVar[PackedOrder] = PackedOrder.UVY


@dataclass
class YUV422_VYUY(YUV422Packed):
    name: ClassVar[str] = 'YUV422_VYUY'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.VU
    packed_order: ClassVar[PackedOrder] = PackedOrder.UVY


@dataclass
class YUV444(YUVFormat):
    is_integer: ClassVar[bool] = True

    @functools.cached_property
    def bits_per_pixel(self) -> int:
        return int(self.item_size * 24)

    @functools.cached_property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride * 3


@dataclass
class YUV444Planar(YUV444):
    def to_yuv(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        value8 = value8.view(np.uint8).reshape((3, self.height, self.stride))
        y8 = value8[0, : self.height, : self.stride]
        y8 = y8.reshape((self.height, self.stride))[: self.height, : self.pitch]
        y = self.unpack(y8)
        y = y.view(self.dtype).reshape((self.height, self.width))
        uv8 = value8[1:, : self.height, : self.stride]
        uv8 = uv8.reshape((2, self.height, self.stride))[
            :, : self.height, : self.pitch
        ]
        uv = self.unpack(uv8)
        uv = uv.view(self.dtype).reshape((2, self.height, self.width))
        uv = uv.transpose((1, 2, 0))
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[..., ::-1]
        value = np.dstack((y, uv))
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, uv = np.dsplit(value, [1])
        y8 = self.pack(y)
        y8 = y8.view(np.uint8).reshape((self.height, self.pitch))
        y8 = np.pad(y8, {1: (0, self.padding)}, constant_values=0)
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[::-1, ...]
        uv = uv.transpose((2, 0, 1)).reshape((2, self.height, self.width))
        uv8 = self.pack(np.ascontiguousarray(uv))
        uv8 = uv8.view(np.uint8).reshape((2, self.height, self.pitch))
        uv8 = np.pad(
            uv8, {2: (0, self.padding)}, constant_values=self.fill_value
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value8 = value8.view(np.uint8).reshape((3, self.height, self.stride))
        return value8


@dataclass
class YUV444_I444(YUV444Planar):
    name: ClassVar[str] = 'YUV444_I444'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.UV


@dataclass
class YUV444_YV24(YUV444Planar):
    name: ClassVar[str] = 'YUV444_YV24'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.VU


@dataclass
class YUV444SemiPlanar(YUV444):
    def to_yuv(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        value8 = value8.view(np.uint8).reshape((3, self.height, self.stride))
        y8 = value8[0, : self.height, : self.stride]
        y8 = y8.reshape((self.height, self.stride))[: self.height, : self.pitch]
        y = self.unpack(y8)
        y = y.view(self.dtype).reshape((self.height, self.width))
        uv8 = value8[1:, : self.height, : self.stride]
        uv8 = uv8.reshape((self.height, self.stride * 2))[
            : self.height, : self.pitch * 2
        ]
        uv = self.unpack(uv8)
        uv = uv.view(self.dtype).reshape((self.height, self.width, 2))
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[..., ::-1]
        value = np.dstack((y, uv))
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, uv = np.dsplit(value, [1])
        y8 = self.pack(y)
        y8 = y8.view(np.uint8).reshape((self.height, self.pitch))
        y8 = np.pad(y8, {1: (0, self.padding)}, constant_values=0)
        if self.chroma_order == ChromaOrder.VU:
            uv = uv[..., ::-1]
        uv8 = self.pack(uv)
        uv8 = uv8.view(np.uint8).reshape((self.height, self.pitch * 2))
        uv8 = np.pad(
            uv8, {1: (0, self.padding * 2)}, constant_values=self.fill_value
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value8 = value8.view(np.uint8).reshape((3, self.height, self.stride))
        return value8


@dataclass
class YUV444_NV24(YUV444SemiPlanar):
    name: ClassVar[str] = 'YUV444_NV24'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.UV


@dataclass
class YUV444_NV42(YUV444SemiPlanar):
    name: ClassVar[str] = 'YUV444_NV42'
    chroma_order: ClassVar[ChromaOrder] = ChromaOrder.VU


@dataclass
class RGBFormat(PixelFormat):
    def to_rgb(self, value8: NDArray[np.uint8]) -> NDArray[Any]:
        raise NotImplementedError(
            'to_rgb not implemented for this pixel format.'
        )

    def from_rgb(self, value: NDArray[Any]) -> NDArray[np.uint8]:
        raise NotImplementedError(
            'from_rgb not implemented for this pixel format.'
        )


@dataclass
class RGB24(RGBFormat):
    name: ClassVar[str] = 'RGB24'

    @functools.cached_property
    def pitch(self) -> int:
        return self.width * 3

    @functools.cached_property
    def bits_per_pixel(self) -> int:
        return 24

    @functools.cached_property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride

    def data2key(self, value8: NDArray[np.uint8]) -> NDArray[np.float32]:
        return np.astype(self.to_rgb(value8) / 255.0, np.float32)

    def key2data(self, value: NDArray[np.float32]) -> NDArray[np.uint8]:
        return self.from_rgb(np.astype(value * 255.0, np.uint8))

    def to_rgb(self, value8: NDArray[np.uint8]) -> NDArray[np.uint8]:
        value8 = value8.view(np.uint8).reshape((self.height, self.stride))
        value8 = value8[: self.height, : self.pitch]
        value = value8.view(np.uint8).reshape((self.height, self.width, 3))
        return value

    def from_rgb(self, value: NDArray[np.uint8]) -> NDArray[np.uint8]:
        value8 = value.view(np.uint8).reshape((self.height, self.pitch))
        value8 = np.pad(value8, {0: (0, self.padding)}, constant_values=0)
        value8 = value8.view(np.uint8).reshape((self.height, self.stride))
        return value8


@dataclass
class BGR24(RGBFormat):
    name: ClassVar[str] = 'BGR24'

    @functools.cached_property
    def pitch(self) -> int:
        return self.width * 3

    @functools.cached_property
    def bits_per_pixel(self) -> int:
        return 24

    @functools.cached_property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride

    def data2key(self, value8: NDArray[np.uint8]) -> NDArray[np.float32]:
        return np.astype(self.to_rgb(value8) / 255.0, np.float32)

    def key2data(self, value: NDArray[np.float32]) -> NDArray[np.uint8]:
        return self.from_rgb(np.astype(value * 255.0, np.uint8))

    def to_rgb(self, value8: NDArray[np.uint8]) -> NDArray[np.uint8]:
        value8 = value8.view(np.uint8).reshape((self.height, self.stride))
        value8 = value8[: self.height, : self.pitch]
        value = value8.view(np.uint8).reshape((self.height, self.width, 3))
        value = value[..., ::-1]
        return value

    def from_rgb(self, value: NDArray[np.uint8]) -> NDArray[np.uint8]:
        value = value.view(np.uint8).reshape((self.height, self.width, 3))
        value = value[..., ::-1]
        value8 = value.view(np.uint8).reshape((self.height, self.pitch))
        value8 = np.pad(value8, {0: (0, self.padding)}, constant_values=0)
        return value8


@dataclass
class RGBF32(RGBFormat):
    name: ClassVar[str] = 'RGBF32'

    @functools.cached_property
    def pitch(self) -> int:
        return self.width * 12

    @functools.cached_property
    def bits_per_pixel(self) -> int:
        return 96

    @functools.cached_property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride

    def data2key(self, value8: NDArray[Any]) -> NDArray[np.float32]:
        return self.to_rgb(value8)

    def key2data(self, value: NDArray[np.float32]) -> NDArray[Any]:
        return self.from_rgb(value)

    def to_rgb(self, value8: NDArray[np.uint8]) -> NDArray[np.float32]:
        value8 = value8.view(np.uint8).reshape((self.height, self.stride))
        value8 = value8[: self.height, : self.pitch]
        value = value8.view(np.float32).reshape((self.height, self.width, 3))
        return value

    def from_rgb(self, value: NDArray[np.float32]) -> NDArray[np.uint8]:
        value8 = value.view(np.uint8).reshape((self.height, self.pitch))
        value8 = np.pad(value8, {0: (0, self.padding)}, constant_values=0)
        value8 = value8.view(np.uint8).reshape((self.height, self.stride))
        return value8


@dataclass
class XYZFormat(PixelFormat):
    pass
