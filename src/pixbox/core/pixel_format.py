import math
from dataclasses import dataclass
from typing import Any, ClassVar, Type

import numpy as np
from numpy.typing import NDArray


@dataclass
class PixelFormat:
    name: ClassVar[str]
    height: int
    width: int
    stride: int = 0

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

    def data2key(self, value: NDArray[Any]) -> NDArray[np.float32]:
        raise NotImplementedError(
            'data2key not implemented for this pixel format.'
        )

    def key2data(self, value: NDArray[np.float32]) -> NDArray[Any]:
        raise NotImplementedError(
            'key2data not implemented for this pixel format.'
        )


@dataclass
class YUVFormat(PixelFormat):
    bits: int = 8
    packed: bool = False

    @property
    def item_size(self) -> float:
        if self.packed:
            return self.bits / 8
        else:
            return math.ceil(self.bits / 8)

    @property
    def dtype(self) -> Type[np.uint8 | np.uint16]:
        if self.bits <= 8:
            return np.uint8
        elif self.bits <= 16:
            return np.uint16
        raise ValueError(f'Unsupported bit depth: {self.bits}.')

    def data2key(self, value: NDArray[Any]) -> NDArray[np.float32]:
        value = np.astype((self.to_yuv(value) / (2**self.bits - 1)), np.float32)
        value[:, :, 1:] -= np.float32(0.5)
        return value

    def key2data(self, value: NDArray[np.float32]) -> NDArray[Any]:
        value[:, :, 1:] += np.float32(0.5)
        value = self.from_yuv(np.astype(value * (2**self.bits - 1), self.dtype))
        return value

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        raise NotImplementedError(
            'to_yuv not implemented for this pixel format.'
        )

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        raise NotImplementedError(
            'from_yuv not implemented for this pixel format.'
        )


@dataclass
class YUV420(YUVFormat):
    @property
    def bits_per_pixel(self) -> int:
        return int(self.item_size * 8 * 3 // 2)

    @property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride * 3 // 2


@dataclass
class YUV420Planar(YUV420):
    def __post_init__(self) -> None:
        self.stride = max(self.stride, int(self.width * self.item_size))


@dataclass
class YUV420_I420(YUV420Planar):
    name: ClassVar[str] = 'YUV420_I420'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape(
            (self.height * 3 // 2, self.stride)
        )
        y8 = value8[: self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        u8 = value8[self.height : self.height * 5 // 4, : self.stride]
        u = u8.view(self.dtype).reshape((self.height // 2, -1))
        u = (
            u[: self.height // 2, : self.width // 2]
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )
        v8 = value8[self.height * 5 // 4 :, : self.stride]
        v = v8.view(self.dtype).reshape((self.height // 2, -1))
        v = (
            v[: self.height // 2, : self.width // 2]
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = np.astype(
            u.reshape((self.height // 2, 2, self.width // 2, 2)).mean(
                axis=(1, 3)
            )
            + 0.5,
            self.dtype,
        )
        u8 = u.view(np.uint8).reshape((self.height // 2, -1))
        u8 = np.pad(
            u8,
            ((0, self.stride // 2 - u8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        v = np.astype(
            v.reshape((self.height // 2, 2, self.width // 2, 2)).mean(
                axis=(1, 3)
            )
            + 0.5,
            self.dtype,
        )
        v8 = v.view(np.uint8).reshape((self.height // 2, -1))
        v8 = np.pad(
            v8,
            ((0, self.stride // 2 - v8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), u8.ravel(), v8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((self.height * 3 // 2, -1))
        return value


@dataclass
class YUV420_YV12(YUV420Planar):
    name: ClassVar[str] = 'YUV420_YV12'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape(
            (self.height * 3 // 2, self.stride)
        )
        y8 = value8[: self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        v8 = value8[self.height : self.height * 5 // 4, : self.stride]
        v = v8.view(self.dtype).reshape((self.height // 2, -1))
        v = (
            v[: self.height // 2, : self.width // 2]
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )
        u8 = value8[self.height * 5 // 4 :, : self.stride]
        u = u8.view(self.dtype).reshape((self.height // 2, -1))
        u = (
            u[: self.height // 2, : self.width // 2]
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = np.astype(
            u.reshape((self.height // 2, 2, self.width // 2, 2)).mean(
                axis=(1, 3)
            )
            + 0.5,
            self.dtype,
        )
        u8 = u.view(np.uint8).reshape((self.height // 2, -1))
        u8 = np.pad(
            u8,
            ((0, self.stride // 2 - u8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        v = np.astype(
            v.reshape((self.height // 2, 2, self.width // 2, 2)).mean(
                axis=(1, 3)
            )
            + 0.5,
            self.dtype,
        )
        v8 = v.view(np.uint8).reshape((self.height // 2, -1))
        v8 = np.pad(
            v8,
            ((0, self.stride // 2 - v8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), v8.ravel(), u8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((self.height * 3 // 2, -1))
        return value


@dataclass
class YUV420SemiPlanar(YUV420):
    def __post_init__(self) -> None:
        self.stride = max(self.stride, int(self.width * self.item_size))


@dataclass
class YUV420_NV12(YUV420SemiPlanar):
    name: ClassVar[str] = 'YUV420_NV12'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape(
            (self.height * 3 // 2, self.stride)
        )
        y8 = value8[: self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        uv8 = value8[self.height :, : self.stride]
        uv = uv8.view(self.dtype).reshape((self.height // 2, -1))
        u = (
            uv[: self.height // 2, : self.width : 2]
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )
        v = (
            uv[: self.height // 2, 1 : self.width : 2]
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = np.astype(
            u.reshape((self.height // 2, 2, self.width // 2, 2)).mean(
                axis=(1, 3)
            )
            + 0.5,
            self.dtype,
        )
        v = np.astype(
            v.reshape((self.height // 2, 2, self.width // 2, 2)).mean(
                axis=(1, 3)
            )
            + 0.5,
            self.dtype,
        )
        uv = np.dstack((u, v))
        uv8 = uv.view(np.uint8).reshape((self.height // 2, -1))
        uv8 = np.pad(
            uv8,
            ((0, self.stride - uv8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((self.height * 3 // 2, -1))
        return value


@dataclass
class YUV420_NV21(YUV420SemiPlanar):
    name: ClassVar[str] = 'YUV420_NV21'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape(
            (self.height * 3 // 2, self.stride)
        )
        y8 = value8[: self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        uv8 = value8[self.height :, : self.stride]
        uv = uv8.view(self.dtype).reshape((self.height // 2, -1))
        v = (
            uv[: self.height // 2, : self.width : 2]
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )
        u = (
            uv[: self.height // 2, 1 : self.width : 2]
            .repeat(2, axis=0)
            .repeat(2, axis=1)
        )
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = np.astype(
            u.reshape((self.height // 2, 2, self.width // 2, 2)).mean(
                axis=(1, 3)
            )
            + 0.5,
            self.dtype,
        )
        v = np.astype(
            v.reshape((self.height // 2, 2, self.width // 2, 2)).mean(
                axis=(1, 3)
            )
            + 0.5,
            self.dtype,
        )
        uv = np.dstack((v, u))
        uv8 = uv.view(np.uint8).reshape((self.height // 2, -1))
        uv8 = np.pad(
            uv8,
            ((0, self.stride - uv8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((self.height * 3 // 2, -1))
        return value


@dataclass
class YUV422(YUVFormat):
    @property
    def bits_per_pixel(self) -> int:
        return int(self.item_size * 8 * 2)

    @property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride * 2


@dataclass
class YUV422Planar(YUV422):
    def __post_init__(self) -> None:
        self.stride = max(self.stride, int(self.width * self.item_size))


@dataclass
class YUV422_I422(YUV422Planar):
    name: ClassVar[str] = 'YUV422_I422'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape((self.height * 2, self.stride))
        y8 = value8[: self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        u8 = value8[self.height : self.height * 3 // 2, : self.stride]
        u = u8.view(self.dtype).reshape((self.height, -1))
        u = u[: self.height, : self.width // 2].repeat(2, axis=1)
        v8 = value8[self.height * 3 // 2 :, : self.stride]
        v = v8.view(self.dtype).reshape((self.height, -1))
        v = v[: self.height, : self.width // 2].repeat(2, axis=1)
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = np.astype(
            u.reshape((self.height, self.width // 2, 2)).mean(axis=2) + 0.5,
            self.dtype,
        )
        u8 = u.view(np.uint8).reshape((self.height, -1))
        u8 = np.pad(
            u8,
            ((0, self.stride // 2 - u8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        v = np.astype(
            v.reshape((self.height, self.width // 2, 2)).mean(axis=2) + 0.5,
            self.dtype,
        )
        v8 = v.view(np.uint8).reshape((self.height, -1))
        v8 = np.pad(
            v8,
            ((0, self.stride // 2 - v8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), u8.ravel(), v8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((self.height * 2, -1))
        return value


@dataclass
class YUV422_YV16(YUV422Planar):
    name: ClassVar[str] = 'YUV422_YV16'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape((self.height * 2, self.stride))
        y8 = value8[: self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        v8 = value8[self.height : self.height * 3 // 2, : self.stride]
        v = v8.view(self.dtype).reshape((self.height, -1))
        v = v[: self.height, : self.width // 2].repeat(2, axis=1)
        u8 = value8[self.height * 3 // 2 :, : self.stride]
        u = u8.view(self.dtype).reshape((self.height, -1))
        u = u[: self.height, : self.width // 2].repeat(2, axis=1)
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = np.astype(
            u.reshape((self.height, self.width // 2, 2)).mean(axis=2) + 0.5,
            self.dtype,
        )
        u8 = u.view(np.uint8).reshape((self.height, -1))
        u8 = np.pad(
            u8,
            ((0, self.stride // 2 - u8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        v = np.astype(
            v.reshape((self.height, self.width // 2, 2)).mean(axis=2) + 0.5,
            self.dtype,
        )
        v8 = v.view(np.uint8).reshape((self.height, -1))
        v8 = np.pad(
            v8,
            ((0, self.stride // 2 - v8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), v8.ravel(), u8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((self.height * 2, -1))
        return value


@dataclass
class YUV422SemiPlanar(YUV422):
    def __post_init__(self) -> None:
        self.stride = max(self.stride, int(self.width * self.item_size))


@dataclass
class YUV422_NV16(YUV422SemiPlanar):
    name: ClassVar[str] = 'YUV422_NV16'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape((self.height * 2, self.stride))
        y8 = value8[: self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        uv8 = value8[self.height :, : self.stride]
        uv = uv8.view(self.dtype).reshape((self.height, -1))
        u = uv[: self.height, : self.width : 2].repeat(2, axis=1)
        v = uv[: self.height, 1 : self.width : 2].repeat(2, axis=1)
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = np.astype(
            u.reshape((self.height, self.width // 2, 2)).mean(axis=2) + 0.5,
            self.dtype,
        )
        v = np.astype(
            v.reshape((self.height, self.width // 2, 2)).mean(axis=2) + 0.5,
            self.dtype,
        )
        uv = np.dstack((u, v))
        uv8 = uv.view(np.uint8).reshape((self.height, -1))
        uv8 = np.pad(
            uv8,
            ((0, self.stride - uv8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((self.height * 2, -1))
        return value


@dataclass
class YUV422_NV61(YUV422SemiPlanar):
    name: ClassVar[str] = 'YUV422_NV61'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape((self.height * 2, self.stride))
        y8 = value8[: self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        uv8 = value8[self.height :, : self.stride]
        uv = uv8.view(self.dtype).reshape((self.height, -1))
        v = uv[: self.height, : self.width : 2].repeat(2, axis=1)
        u = uv[: self.height, 1 : self.width : 2].repeat(2, axis=1)
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = np.astype(
            u.reshape((self.height, self.width // 2, 2)).mean(axis=2) + 0.5,
            self.dtype,
        )
        v = np.astype(
            v.reshape((self.height, self.width // 2, 2)).mean(axis=2) + 0.5,
            self.dtype,
        )
        uv = np.dstack((v, u))
        uv8 = uv.view(np.uint8).reshape((self.height, -1))
        uv8 = np.pad(
            uv8,
            ((0, self.stride - uv8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((self.height * 2, -1))
        return value


@dataclass
class YUV422Packed(YUV422):
    pass


@dataclass
class YUV422_UYVY(YUV422Packed):
    pass


@dataclass
class YUV422_VYUY(YUV422Packed):
    pass


@dataclass
class YUV444(YUVFormat):
    @property
    def bits_per_pixel(self) -> int:
        return int(self.item_size * 8 * 3)

    @property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride * 3


@dataclass
class YUV444Planar(YUV444):
    def __post_init__(self) -> None:
        self.stride = max(self.stride, int(self.width * self.item_size))


@dataclass
class YUV444_I444(YUV444Planar):
    name: ClassVar[str] = 'YUV444_I444'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape((3, self.height, self.stride))
        y8 = value8[0, : self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        u8 = value8[1, : self.height, : self.stride]
        u = u8.view(self.dtype).reshape((self.height, -1))
        u = u[: self.height, : self.width]
        v8 = value8[2, : self.height, : self.stride]
        v = v8.view(self.dtype).reshape((self.height, -1))
        v = v[: self.height, : self.width]
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = u.reshape((self.height, self.width)).copy()
        u8 = u.view(np.uint8).reshape((self.height, -1))
        u8 = np.pad(
            u8,
            ((0, self.stride - u8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        v = v.reshape((self.height, self.width)).copy()
        v8 = v.view(np.uint8).reshape((self.height, -1))
        v8 = np.pad(
            v8,
            ((0, self.stride - v8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), u8.ravel(), v8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((3, self.height, -1))
        return value


@dataclass
class YUV444_YV24(YUV444Planar):
    name: ClassVar[str] = 'YUV444_YV24'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape((3, self.height, self.stride))
        y8 = value8[0, : self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        v8 = value8[1, : self.height, : self.stride]
        v = v8.view(self.dtype).reshape((self.height, -1))
        v = v[: self.height, : self.width]
        u8 = value8[2, : self.height, : self.stride]
        u = u8.view(self.dtype).reshape((self.height, -1))
        u = u[: self.height, : self.width]
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = u.reshape((self.height, self.width)).copy()
        u8 = u.view(np.uint8).reshape((self.height, -1))
        u8 = np.pad(
            u8,
            ((0, self.stride - u8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        v = v.reshape((self.height, self.width)).copy()
        v8 = v.view(np.uint8).reshape((self.height, -1))
        v8 = np.pad(
            v8,
            ((0, self.stride - v8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), v8.ravel(), u8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((3, self.height, -1))
        return value


@dataclass
class YUV444SemiPlanar(YUV444):
    def __post_init__(self) -> None:
        self.stride = max(self.stride, int(self.width * self.item_size))


@dataclass
class YUV444_NV24(YUV444SemiPlanar):
    name: ClassVar[str] = 'YUV444_NV24'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape((self.height * 3, self.stride))
        y8 = value8[: self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        uv8 = value8[self.height :, : self.stride]
        uv = uv8.view(self.dtype).reshape((self.height, -1))
        u = uv[: self.height, : self.width * 2 : 2]
        v = uv[: self.height, 1 : self.width * 2 : 2]
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = u.reshape((self.height, self.width))
        v = v.reshape((self.height, self.width))
        uv = np.dstack((u, v))
        uv8 = uv.view(np.uint8).reshape((self.height, -1))
        uv8 = np.pad(
            uv8,
            ((0, self.stride * 2 - uv8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((self.height * 3, -1))
        return value


@dataclass
class YUV444_NV42(YUV444SemiPlanar):
    name: ClassVar[str] = 'YUV444_NV42'

    def to_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value8 = value.view(np.uint8).reshape((self.height * 3, self.stride))
        y8 = value8[: self.height, : self.stride]
        y = y8.view(self.dtype).reshape((self.height, -1))
        y = y[: self.height, : self.width]
        uv8 = value8[self.height :, : self.stride]
        uv = uv8.view(self.dtype).reshape((self.height, -1))
        v = uv[: self.height, : self.width * 2 : 2]
        u = uv[: self.height, 1 : self.width * 2 : 2]
        value = np.dstack((y, u, v))
        return value

    def from_yuv(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.view(self.dtype).reshape((self.height, self.width, 3))
        y, u, v = np.dsplit(value, 3)
        y8 = y.view(np.uint8).reshape((self.height, -1))
        y8 = np.pad(
            y8, ((0, self.stride - y8.shape[1]), (0, 0)), constant_values=0
        )
        u = u.reshape((self.height, self.width))
        v = v.reshape((self.height, self.width))
        uv = np.dstack((v, u))
        uv8 = uv.view(np.uint8).reshape((self.height, -1))
        uv8 = np.pad(
            uv8,
            ((0, self.stride * 2 - uv8.shape[1]), (0, 0)),
            constant_values=1 << (self.bits - 1),
        )
        value8 = np.concatenate((y8.ravel(), uv8.ravel()), axis=-1)
        value = value8.view(self.dtype).reshape((self.height * 3, -1))
        return value


@dataclass
class RGBFormat(PixelFormat):
    def to_rgb(self, value: NDArray[Any]) -> NDArray[Any]:
        raise NotImplementedError(
            'to_rgb not implemented for this pixel format.'
        )

    def from_rgb(self, value: NDArray[Any]) -> NDArray[Any]:
        raise NotImplementedError(
            'from_rgb not implemented for this pixel format.'
        )


@dataclass
class RGB24(RGBFormat):
    def __post_init__(self) -> None:
        self.stride = max(self.stride, self.width * 3)

    @property
    def bits_per_pixel(self) -> int:
        return 24

    @property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride * 3

    def data2key(self, value: NDArray[np.uint8]) -> NDArray[np.float32]:
        return np.astype(self.to_rgb(value) / 255.0, np.float32)

    def key2data(self, value: NDArray[np.float32]) -> NDArray[np.uint8]:
        return self.from_rgb(np.astype(value * 255.0, np.uint8))

    def to_rgb(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.reshape((self.height, -1, 3))
        value = value[: self.height, : self.width, :]
        return value

    def from_rgb(self, value: NDArray[Any]) -> NDArray[Any]:
        value = value.reshape((self.height, self.width, 3))
        value = np.pad(
            value,
            ((0, self.stride // 3 - self.width), (0, 0), (0, 0)),
            constant_values=0,
        )
        return value


@dataclass
class BGR24(RGBFormat):
    def __post_init__(self) -> None:
        self.stride = max(self.stride, self.width * 3)

    @property
    def bits_per_pixel(self) -> int:
        return 24

    @property
    def bytes_per_frame(self) -> int:
        return self.height * self.stride * 3

    def data2key(self, value: NDArray[np.uint8]) -> NDArray[np.float32]:
        return np.astype(self.to_rgb(value) / 255.0, np.float32)

    def key2data(self, value: NDArray[np.float32]) -> NDArray[np.uint8]:
        return self.from_rgb(np.astype(value * 255.0, np.uint8))

    def to_rgb(self, value: NDArray[np.uint8]) -> NDArray[np.uint8]:
        value = value.reshape((self.height, -1, 3))
        value = value[: self.height, : self.width, ::-1]
        return value

    def from_rgb(self, value: NDArray[np.uint8]) -> NDArray[np.uint8]:
        value = value.reshape((self.height, self.width, 3))
        value = np.pad(
            value,
            ((0, self.stride // 3 - self.width), (0, 0), (0, 0)),
            constant_values=0,
        )[: self.height, : self.stride, ::-1]
        return value


@dataclass
class XYZFormat(PixelFormat):
    pass
