from typing import Type

import ffmpeg
import numpy as np
import pytest

from pixbox.core.pixel_format import (
    BGR24,
    RGB24,
    YUV420,
    YUV420_I420,
    YUV420_NV12,
    YUV420_NV21,
    YUV422,
    YUV422_I422,
    YUV422_NV16,
    YUV444_I444,
    YUV444_NV24,
    YUV444_NV42,
    RGBFormat,
)


class TestPixelFormat:
    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('pixel_format_cls', 'from_format', 'bits', 'dtype'),
        [
            (YUV420_I420, 'yuv420p', 8, np.uint8),
            (YUV420_I420, 'yuv420p9le', 9, np.uint16),
            (YUV420_I420, 'yuv420p10le', 10, np.uint16),
            (YUV420_I420, 'yuv420p12le', 12, np.uint16),
            (YUV420_I420, 'yuv420p14le', 14, np.uint16),
            (YUV420_I420, 'yuv420p16le', 16, np.uint16),
            (YUV420_NV12, 'nv12', 8, np.uint8),
            (YUV420_NV12, 'p010le', 16, np.uint16),
            (YUV420_NV12, 'p012le', 16, np.uint16),
            (YUV420_NV12, 'p016le', 16, np.uint16),
            (YUV420_NV21, 'nv21', 8, np.uint8),
        ],
    )
    def test_yuv420_to_yuv(
        self,
        seed: int,
        pixel_format_cls: Type[YUV420],
        from_format: str,
        bits: int,
        dtype: Type[np.uint8 | np.uint16],
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        value = rng.integers(256, size=(6, 4), dtype=dtype) << (bits - 8)

        pixel_format = pixel_format_cls(height=4, width=4, bits=bits)
        actual = pixel_format.to_yuv(value)

        out, _ = (
            ffmpeg.input('pipe:', f='rawvideo', pix_fmt=from_format, s='4x4')
            .output('pipe:', f='rawvideo', pix_fmt='yuv444p', vframes=1)
            .run(input=value.tobytes(), capture_stdout=True)
        )

        desired = (
            np.frombuffer(out, dtype=np.uint8)
            .reshape((3, 4, 4))
            .transpose((1, 2, 0))
        )
        desired = desired.astype(dtype) << (bits - 8)  # type: ignore[operator]

        np.testing.assert_array_almost_equal(
            actual,
            desired,
            decimal=3,
        )

    @pytest.mark.parametrize(
        'seed',
        range(5)[:1],
    )
    @pytest.mark.parametrize(
        ('pixel_format_cls', 'to_format', 'bits', 'dtype'),
        [
            (YUV420_I420, 'yuv420p', 8, np.uint8),
            (YUV420_I420, 'yuv420p9le', 9, np.uint16),
            (YUV420_I420, 'yuv420p10le', 10, np.uint16),
            (YUV420_I420, 'yuv420p12le', 12, np.uint16),
            (YUV420_I420, 'yuv420p14le', 14, np.uint16),
            (YUV420_I420, 'yuv420p16le', 16, np.uint16),
            (YUV420_NV12, 'nv12', 8, np.uint8),
            (YUV420_NV12, 'p010le', 16, np.uint16),
            (YUV420_NV12, 'p012le', 16, np.uint16),
            (YUV420_NV12, 'p016le', 16, np.uint16),
            (YUV420_NV21, 'nv21', 8, np.uint8),
        ],
    )
    def test_yuv420_from_yuv(
        self,
        seed: int,
        pixel_format_cls: Type[YUV420],
        to_format: str,
        bits: int,
        dtype: Type[np.uint8 | np.uint16],
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        value = rng.integers(256, size=(4, 4, 3), dtype=np.uint8)

        pixel_format = pixel_format_cls(height=4, width=4, bits=bits)
        actual = pixel_format.from_yuv(value.astype(dtype) << (bits - 8))  # type: ignore[operator]

        out, _ = (
            ffmpeg.input('pipe:', f='rawvideo', pix_fmt='yuv444p', s='4x4')
            .output('pipe:', f='rawvideo', pix_fmt=to_format, vframes=1)
            .run(
                input=value.transpose((2, 0, 1)).tobytes(), capture_stdout=True
            )
        )

        desired = np.frombuffer(out, dtype=dtype).reshape((6, 4))

        np.testing.assert_array_almost_equal(
            actual,
            desired,  # type: ignore[arg-type]
            decimal=3,
        )

    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('pixel_format_cls', 'from_format', 'bits', 'dtype'),
        [
            (YUV422_I422, 'yuv422p', 8, np.uint8),
            (YUV422_I422, 'yuv422p9le', 9, np.uint16),
            (YUV422_I422, 'yuv422p10le', 10, np.uint16),
            (YUV422_I422, 'yuv422p12le', 12, np.uint16),
            (YUV422_I422, 'yuv422p14le', 14, np.uint16),
            (YUV422_I422, 'yuv422p16le', 16, np.uint16),
            (YUV422_NV16, 'nv16', 8, np.uint8),
            (YUV422_NV16, 'nv20le', 10, np.uint16),
            (YUV422_NV16, 'p210le', 16, np.uint16),
            (YUV422_NV16, 'p212le', 16, np.uint16),
            (YUV422_NV16, 'p216le', 16, np.uint16),
        ],
    )
    def test_yuv422_to_yuv(
        self,
        seed: int,
        pixel_format_cls: Type[YUV422],
        from_format: str,
        bits: int,
        dtype: Type[np.uint8 | np.uint16],
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        value = rng.integers(256, size=(8, 4), dtype=dtype) << (bits - 8)

        pixel_format = pixel_format_cls(height=4, width=4, bits=bits)
        actual = pixel_format.to_yuv(value)

        out, _ = (
            ffmpeg.input('pipe:', f='rawvideo', pix_fmt=from_format, s='4x4')
            .output('pipe:', f='rawvideo', pix_fmt='yuv444p', vframes=1)
            .run(input=value.tobytes(), capture_stdout=True)
        )

        desired = (
            np.frombuffer(out, dtype=np.uint8)
            .reshape((3, 4, 4))
            .transpose((1, 2, 0))
        )
        desired = desired.astype(dtype) << (bits - 8)  # type: ignore[operator]

        np.testing.assert_array_almost_equal(
            actual,
            desired,
            decimal=3,
        )

    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('pixel_format_cls', 'to_format', 'bits', 'dtype'),
        [
            (YUV422_I422, 'yuv422p', 8, np.uint8),
            (YUV422_I422, 'yuv422p9le', 9, np.uint16),
            (YUV422_I422, 'yuv422p10le', 10, np.uint16),
            (YUV422_I422, 'yuv422p12le', 12, np.uint16),
            (YUV422_I422, 'yuv422p14le', 14, np.uint16),
            (YUV422_I422, 'yuv422p16le', 16, np.uint16),
            (YUV422_NV16, 'nv16', 8, np.uint8),
            (YUV422_NV16, 'nv20le', 10, np.uint16),
            (YUV422_NV16, 'p210le', 16, np.uint16),
            (YUV422_NV16, 'p212le', 16, np.uint16),
            (YUV422_NV16, 'p216le', 16, np.uint16),
        ],
    )
    def test_yuv422_from_yuv(
        self,
        seed: int,
        pixel_format_cls: Type[YUV420],
        to_format: str,
        bits: int,
        dtype: Type[np.uint8 | np.uint16],
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        value = rng.integers(256, size=(4, 4, 3), dtype=np.uint8)

        pixel_format = pixel_format_cls(height=4, width=4, bits=bits)
        actual = pixel_format.from_yuv(value.astype(dtype) << (bits - 8))  # type: ignore[operator]

        out, _ = (
            ffmpeg.input('pipe:', f='rawvideo', pix_fmt='yuv444p', s='4x4')
            .output('pipe:', f='rawvideo', pix_fmt=to_format, vframes=1)
            .run(
                input=value.transpose((2, 0, 1)).tobytes(), capture_stdout=True
            )
        )

        desired = np.frombuffer(out, dtype=dtype).reshape((8, 4))

        np.testing.assert_array_almost_equal(
            actual,
            desired,  # type: ignore[arg-type]
            decimal=3,
        )

    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('pixel_format_cls', 'from_format', 'bits', 'dtype'),
        [
            (YUV444_I444, 'yuv444p', 8, np.uint8),
            (YUV444_I444, 'yuv444p9le', 9, np.uint16),
            (YUV444_I444, 'yuv444p10le', 10, np.uint16),
            (YUV444_I444, 'yuv444p12le', 12, np.uint16),
            (YUV444_I444, 'yuv444p14le', 14, np.uint16),
            (YUV444_I444, 'yuv444p16le', 16, np.uint16),
            (YUV444_NV24, 'nv24', 8, np.uint8),
            (YUV444_NV24, 'p410le', 16, np.uint16),
            (YUV444_NV24, 'p412le', 16, np.uint16),
            (YUV444_NV24, 'p416le', 16, np.uint16),
            (YUV444_NV42, 'nv42', 8, np.uint8),
        ],
    )
    def test_yuv444_to_yuv(
        self,
        seed: int,
        pixel_format_cls: Type[YUV422],
        from_format: str,
        bits: int,
        dtype: Type[np.uint8 | np.uint16],
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        value = rng.integers(256, size=(12, 4), dtype=dtype) << (bits - 8)

        pixel_format = pixel_format_cls(height=4, width=4, bits=bits)
        actual = pixel_format.to_yuv(value)

        out, _ = (
            ffmpeg.input('pipe:', f='rawvideo', pix_fmt=from_format, s='4x4')
            .output('pipe:', f='rawvideo', pix_fmt='yuv444p', vframes=1)
            .run(input=value.tobytes(), capture_stdout=True)
        )

        desired = (
            np.frombuffer(out, dtype=np.uint8)
            .reshape((3, 4, 4))
            .transpose((1, 2, 0))
        )
        desired = desired.astype(dtype) << (bits - 8)  # type: ignore[operator]

        np.testing.assert_array_almost_equal(
            actual,
            desired,
            decimal=3,
        )

    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('pixel_format_cls', 'to_format', 'bits', 'dtype'),
        [
            (YUV444_I444, 'yuv444p', 8, np.uint8),
            (YUV444_I444, 'yuv444p9le', 9, np.uint16),
            (YUV444_I444, 'yuv444p10le', 10, np.uint16),
            (YUV444_I444, 'yuv444p12le', 12, np.uint16),
            (YUV444_I444, 'yuv444p14le', 14, np.uint16),
            (YUV444_I444, 'yuv444p16le', 16, np.uint16),
            (YUV444_NV24, 'nv24', 8, np.uint8),
            (YUV444_NV24, 'p410le', 16, np.uint16),
            (YUV444_NV24, 'p412le', 16, np.uint16),
            (YUV444_NV24, 'p416le', 16, np.uint16),
            (YUV444_NV42, 'nv42', 8, np.uint8),
        ],
    )
    def test_yuv444_from_yuv(
        self,
        seed: int,
        pixel_format_cls: Type[YUV420],
        to_format: str,
        bits: int,
        dtype: Type[np.uint8 | np.uint16],
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        value = rng.integers(256, size=(4, 4, 3), dtype=np.uint8)

        pixel_format = pixel_format_cls(height=4, width=4, bits=bits)
        actual = pixel_format.from_yuv(value.astype(dtype) << (bits - 8))  # type: ignore[operator]
        actual = actual.reshape((12, 4))

        out, _ = (
            ffmpeg.input('pipe:', f='rawvideo', pix_fmt='yuv444p', s='4x4')
            .output('pipe:', f='rawvideo', pix_fmt=to_format, vframes=1)
            .run(
                input=value.transpose((2, 0, 1)).tobytes(), capture_stdout=True
            )
        )

        desired = np.frombuffer(out, dtype=dtype).reshape((12, 4))

        np.testing.assert_array_almost_equal(
            actual,
            desired,  # type: ignore[arg-type]
            decimal=3,
        )

    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('pixel_format_cls', 'from_format'),
        [
            (RGB24, 'rgb24'),
            (BGR24, 'bgr24'),
        ],
    )
    def test_rgb24_to_rgb(
        self,
        seed: int,
        pixel_format_cls: Type[RGBFormat],
        from_format: str,
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        value = rng.integers(256, size=(4, 4, 3), dtype=np.uint8)

        pixel_format = pixel_format_cls(height=4, width=4)
        actual = pixel_format.to_rgb(value)

        out, _ = (
            ffmpeg.input('pipe:', f='rawvideo', pix_fmt=from_format, s='4x4')
            .output('pipe:', f='rawvideo', pix_fmt='rgb24', vframes=1)
            .run(input=value.tobytes(), capture_stdout=True)
        )

        desired = np.frombuffer(out, dtype=np.uint8).reshape((4, 4, 3))

        np.testing.assert_array_almost_equal(
            actual,
            desired,
            decimal=3,
        )

    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('pixel_format_cls', 'to_format'),
        [
            (RGB24, 'rgb24'),
            (BGR24, 'bgr24'),
        ],
    )
    def test_rgb24_from_rgb(
        self,
        seed: int,
        pixel_format_cls: Type[RGBFormat],
        to_format: str,
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        value = rng.integers(256, size=(4, 4, 3), dtype=np.uint8)

        pixel_format = pixel_format_cls(height=4, width=4)
        actual = pixel_format.from_rgb(value)

        out, _ = (
            ffmpeg.input('pipe:', f='rawvideo', pix_fmt='rgb24', s='4x4')
            .output('pipe:', f='rawvideo', pix_fmt=to_format, vframes=1)
            .run(input=value.tobytes(), capture_stdout=True)
        )

        desired = np.frombuffer(out, dtype=np.uint8).reshape((4, 4, 3))

        np.testing.assert_array_almost_equal(
            actual,
            desired,
            decimal=3,
        )
