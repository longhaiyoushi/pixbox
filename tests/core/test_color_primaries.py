from typing import Type

import numpy as np
import pytest
from colour.models import (
    RGB_COLOURSPACE_BT709,
    RGB_COLOURSPACE_BT2020,
    RGB_Colourspace,
    RGB_COLOURSPACE_sRGB,
)
from colour.models.rgb.ycbcr import (
    WEIGHTS_YCBCR,
    RGB_to_YCbCr,
    YCbCr_to_RGB,
    matrix_YCbCr,
)

from pixbox.core.color_primaries import (
    BT709Primaries,
    BT2020Primaries,
    ColorPrimaries,
    SrgbPrimaries,
)
from pixbox.core.color_range import (
    ColorRange,
    FullRange,
    LimitedRange,
)


class TestColorPrimaries:
    @pytest.mark.parametrize(
        ('primaries_cls', 'colour_space'),
        [
            (SrgbPrimaries, RGB_COLOURSPACE_sRGB),
            (BT709Primaries, RGB_COLOURSPACE_BT709),
            (BT2020Primaries, RGB_COLOURSPACE_BT2020),
        ],
    )
    def test_rgb_xyz_matrix(
        self, primaries_cls: Type[ColorPrimaries], colour_space: RGB_Colourspace
    ) -> None:
        color_primaries = primaries_cls()
        np.testing.assert_array_almost_equal(
            color_primaries.matrix_rgb2xyz,
            colour_space.matrix_RGB_to_XYZ,
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            color_primaries.matrix_xyz2rgb,
            colour_space.matrix_XYZ_to_RGB,
            decimal=3,
        )

    @pytest.mark.parametrize(
        ('primaries_cls', 'weights_name'),
        [
            (SrgbPrimaries, 'ITU-R BT.709'),
            (BT709Primaries, 'ITU-R BT.709'),
            (BT2020Primaries, 'ITU-R BT.2020'),
        ],
    )
    def test_rgb_yuv_matrix(
        self, primaries_cls: Type[ColorPrimaries], weights_name: str
    ) -> None:
        np.testing.assert_array_almost_equal(
            primaries_cls().matrix_yuv2rgb,
            matrix_YCbCr(K=WEIGHTS_YCBCR[weights_name]),
            decimal=3,
        )

    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('primaries_cls', 'weights_name'),
        [
            (SrgbPrimaries, 'ITU-R BT.709'),
            (BT709Primaries, 'ITU-R BT.709'),
            (BT2020Primaries, 'ITU-R BT.2020'),
        ],
    )
    @pytest.mark.parametrize(
        ('range_cls', 'is_legal'),
        [
            (FullRange, False),
            (LimitedRange, True),
        ],
    )
    @pytest.mark.parametrize(
        'bits',
        [8, 10],
    )
    def test_rgb2yuv(
        self,
        seed: int,
        primaries_cls: Type[ColorPrimaries],
        weights_name: str,
        range_cls: Type[ColorRange],
        is_legal: bool,
        bits: int,
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        values = rng.random(3, np.float32)

        actual = primaries_cls().rgb2yuv(values, range_cls(bits))
        desired = RGB_to_YCbCr(
            values,
            K=WEIGHTS_YCBCR[weights_name],
            in_bits=8,
            in_legal=False,
            out_bits=bits,
            out_legal=is_legal,
        )
        if not is_legal:
            desired[1:] += np.float32(0.5)  # type: ignore[arg-type]

        np.testing.assert_array_almost_equal(actual, desired, decimal=3)

    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('primaries_cls', 'weights_name'),
        [
            (SrgbPrimaries, 'ITU-R BT.709'),
            (BT709Primaries, 'ITU-R BT.709'),
            (BT2020Primaries, 'ITU-R BT.2020'),
        ],
    )
    @pytest.mark.parametrize(
        ('range_cls', 'is_legal'),
        [
            (FullRange, False),
            (LimitedRange, True),
        ],
    )
    @pytest.mark.parametrize(
        'bits',
        [8, 10],
    )
    def test_yuv2rgb(
        self,
        seed: int,
        primaries_cls: Type[ColorPrimaries],
        weights_name: str,
        range_cls: Type[ColorRange],
        is_legal: bool,
        bits: int,
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        values = rng.random(3, np.float32)

        actual = primaries_cls().yuv2rgb(values, range_cls(bits))
        if not is_legal:
            values[1:] -= np.float32(0.5)
        desired = YCbCr_to_RGB(
            values,
            K=WEIGHTS_YCBCR[weights_name],
            in_bits=bits,
            in_legal=is_legal,
            out_bits=8,
            out_legal=False,
        )

        np.testing.assert_array_almost_equal(actual, desired, decimal=3)
