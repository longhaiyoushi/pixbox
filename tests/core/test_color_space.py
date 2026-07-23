import numpy as np
import pytest
from colour.models.rgb.rgb_colourspace import (
    RGB_to_XYZ,
    XYZ_to_RGB,
)

from pixbox.core.color_space import (
    BT709ColorSpace,
    BT2020ColorSpace,
    ColorSpace,
    SrgbColorSpace,
)
from pixbox.core.color_transfer import (
    LinearTransfer,
)
from pixbox.core.pixel_format import RGBF32


class TestColorSpace:
    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('color_space_cls', 'color_space_name'),
        [
            (SrgbColorSpace, 'sRGB'),
            (BT709ColorSpace, 'ITU-R BT.709'),
            (BT2020ColorSpace, 'ITU-R BT.2020'),
        ],
    )
    @pytest.mark.parametrize(
        'apply_cctf_decoding',
        [True, False],
    )
    def test_rgb2xyz(
        self,
        seed: int,
        color_space_cls: type[ColorSpace],
        color_space_name: str,
        apply_cctf_decoding: bool,
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        values = rng.random(3, np.float32)

        color_space = color_space_cls(RGBF32(1, 1))
        if not apply_cctf_decoding:
            color_space.color_transfer = LinearTransfer()
        actual = color_space.rgb2xyz(values)
        desired = RGB_to_XYZ(
            values,
            color_space_name,
            chromatic_adaptation_transform=None,
            apply_cctf_decoding=apply_cctf_decoding,
        )

        np.testing.assert_array_almost_equal(actual, desired, decimal=2.5)

    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('color_space_cls', 'color_space_name'),
        [
            (SrgbColorSpace, 'sRGB'),
            (BT709ColorSpace, 'ITU-R BT.709'),
            (BT2020ColorSpace, 'ITU-R BT.2020'),
        ],
    )
    @pytest.mark.parametrize(
        'apply_cctf_encoding',
        [True, False],
    )
    def test_xyz2rgb(
        self,
        seed: int,
        color_space_cls: type[ColorSpace],
        color_space_name: str,
        apply_cctf_encoding: bool,
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        values = rng.random(3, np.float32)

        color_space = color_space_cls(RGBF32(1, 1))
        if not apply_cctf_encoding:
            color_space.color_transfer = LinearTransfer()
        actual = color_space.xyz2rgb(values)
        desired = XYZ_to_RGB(
            values,
            color_space_name,
            chromatic_adaptation_transform=None,
            apply_cctf_encoding=apply_cctf_encoding,
        )

        np.testing.assert_array_almost_equal(actual, desired, decimal=2.5)
