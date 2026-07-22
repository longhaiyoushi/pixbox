import numpy as np
import pytest
from colour.models.rgb.ycbcr import (
    offset_YCbCr,
    ranges_YCbCr,
)

from pixbox.core.color_range import (
    ColorRange,
    FullRange,
    LimitedRange,
)


class TestColorRange:
    @pytest.mark.parametrize(
        ('range_cls', 'is_legal'),
        [(FullRange, False), (LimitedRange, True)],
    )
    @pytest.mark.parametrize(
        'bits',
        [8, 10],
    )
    def test_range_parameters(
        self, range_cls: type[ColorRange], is_legal: bool, bits: int
    ) -> None:
        color_range = range_cls(bits)
        y_min, y_max, c_min, c_max = ranges_YCbCr(bits, is_legal, False)
        y_off, c_off, c_off = offset_YCbCr(bits, is_legal, False)

        actual = color_range.min
        desired = np.array([y_min, c_min, c_min])
        if not is_legal:
            desired[1:] += np.float32(0.5)
        np.testing.assert_array_almost_equal(
            actual,
            desired,
            decimal=6,
        )

        actual = color_range.max
        desired = np.array([y_max, c_max, c_max])
        if not is_legal:
            desired[1:] += np.float32(0.5)
        np.testing.assert_array_almost_equal(
            actual,
            desired,
            decimal=6,
        )

        actual = color_range.offset
        desired = np.array([y_off, c_off, c_off])
        if not is_legal:
            desired[1:] += np.float32(0.5)
        np.testing.assert_array_almost_equal(
            actual,
            desired,
            decimal=6,
        )
