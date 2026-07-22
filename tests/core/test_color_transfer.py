from collections.abc import Callable

import numpy as np
import pytest
from colour.hints import (
    Domain1,
    Range1,
)
from colour.models import (
    eotf_inverse_sRGB,
    eotf_sRGB,
    oetf_BT709,
    oetf_BT2020,
    oetf_inverse_BT709,
    oetf_inverse_BT2020,
)

from pixbox.core.color_transfer import (
    BT709Transfer,
    BT2020Transfer,
    ColorTransfer,
    SrgbTransfer,
)


class TestColorTransfer:
    @pytest.mark.parametrize(
        'seed',
        range(5),
    )
    @pytest.mark.parametrize(
        ('transfer_cls', 'colour_decode'),
        [
            (SrgbTransfer, eotf_sRGB),
            (BT709Transfer, oetf_inverse_BT709),
            (BT2020Transfer, oetf_inverse_BT2020),
        ],
    )
    def test_lin2rgb(
        self,
        seed: int,
        transfer_cls: type[ColorTransfer],
        colour_decode: Callable[[Domain1], Range1],
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        values = rng.random(3, np.float32)

        transfer = transfer_cls()

        actual = transfer.lin2rgb(values)
        desired = colour_decode(values)
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
        ('transfer_cls', 'colour_encode'),
        [
            (SrgbTransfer, eotf_inverse_sRGB),
            (BT709Transfer, oetf_BT709),
            (BT2020Transfer, oetf_BT2020),
        ],
    )
    def test_rgb2lin(
        self,
        seed: int,
        transfer_cls: type[ColorTransfer],
        colour_encode: Callable[[Domain1], Range1],
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        values = rng.random(3, np.float32)

        transfer = transfer_cls()

        actual = transfer.rgb2lin(values)
        desired = colour_encode(values)
        np.testing.assert_array_almost_equal(
            actual,
            desired,
            decimal=3,
        )
