# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# CuRobo


class TestWarp:
    def test_init_warp(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import init_warp

            result = init_warp(quiet=True)
            assert result is True
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_sdf_struct(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import warp_support_sdf_struct

            result = warp_support_sdf_struct()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_kernel_key(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import warp_support_kernel_key

            result = warp_support_kernel_key()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_bvh_constructor_type(self):
        try:
            # Third Party
            import warp as wp

            # CuRobo
            from curobo._src.util.warp import warp_support_bvh_constructor_type

            result = warp_support_bvh_constructor_type()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_sdf_struct_with_old_version(self):
        try:
            # CuRobo
            from curobo._src.util.warp import warp_support_sdf_struct

            mock_wp = MagicMock()
            mock_wp.config.version = "0.9.0"
            result = warp_support_sdf_struct(mock_wp)
            assert result is False
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_kernel_key_with_old_version(self):
        try:
            # CuRobo
            from curobo._src.util.warp import warp_support_kernel_key

            mock_wp = MagicMock()
            mock_wp.config.version = "1.0.0"
            result = warp_support_kernel_key(mock_wp)
            assert result is False
        except ImportError:
            pytest.skip("Warp not installed")

    def test_warp_support_bvh_constructor_type_with_old_version(self):
        try:
            # CuRobo
            from curobo._src.util.warp import warp_support_bvh_constructor_type

            mock_wp = MagicMock()
            mock_wp.config.version = "1.5.0"
            result = warp_support_bvh_constructor_type(mock_wp)
            assert result is False
        except ImportError:
            pytest.skip("Warp not installed")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_get_warp_device_stream_reuses_wrapper(self):
        # cuTAMP fork: a fresh wp.Stream per call registers + unregisters the handle in
        # warp's unlocked global stream map; two threads doing that corrupt the heap.
        # CuRobo
        from curobo._src.util.warp import get_warp_device_stream, init_warp

        init_warp(quiet=True)
        device = torch.device("cuda:0")
        side = torch.cuda.Stream(device=device)
        with torch.cuda.stream(side):
            _, first = get_warp_device_stream(device)
            _, second = get_warp_device_stream(torch.zeros(1, device=device))
        _, default = get_warp_device_stream(device)
        assert first is second
        assert first.cuda_stream == side.cuda_stream
        assert default is not first

