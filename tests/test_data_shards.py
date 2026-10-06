import numpy as np
import pytest
import torch
from data_loader import _load_data_shard, save_1d_token_shard


def test_token_shard_roundtrip_and_overflow_rejection(tmp_path):
    path = tmp_path / "tokens.bin"
    save_1d_token_shard(path, np.array([0, 1, 65535], dtype=np.uint16))
    torch.testing.assert_close(
        _load_data_shard(path), torch.tensor([0, 1, 65535], dtype=torch.uint16)
    )
    with pytest.raises(ValueError, match="uint16 range"):
        save_1d_token_shard(path, np.array([65536], dtype=np.uint32))
    with pytest.raises(ValueError, match="uint16 range"):
        save_1d_token_shard(path, np.array([-1], dtype=np.int32))
