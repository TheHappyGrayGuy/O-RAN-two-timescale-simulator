import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oran_sim import Config, Network  # noqa: E402


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def net(cfg):
    return Network(cfg)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
