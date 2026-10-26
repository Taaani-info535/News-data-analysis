import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from uamis.config import load_config  # noqa: E402
from uamis.demo import make_demo_config, synth_events, synth_market  # noqa: E402
from uamis.storage import Store  # noqa: E402

END = 1_800_000_000 - (1_800_000_000 % 86400) + 10 * 3600   # a Friday 10:00 UTC (synthetic clock)
NOW = (END // 300) * 300 + 60


@pytest.fixture
def base_cfg():
    return load_config(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))


@pytest.fixture
def small_world(tmp_path, base_cfg):
    """SYNTHETIC world: 4 non-forex assets, forex disabled (empty pairs file), 200 days. Test fixture only."""
    cfg = make_demo_config(str(tmp_path), pairs=[], base_cfg=base_cfg)
    store = Store(cfg.db_path)
    evs = synth_events(END - 86400, days=200)
    for e in evs:
        store.upsert_event(e)
    synth_market(cfg, store, END, days=200, events=evs)
    yield cfg, store, evs
    store.close()
