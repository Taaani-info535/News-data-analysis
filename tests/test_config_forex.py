import itertools
from uamis.config import Config


def mk(base_cfg, tmp_path, text, expected=28):
    f = tmp_path / "fx.txt"
    if text is not None:
        f.write_text(text)
    raw = dict(base_cfg.raw); raw["forex"] = dict(raw["forex"], pairs_file=str(f), expected_count=expected)
    return Config(raw)


PAIRS = ["".join(p) for p in itertools.combinations(["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"], 2)]


def test_missing_file_disables(base_cfg, tmp_path):
    p, msg = mk(base_cfg, tmp_path, None).forex_status()
    assert p == [] and "DISABLED" in msg


USER_28 = ("EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD EURJPY GBPJPY EURGBP EURCHF EURCAD GBPCAD GBPCHF GBPAUD GBPNZD "
           "AUDJPY AUDCAD AUDNZD AUDCHF NZDJPY NZDCAD NZDCHF CADJPY CADCHF CHFJPY EURAUD EURNZD").split()


def test_shipped_list_is_exactly_the_users_28_pairs_in_order(base_cfg):
    p, msg = base_cfg.forex_status()
    assert p == USER_28 and len(set(p)) == 28 and "OK" in msg
    a = base_cfg.all_assets()
    assert all(x in a for x in USER_28) and a["USDJPY"]["base"] == "USD" and a["USDJPY"]["quote"] == "JPY" and a["USDJPY"]["symbol"] == "USDJPY=X"


def test_wrong_count_disables_and_does_not_guess(base_cfg, tmp_path):
    p, msg = mk(base_cfg, tmp_path, "\n".join(PAIRS[:27])).forex_status()
    assert p == [] and "27" in msg


def test_invalid_and_duplicate(base_cfg, tmp_path):
    assert mk(base_cfg, tmp_path, "EURUSD\nBAD1\n").forex_status()[0] == []
    assert mk(base_cfg, tmp_path, "EURUSD\nEURUSD\n", expected=2).forex_status()[0] == []


def test_exact_list_reused_verbatim_and_order_kept(base_cfg, tmp_path):
    txt = "# my scanner list\n" + "\n".join(p[:3] + "/" + p[3:] if i % 2 else p for i, p in enumerate(PAIRS))
    cfg = mk(base_cfg, tmp_path, txt)
    pairs, msg = cfg.forex_status()
    assert pairs == PAIRS and "OK" in msg
    assets = cfg.all_assets()
    assert all(p in assets for p in PAIRS) and assets[PAIRS[0]]["base"] == PAIRS[0][:3] and assets[PAIRS[0]]["quote"] == PAIRS[0][3:]
