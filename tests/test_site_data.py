import json

import pandas as pd

from common.config import write_parquet
from figures import site_data


def test_blocks_stay_pending_without_results(tmp_path):
    out = tmp_path / "results.json"
    es = tmp_path / "es.yaml"
    es.write_text(f"output_dir: {tmp_path / 'none'}\nfigure:\n  output: {tmp_path / 'fig.pdf'}\n")
    irt = tmp_path / "irt.yaml"
    irt.write_text(f"output_dir: {tmp_path / 'none'}\nfigure:\n  output: {tmp_path / 'fig2.pdf'}\n")
    site_data.main(["--event-study-config", str(es), "--irt-config", str(irt), "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["status"] == "pending"
    assert all(b["status"] == "pending" for b in data["blocks"].values())
    assert len(data["design"]["checkpoints"]) == 30
    assert data["design"]["timing_filter"]["min_checkpoints_before_t0"] == 3


def test_event_study_block_summarises_and_flags_pretrend(tmp_path):
    eff = pd.DataFrame({"k": [-1, 0, 1, 0, 1], "effect": [0.0, 1.0, 2.0, 0.5, 0.5],
                        "ci_low": 0.0, "ci_high": 1.0, "n_items": 10,
                        "model": ["a", "a", "a", "b", "b"], "stratum": "all"})
    pre = pd.DataFrame({"model": ["a", "b"], "stratum": "all", "n_items": 10, "pretrend": [0.0, -0.4],
                        "pretrend_ci_low": [-0.1, -0.6], "pretrend_ci_high": [0.1, -0.2],
                        "pretrend_flag": [False, True]})
    write_parquet(eff, tmp_path / "effects.parquet", {"git_commit": "x"})
    write_parquet(pre, tmp_path / "pretrend.parquet", {"git_commit": "x"})
    block = site_data.event_study_block({"output_dir": str(tmp_path), "figure": {"output": str(tmp_path / "f.pdf")}})
    assert block["status"] == "available"
    assert "a: +1.50" in block["summary"]
    assert "b: +0.50 (pre-trend: not causal)" in block["summary"]
    assert block["figure"] is None
