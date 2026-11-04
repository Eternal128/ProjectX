from common.config import config_hash, load_config


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_all_configs_load():
    for name in ["models", "benchmarks", "exposure", "evals", "irt", "event_study"]:
        assert isinstance(load_config(f"configs/{name}.yaml"), dict)


def test_checkpoint_list_is_sorted_and_about_30():
    ck = load_config("configs/models.yaml")["checkpoints"]
    assert ck == sorted(set(ck))
    assert 25 <= len(ck) <= 35
    assert ck[-1] == load_config("configs/models.yaml")["training"]["total_steps"]
