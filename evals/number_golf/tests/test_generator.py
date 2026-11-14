from generator import STARTER_CONFIGS, difficulty_rank, OP_TIERS


def test_starter_configs_nonempty_and_named():
    assert len(STARTER_CONFIGS) >= 6
    names = [c.name for c in STARTER_CONFIGS]
    assert len(names) == len(set(names))  # unique names


def test_op_tiers_increasing_richness():
    assert OP_TIERS["arith"] < OP_TIERS["arith_concat"]
    assert OP_TIERS["arith_concat"] < OP_TIERS["full"]


def test_difficulty_rank_arith_harder_than_full():
    arith = [c for c in STARTER_CONFIGS if "arith" == c.name.split("-")[0]][0]
    full = [c for c in STARTER_CONFIGS if "full" == c.name.split("-")[0]][0]
    # fewer ops == harder == larger difficulty rank
    assert difficulty_rank(arith) > difficulty_rank(full)
