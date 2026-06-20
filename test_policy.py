import main


def _obs(width=20, height=21, south=0, step=0):
    return {
        "southBound": south,
        "northBound": south + height - 1,
        "step": step,
        "walls": [0] * (width * height),
        "robots": {},
        "crystals": {},
        "miningNodes": {},
    }


def _cfg(width=20, height=21):
    return {
        "width": width,
        "height": height,
        "factoryMovePeriod": 2,
        "factoryJumpCooldown": 20,
        "scrollRampSteps": 450,
        "scrollStartInterval": 10,
        "scrollEndInterval": 2,
        "scoutCost": 50,
        "workerCost": 200,
        "minerCost": 300,
        "wallRemoveCost": 100,
        "factoryUpkeep": 2,
    }


def _set_wall(obs, config, col, row, direction):
    idx = (row - obs["southBound"]) * config["width"] + col
    obs["walls"][idx] |= main.WALL_BIT[direction]


def _mark_unknown(obs, config, col, row):
    idx = (row - obs["southBound"]) * config["width"] + col
    midx = (row - obs["southBound"]) * config["width"] + (config["width"] - 1 - col)
    obs["walls"][idx] = -1
    obs["walls"][midx] = -1


def test_factory_keeps_route_without_material_gain():
    obs, config = _obs(), _cfg()
    factory = {"col": 5, "row": 10, "owner": 0}
    start = (5, 10)
    old_goal = (5, 12)
    new_goal = (6, 13)
    dist = {
        start: 0, (5, 11): 1, old_goal: 2, (6, 11): 2,
        (6, 12): 3, (7, 12): 4, new_goal: 5
    }
    parent = {
        start: None,
        (5, 11): start,
        old_goal: (5, 11),
        (6, 11): (5, 11),
        (6, 12): (6, 11),
        (7, 12): (6, 12),
        new_goal: (7, 12),
    }
    assert main.choose_factory_target(
        obs, config, factory, dist, parent, old_goal
    ) == old_goal


def test_factory_switches_to_materially_better_route():
    obs, config = _obs(), _cfg()
    factory = {"col": 5, "row": 10, "owner": 0}
    start = (5, 10)
    old_goal = (5, 12)
    new_goal = (5, 15)
    dist = {start: 0, (5, 11): 1, old_goal: 2, (5, 13): 3, (5, 14): 4, new_goal: 5}
    parent = {
        start: None,
        (5, 11): start,
        old_goal: (5, 11),
        (5, 13): old_goal,
        (5, 14): (5, 13),
        new_goal: (5, 14),
    }
    assert main.choose_factory_target(
        obs, config, factory, dist, parent, old_goal
    ) == new_goal


def test_factory_avoids_spawning_into_corridor():
    obs, config = _obs(), _cfg()
    factory = {"col": 5, "row": 10}
    occupied = {(5, 10)}
    corridor = {(5, 11), (6, 10)}
    assert main._build_direction(obs, config, factory, occupied, corridor) == "WEST"


def test_scout_frontier_biases_toward_corridor():
    obs, config = _obs(), _cfg()
    start = (5, 5)
    factory = {"col": 5, "row": 5}
    near = (5, 8)
    far = (1, 8)
    for cell in (near, far):
        c, r = cell
        _mark_unknown(obs, config, c, r)
        _mark_unknown(obs, config, c, r + 1)
    dist = {start: 0, near: 3, far: 3}
    assert main.choose_frontier_target(
        obs, config, start, 1, dist, factory, {(5, 7)}
    ) == near


def test_worker_builds_when_route_is_blocked_before_boxed():
    obs, config = _obs(), _cfg()
    _set_wall(obs, config, 5, 10, "NORTH")
    factory = {
        "uid": "f",
        "type": main.FACTORY,
        "col": 5,
        "row": 10,
        "energy": 700,
        "owner": 0,
        "move_cd": 2,
        "jump_cd": 5,
        "build_cd": 0,
    }
    mine = {
        "f": factory,
        "s1": {"type": main.SCOUT, "col": 1, "row": 10},
        "s2": {"type": main.SCOUT, "col": 2, "row": 10},
    }
    occupied = {(5, 10), (1, 10), (2, 10)}
    assert main._factory_action(obs, config, factory, mine, occupied).startswith("BUILD_WORKER_")


def test_jump_uses_actual_route_distance_pressure():
    obs, config = _obs(step=500), _cfg()
    factory = {"col": 5, "row": 2}
    assert main._factory_should_jump(
        obs, config, factory, has_north_target=True, boxed=False, distance_to_progress=10
    )


def test_jump_is_saved_when_route_exists_and_margin_is_healthy():
    obs, config = _obs(step=500), _cfg()
    factory = {"col": 5, "row": 8}
    assert not main._factory_should_jump(
        obs, config, factory, has_north_target=True, boxed=False, distance_to_progress=10
    )


def test_escape_worker_can_spend_below_normal_reserve():
    obs, config = _obs(), _cfg()
    _set_wall(obs, config, 5, 10, "NORTH")
    factory = {
        "uid": "f",
        "type": main.FACTORY,
        "col": 5,
        "row": 10,
        "energy": 260,
        "owner": 0,
        "move_cd": 2,
        "jump_cd": 5,
        "build_cd": 0,
    }
    mine = {"f": factory, "s1": {"type": main.SCOUT}, "s2": {"type": main.SCOUT}}
    occupied = {(5, 10)}
    assert main._factory_action(obs, config, factory, mine, occupied).startswith("BUILD_WORKER_")


def test_refuel_uses_projected_factory_energy():
    obs, config = _obs(), _cfg()
    factory = {"col": 5, "row": 10, "energy": main.FACTORY_REFUEL_BELOW + 5}
    scout = {"col": 5, "row": 9, "energy": 50}
    assert main._refuel_factory_action(obs, config, scout, factory) == "TRANSFER_NORTH"


def run_all():
    test_factory_keeps_route_without_material_gain()
    test_factory_switches_to_materially_better_route()
    test_factory_avoids_spawning_into_corridor()
    test_scout_frontier_biases_toward_corridor()
    test_worker_builds_when_route_is_blocked_before_boxed()
    test_jump_uses_actual_route_distance_pressure()
    test_jump_is_saved_when_route_exists_and_margin_is_healthy()
    test_escape_worker_can_spend_below_normal_reserve()
    test_refuel_uses_projected_factory_energy()


if __name__ == "__main__":
    run_all()
    print("policy tests passed")
