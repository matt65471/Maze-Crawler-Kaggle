"""Maze Crawler ("crawl") competition bot — V1 rule-based agent with BFS pathfinding.

Strategy at a glance (see module-level explanation at the bottom):
  - Factory: BFS north/center, build Scouts/Workers/Miners on a priority order,
    JUMP_NORTH only when genuinely stuck.
  - Scout: chase the best crystal, otherwise explore the highest-information frontier.
  - Worker: shadow the Factory and clear walls that block its northward route.
  - Miner: reach a mining node and TRANSFORM into a mine when it is safely north.

All robots share a BFS planner and a per-turn reservation system so two friendly
units never aim for the same cell (friendly fire is real in this game).

The maze scrolls NORTH (row increases northward). A robot whose row drops below
`southBound` is destroyed, so "go north, stay north" is the survival baseline.
"""

from collections import deque

# --- Constants -------------------------------------------------------------

FACTORY, SCOUT, WORKER, MINER = 0, 1, 2, 3

# NORTH increases row; EAST increases col (matches the environment's DIR_OFFSETS).
DELTA = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}
WALL_BIT = {"NORTH": 1, "EAST": 2, "SOUTH": 4, "WEST": 8}
OPPOSITE = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}
DIRS = ["NORTH", "EAST", "WEST", "SOUTH"]

# Don't TRANSFORM (or generally loiter) on a cell this close to the southern edge;
# the maze can scroll out from under us.
MINER_SAFE_MARGIN = 5

# The factory has no passive income and pays 1 energy/turn upkeep plus build
# costs. Hitting 0 energy forces it to IDLE and it gets eaten by the scroll, so
# we keep a reserve before building and let nearby units refuel it.
FACTORY_RESERVE = 350          # never build below this much factory energy
FACTORY_REFUEL_BELOW = 400     # adjacent units donate energy when factory dips under this
DONOR_MIN_ENERGY = 25          # ...but only if the donor can spare it


# --- Defensive accessors ---------------------------------------------------
# Kaggle observations/configs may be dict-like or attribute objects. Read both.

def _get(obj, key, default=None):
    if obj is None:
        return default
    try:
        if isinstance(obj, dict):
            v = obj.get(key, default)
            return default if v is None else v
    except Exception:
        pass
    try:
        v = getattr(obj, key)
        return default if v is None else v
    except Exception:
        return default


def _width(config):
    return int(_get(config, "width", 20))


def _height(config):
    return int(_get(config, "height", 20))


def _south(obs):
    return int(_get(obs, "southBound", 0))


def _north(obs, config):
    return int(_get(obs, "northBound", _south(obs) + _height(config) - 1))


# --- Basic geometry / wall helpers -----------------------------------------

def move_delta(direction):
    return DELTA[direction]


def apply_move(col, row, direction):
    dc, dr = DELTA[direction]
    return col + dc, row + dr


def get_wall(obs, config, col, row):
    """Raw wall bitfield at (col,row); returns -1 if the cell is undiscovered/off-window."""
    walls = _get(obs, "walls", []) or []
    width = _width(config)
    south = _south(obs)
    if not (0 <= col < width):
        return -1
    idx = (row - south) * width + col
    if 0 <= idx < len(walls):
        return walls[idx]
    return -1


def is_known(obs, config, col, row):
    return get_wall(obs, config, col, row) >= 0


def has_wall(obs, config, col, row, direction):
    """True only if a wall is KNOWN to block this direction. Unknown cells are
    treated as open (optimistic) so we can plan into the fog and explore."""
    w = get_wall(obs, config, col, row)
    if w < 0:
        return False
    return bool(w & WALL_BIT[direction])


def in_bounds(obs, config, col, row):
    width = _width(config)
    south = _south(obs)
    north = _north(obs, config)
    return 0 <= col < width and south <= row <= north


def is_fixed_wall(col, direction, width):
    """Perimeter E/W walls and the central mirror-axis walls cannot be built/removed."""
    if direction == "WEST" and col == 0:
        return True
    if direction == "EAST" and col == width - 1:
        return True
    half = width // 2
    if direction == "EAST" and col == half - 1:
        return True
    if direction == "WEST" and col == half:
        return True
    return False


# --- Robot parsing ----------------------------------------------------------

def parse_robots(obs):
    """uid -> robot dict. Robot list layout:
    [type, col, row, energy, owner, move_cd, jump_cd, build_cd]."""
    raw = _get(obs, "robots", {}) or {}
    out = {}
    for uid, data in raw.items():
        d = list(data)
        while len(d) < 8:
            d.append(0)
        out[uid] = {
            "uid": uid,
            "type": d[0],
            "col": d[1],
            "row": d[2],
            "energy": d[3],
            "owner": d[4],
            "move_cd": d[5],
            "jump_cd": d[6],
            "build_cd": d[7],
        }
    return out


def _parse_cell(key):
    c, r = key.split(",")
    return int(c), int(r)


# Cooldowns in the observation are the post-action values; the interpreter
# decrements them BEFORE acting. So a unit can actually act this turn when its
# stored cooldown is <= 1 (it ticks to 0 before the move/build is checked).
def _can_move_now(robot):
    return robot["move_cd"] <= 1


def _can_build_now(robot):
    return robot["build_cd"] <= 1


def _can_jump_now(robot):
    return robot["move_cd"] <= 1 and robot["jump_cd"] <= 1


# --- Movement legality + neighbors -----------------------------------------

def can_move(obs, config, col, row, direction, reserved, friendly_positions, min_row=None):
    """A single step is legal if it stays in-window, isn't blocked by a known
    wall, isn't claimed/occupied by a friendly, and doesn't drop below `min_row`
    (the southern safety floor)."""
    if has_wall(obs, config, col, row, direction):
        return False
    nc, nr = apply_move(col, row, direction)
    if not in_bounds(obs, config, nc, nr):
        return False
    if min_row is not None and nr < min_row:
        return False
    if (nc, nr) in reserved:
        return False
    return True


def get_neighbors(obs, config, col, row, reserved, friendly_positions, min_row=None):
    """Passable neighbor cells for BFS. Treats unknown cells as passable, but
    avoids cells claimed by committed friendlies (`reserved`) and cells below
    `min_row` (so units never path into the lethal southern boundary)."""
    out = []
    for d in DIRS:
        if has_wall(obs, config, col, row, d):
            continue
        nc, nr = apply_move(col, row, d)
        if not in_bounds(obs, config, nc, nr):
            continue
        if min_row is not None and nr < min_row:
            continue
        if (nc, nr) in reserved:
            continue
        out.append((nc, nr))
    return out


# --- BFS --------------------------------------------------------------------

def _bfs_all(obs, config, start, reserved, min_row=None):
    """Breadth-first flood from `start`. Returns (dist, parent) over reachable cells."""
    dist = {start: 0}
    parent = {start: None}
    dq = deque([start])
    while dq:
        c, r = dq.popleft()
        for nc, nr in get_neighbors(obs, config, c, r, reserved, None, min_row):
            if (nc, nr) in dist:
                continue
            dist[(nc, nr)] = dist[(c, r)] + 1
            parent[(nc, nr)] = (c, r)
            dq.append((nc, nr))
    return dist, parent


def _reconstruct(parent, start, goal):
    if goal != start and goal not in parent:
        return None
    path = []
    cur = goal
    while cur is not None and cur != start:
        path.append(cur)
        cur = parent.get(cur)
    if cur != start:
        return None
    path.append(start)
    path.reverse()
    return path


def bfs_path(obs, config, start, goals, reserved, friendly_positions, min_row=None):
    """Shortest path from `start` to the nearest cell in `goals` (set of (col,row)).
    Ties broken toward the north (higher row). Returns a cell list or None."""
    if not goals:
        return None
    dist, parent = _bfs_all(obs, config, start, reserved, min_row)
    reachable = [g for g in goals if g in dist and g != start]
    if not reachable:
        return None
    reachable.sort(key=lambda g: (dist[g], -g[1]))
    return _reconstruct(parent, start, reachable[0])


def first_action_from_path(path):
    """Direction string for the first hop of a path, or None."""
    if not path or len(path) < 2:
        return None
    (c0, r0), (c1, r1) = path[0], path[1]
    for d, (dc, dr) in DELTA.items():
        if (c0 + dc, r0 + dr) == (c1, r1):
            return d
    return None


# --- Information / target selection ----------------------------------------

def count_unknown_visible_cells(obs, config, col, row, vision):
    """How many still-undiscovered cells sit within `vision` (Manhattan) of (col,row),
    inside the active window. Higher = more valuable to scout."""
    width = _width(config)
    south = _south(obs)
    north = _north(obs, config)
    count = 0
    for dc in range(-vision, vision + 1):
        rem = vision - abs(dc)
        for dr in range(-rem, rem + 1):
            c, r = col + dc, row + dr
            if not (0 <= c < width and south <= r <= north):
                continue
            if get_wall(obs, config, c, r) < 0:
                count += 1
    return count


def choose_factory_target(obs, config, factory):
    """A band of cells across the top few rows, near the center of our own half.
    BFS will pick the nearest reachable one, continually nudging us north."""
    width = _width(config)
    south = _south(obs)
    north = _north(obs, config)
    half = width // 2
    if factory["col"] < half:
        center = max(1, half // 2)
    else:
        center = min(width - 2, half + half // 2)
    goals = set()
    lowest = max(south, north - 3)
    for r in range(north, lowest - 1, -1):
        for c in range(max(0, center - 2), min(width, center + 3)):
            goals.add((c, r))
    return goals


def choose_crystal_target(obs, config, start, dist):
    """Best visible crystal by score = energy - 2 * path_distance."""
    crystals = _get(obs, "crystals", {}) or {}
    best, best_score = None, None
    for key, energy in crystals.items():
        try:
            cell = _parse_cell(key)
        except Exception:
            continue
        if cell not in dist:
            continue
        score = energy - 2 * dist[cell]
        if best_score is None or score > best_score:
            best, best_score = cell, score
    return best


def choose_frontier_target(obs, config, start, vision, dist):
    """Reachable cell that maximizes 5*unknown_in_vision + 2*row_progress - distance."""
    south = _south(obs)
    best, best_score = None, None
    for cell, d in dist.items():
        if cell == start:
            continue
        c, r = cell
        unknown = count_unknown_visible_cells(obs, config, c, r, vision)
        if unknown == 0:
            continue
        score = 5 * unknown + 2 * (r - south) - d
        if best_score is None or score > best_score:
            best, best_score = cell, score
    return best


def choose_mining_node_target(obs, config, start, dist):
    """Nearest reachable mining node, preferring nodes safely north of the edge."""
    nodes = _get(obs, "miningNodes", {}) or {}
    south = _south(obs)
    best, best_key = None, None
    for key in nodes:
        try:
            cell = _parse_cell(key)
        except Exception:
            continue
        if cell not in dist:
            continue
        safe = (cell[1] - south) >= MINER_SAFE_MARGIN
        # Prefer safe nodes; among equals prefer the closest.
        rank = (0 if safe else 1, dist[cell])
        if best_key is None or rank < best_key:
            best, best_key = cell, rank
    return best


# --- Reservation-aware movement helper -------------------------------------

def _move_action(obs, config, robot, goals, occupied, min_row=None):
    """Plan a BFS move toward `goals`. Returns (action, new_cell) or (None, None)."""
    if not _can_move_now(robot):
        return None, None
    start = (robot["col"], robot["row"])
    reserved = set(occupied)
    reserved.discard(start)
    path = bfs_path(obs, config, start, goals, reserved, occupied, min_row)
    if not path:
        return None, None
    direction = first_action_from_path(path)
    if direction is None:
        return None, None
    if not can_move(obs, config, robot["col"], robot["row"], direction, reserved, occupied, min_row):
        return None, None
    return direction, apply_move(robot["col"], robot["row"], direction)


def safe_fallback_move(obs, config, robot, reserved, friendly_positions, min_row=None):
    """Pick a safe step preferring NORTH, then toward center, avoiding SOUTH.
    Returns (direction, new_cell) or (None, None)."""
    if not _can_move_now(robot):
        return None, None
    width = _width(config)
    col, row = robot["col"], robot["row"]
    half = width // 2
    horiz_first = "EAST" if col < half else "WEST"
    horiz_second = "WEST" if horiz_first == "EAST" else "EAST"
    for d in ["NORTH", horiz_first, horiz_second, "SOUTH"]:
        if can_move(obs, config, col, row, d, reserved, friendly_positions, min_row):
            return d, apply_move(col, row, d)
    return None, None


# --- Per-type policies ------------------------------------------------------

def _factory_action(obs, config, factory, mine_robots, occupied):
    """Build priority, then BFS north/center, then JUMP_NORTH if stuck."""
    scouts = sum(1 for r in mine_robots.values() if r["type"] == SCOUT)
    workers = sum(1 for r in mine_robots.values() if r["type"] == WORKER)
    miners = sum(1 for r in mine_robots.values() if r["type"] == MINER)
    mining_visible = len(_get(obs, "miningNodes", {}) or {}) > 0

    energy = factory["energy"]
    scout_cost = int(_get(config, "scoutCost", 50))
    worker_cost = int(_get(config, "workerCost", 200))
    miner_cost = int(_get(config, "minerCost", 300))

    # Only build when we can do so without dropping below the survival reserve.
    if _can_build_now(factory):
        if scouts < 2 and energy - scout_cost >= FACTORY_RESERVE:
            d = _build_direction(obs, config, factory, occupied)
            if d:
                return "BUILD_SCOUT_" + d
        elif workers < 1 and energy - worker_cost >= FACTORY_RESERVE:
            d = _build_direction(obs, config, factory, occupied)
            if d:
                return "BUILD_WORKER_" + d
        elif mining_visible and miners < 1 and energy - miner_cost >= FACTORY_RESERVE:
            d = _build_direction(obs, config, factory, occupied)
            if d:
                return "BUILD_MINER_" + d

    # Otherwise advance north/center. Keep a safety buffer above the boundary so
    # the scroll can never catch the factory, but allow lateral / small detours
    # so it can navigate around walls (a hard "never decrease row" floor boxes
    # it in). The northern goal band biases overall progress upward.
    south = _south(obs)
    min_row = min(factory["row"], south + 2)
    goals = choose_factory_target(obs, config, factory)
    action, new_cell = _move_action(obs, config, factory, goals, occupied, min_row)
    if action:
        _commit(occupied, (factory["col"], factory["row"]), new_cell)
        return action

    # Stuck: jump two cells north if the landing is safely inside the window.
    if _can_jump_now(factory):
        lc, lr = factory["col"], factory["row"] + 2
        if in_bounds(obs, config, lc, lr) and (lc, lr) not in occupied:
            if (lr - _south(obs)) >= 1:
                _commit(occupied, (factory["col"], factory["row"]), (lc, lr))
                return "JUMP_NORTH"
    return "IDLE"


def _build_direction(obs, config, factory, occupied):
    """Choose a spawn direction whose neighbor is open, in-window, and unoccupied."""
    for d in ["NORTH", "EAST", "WEST", "SOUTH"]:
        if has_wall(obs, config, factory["col"], factory["row"], d):
            continue
        nc, nr = apply_move(factory["col"], factory["row"], d)
        if not in_bounds(obs, config, nc, nr):
            continue
        if (nc, nr) in occupied:
            continue
        return d
    return None


def _scout_action(obs, config, scout, occupied, factory):
    refuel = _refuel_factory_action(obs, config, scout, factory)
    if refuel:
        return refuel
    start = (scout["col"], scout["row"])
    min_row = _south(obs) + 1  # never step onto the lethal boundary row
    reserved = set(occupied)
    reserved.discard(start)
    dist, _parent = _bfs_all(obs, config, start, reserved, min_row)
    vision = int(_get(config, "visionScout", 5))

    # 1) crystals, 2) frontier exploration.
    target = choose_crystal_target(obs, config, start, dist)
    if target is None:
        target = choose_frontier_target(obs, config, start, vision, dist)

    if target is not None:
        action, new_cell = _move_action(obs, config, scout, {target}, occupied, min_row)
        if action:
            _commit(occupied, start, new_cell)
            return action

    d, new_cell = safe_fallback_move(obs, config, scout, reserved, occupied, min_row)
    if d:
        _commit(occupied, start, new_cell)
        return d
    return "IDLE"


def _worker_action(obs, config, worker, factory, occupied):
    """Shadow the factory; clear a removable wall that pens it in to the north."""
    refuel = _refuel_factory_action(obs, config, worker, factory)
    if refuel:
        return refuel
    width = _width(config)
    start = (worker["col"], worker["row"])
    min_row = _south(obs) + 1
    remove_cost = int(_get(config, "wallRemoveCost", 100))

    if factory is not None and worker["energy"] >= remove_cost:
        fc, fr = factory["col"], factory["row"]
        # Worker sitting directly north of the factory can open the shared wall.
        if worker["col"] == fc and worker["row"] == fr + 1:
            if has_wall(obs, config, worker["col"], worker["row"], "SOUTH") \
                    and not is_fixed_wall(worker["col"], "SOUTH", width):
                return "REMOVE_SOUTH"
            # Keep carving the corridor north for the factory to follow.
            if has_wall(obs, config, worker["col"], worker["row"], "NORTH") \
                    and not is_fixed_wall(worker["col"], "NORTH", width):
                return "REMOVE_NORTH"

    # Otherwise get into the staging cell just north of the factory, else follow north.
    goals = None
    if factory is not None:
        goals = {(factory["col"], factory["row"] + 1)}
    if goals:
        action, new_cell = _move_action(obs, config, worker, goals, occupied, min_row)
        if action:
            _commit(occupied, start, new_cell)
            return action

    reserved = set(occupied)
    reserved.discard(start)
    d, new_cell = safe_fallback_move(obs, config, worker, reserved, occupied, min_row)
    if d:
        _commit(occupied, start, new_cell)
        return d
    return "IDLE"


def _miner_action(obs, config, miner, factory, occupied):
    refuel = _refuel_factory_action(obs, config, miner, factory)
    if refuel:
        return refuel
    start = (miner["col"], miner["row"])
    south = _south(obs)
    min_row = south + 1
    nodes = _get(obs, "miningNodes", {}) or {}
    transform_cost = int(_get(config, "transformCost", 100))

    key = f"{miner['col']},{miner['row']}"
    on_node = key in nodes
    safe_here = (miner["row"] - south) >= MINER_SAFE_MARGIN
    if on_node and miner["energy"] >= transform_cost and safe_here:
        return "TRANSFORM"

    reserved = set(occupied)
    reserved.discard(start)
    dist, _parent = _bfs_all(obs, config, start, reserved, min_row)

    target = choose_mining_node_target(obs, config, start, dist)
    if target is None and factory is not None:
        # No nodes known: tag along north near the factory.
        target = (factory["col"], factory["row"] + 1)
        if target not in dist:
            target = None
    if target is None:
        # Fall back to frontier exploration (lower priority than scouts).
        target = choose_frontier_target(obs, config, start, int(_get(config, "visionMiner", 3)), dist)

    if target is not None:
        action, new_cell = _move_action(obs, config, miner, {target}, occupied, min_row)
        if action:
            _commit(occupied, start, new_cell)
            return action

    d, new_cell = safe_fallback_move(obs, config, miner, reserved, occupied, min_row)
    if d:
        _commit(occupied, start, new_cell)
        return d
    return "IDLE"


def _commit(occupied, old_cell, new_cell):
    """Release the cell we're leaving and claim the one we're moving into."""
    occupied.discard(old_cell)
    occupied.add(new_cell)


def _refuel_factory_action(obs, config, robot, factory):
    """If this unit is adjacent to a low-energy factory (no wall between) and can
    spare energy, donate it via TRANSFER. Keeping the factory alive is the whole
    game, so this income trumps whatever else the unit was going to do."""
    if factory is None:
        return None
    if factory["energy"] >= FACTORY_REFUEL_BELOW:
        return None
    if robot["energy"] < DONOR_MIN_ENERGY:
        return None
    dc = factory["col"] - robot["col"]
    dr = factory["row"] - robot["row"]
    if abs(dc) + abs(dr) != 1:
        return None
    for d, (ddc, ddr) in DELTA.items():
        if (ddc, ddr) == (dc, dr):
            if has_wall(obs, config, robot["col"], robot["row"], d):
                return None
            return "TRANSFER_" + d
    return None


# --- Entry point ------------------------------------------------------------

def agent(obs, config):
    """Return {uid: action_string} for every robot we own. Never raises."""
    try:
        return _agent_impl(obs, config)
    except Exception:
        # Last-resort safety: idle everything we own so we never crash out.
        try:
            me = _get(obs, "player", 0)
            return {uid: "IDLE" for uid, r in parse_robots(obs).items() if r["owner"] == me}
        except Exception:
            return {}


def _agent_impl(obs, config):
    me = _get(obs, "player", 0)
    robots = parse_robots(obs)
    mine = {uid: r for uid, r in robots.items() if r["owner"] == me}

    actions = {}
    # Cells that will be occupied next turn. Seeded with all our current
    # positions; a robot that moves releases its origin and claims its target
    # (via _commit). This is the reservation system that prevents friendly
    # collisions / friendly fire.
    occupied = {(r["col"], r["row"]) for r in mine.values()}

    factory = next((r for r in mine.values() if r["type"] == FACTORY), None)

    # Process order minimizes collisions: Factory, Workers, Miners, Scouts.
    order = (
        [r for r in mine.values() if r["type"] == FACTORY]
        + [r for r in mine.values() if r["type"] == WORKER]
        + [r for r in mine.values() if r["type"] == MINER]
        + [r for r in mine.values() if r["type"] == SCOUT]
    )

    for robot in order:
        uid = robot["uid"]
        rtype = robot["type"]
        if rtype == FACTORY:
            actions[uid] = _factory_action(obs, config, robot, mine, occupied)
        elif rtype == WORKER:
            actions[uid] = _worker_action(obs, config, robot, factory, occupied)
        elif rtype == MINER:
            actions[uid] = _miner_action(obs, config, robot, factory, occupied)
        elif rtype == SCOUT:
            actions[uid] = _scout_action(obs, config, robot, occupied, factory)
        else:
            actions[uid] = "IDLE"

    return actions


# --- Local test harness -----------------------------------------------------
# Guarded so it never runs when Kaggle imports this module for submission.
if __name__ == "__main__":
    from kaggle_environments import make

    env = make("crawl", configuration={"randomSeed": 42}, debug=True)
    env.run([agent, "random"])
    print(env.render(mode="ansi"))
    print("Episode finished. Rewards:", [s["reward"] for s in env.steps[-1]])
