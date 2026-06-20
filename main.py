"""Maze Crawler ("crawl") competition bot — Factory-North strategy.

Prime directive: drive the Factory as far north as possible (the maze scrolls
north and crushes anything that falls below `southBound`).

  - Factory: move to the north-most reachable cell every turn it can; build only
    on move-cooldown turns or when boxed (Scouts, a Miner when a node is visible,
    a Worker only as a last resort to break out); JUMP_NORTH when truly walled in.
  - Scout: exploration-first, clearing fog along the Factory's path; only grabs a
    crystal when it is a cheap detour.
  - Miner: built only when a mining node is visible; TRANSFORM on a node when safe.
  - Worker: minimal — spawned only when the Factory is boxed, to remove the wall.

All robots share a BFS planner (unknown cells treated as passable, so we can plan
into the fog) and a per-turn reservation system so two friendly units never aim
for the same cell (friendly fire is real). Nearby units TRANSFER energy to refuel
a low Factory, since a Factory at 0 energy is forced idle and gets scrolled away.
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

# Scouts are exploration-first: they only divert for a crystal if it is within a
# short detour of where they already are.
CRYSTAL_DETOUR = 4
# Frontier scoring bonuses that bias scouts to clear fog the factory will walk
# through: reward revealing cells north of the factory and near its column.
FRONTIER_AHEAD_WEIGHT = 2      # per row north of the factory
FRONTIER_COL_WEIGHT = 1        # penalty per column away from the factory

# The factory must not backtrack far south chasing a winding path — losing
# ground is how it gets caught by the scroll. Allow only a small dip below its
# current row to sidestep a wall.
FACTORY_BACKTRACK_LIMIT = 2
# Jump/survival lookahead: rather than a fixed margin trigger, project whether
# walking can out-run the scroll. We jump only when the projection says we'd
# lose the race within the jump's own cooldown window (or we're truly boxed),
# so the 20-turn jump is saved for when it actually matters.
WALK_EFFICIENCY = 0.75        # fraction of nominal walk speed realized (mazes force detours)
FACTORY_CRITICAL_MARGIN = 1   # at/below this buffer above the boundary, jump now if able
FACTORY_GOAL_SWITCH_MARGIN = 8
FACTORY_CORRIDOR_DEPTH = 3
FACTORY_ROUTE_LOOKAHEAD = 6
FACTORY_LOW_MARGIN = 4
FACTORY_JUMP_ROUTE_MARGIN = 5
FACTORY_REFUEL_LOOKAHEAD = 6

# Per-side memory persists across turns in the Kaggle worker. The factory uses
# it to avoid thrashing between newly discovered routes unless the new route is
# materially faster than the route it is already walking.
_FACTORY_GOALS = {}
_FACTORY_CORRIDORS = {}

# Crush table from the environment (see crawl.py): an (attacker, victim) pair
# means the attacker wins. Strength order Factory > Miner > Worker > Scout, and
# two units of the SAME type destroy each other (mutual) regardless of owner.
_CRUSHES = {
    (FACTORY, MINER), (FACTORY, WORKER), (FACTORY, SCOUT),
    (MINER, WORKER), (MINER, SCOUT),
    (WORKER, SCOUT),
}


def _loses_to(my_type, enemy_type):
    """True if a head-on collision would destroy my unit: the enemy crushes me,
    or we are the same type (mutual destruction — including factory vs factory)."""
    if (enemy_type, my_type) in _CRUSHES:
        return True
    return enemy_type == my_type


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


def _scroll_interval(step, config):
    """Turns between scroll advances at `step` (mirrors the environment's
    get_scroll_interval). The period is deterministic from the step; only the
    phase/counter is hidden, so we use this as the scroll *rate*."""
    ramp = int(_get(config, "scrollRampSteps", 450))
    start = int(_get(config, "scrollStartInterval", 10))
    end = int(_get(config, "scrollEndInterval", 2))
    if step >= ramp:
        return end
    progress = step / max(1, ramp)
    interval = start - (start - end) * progress
    return max(end, round(interval))


def _factory_should_jump(obs, config, factory, has_north_target, boxed,
                         distance_to_progress=None):
    """Lookahead jump decision: project the survival margin over the jump's
    cooldown window and only jump when walking can't keep us ahead of the scroll
    (or we're boxed in). Returns True to jump now."""
    if boxed:
        return True  # can't progress on foot — jump to break out
    margin = factory["row"] - _south(obs)
    if margin <= FACTORY_CRITICAL_MARGIN:
        return True  # about to be overrun; jump regardless
    if distance_to_progress is not None:
        projected = project_factory_margin(obs, config, factory, distance_to_progress)
        if margin <= FACTORY_JUMP_ROUTE_MARGIN and projected <= FACTORY_CRITICAL_MARGIN:
            return True
    if has_north_target and margin > FACTORY_JUMP_ROUTE_MARGIN:
        return False
    step = int(_get(obs, "step", 0))
    scroll_rate = 1.0 / max(1, _scroll_interval(step, config))
    move_period = int(_get(config, "factoryMovePeriod", 2))
    walk_rate = WALK_EFFICIENCY / max(1, move_period)
    net = walk_rate - scroll_rate
    if net >= 0:
        return False  # we out-run the scroll on foot — save the jump
    # Margin shrinks: will we reach the danger floor within the jump's cooldown?
    horizon = int(_get(config, "factoryJumpCooldown", 20))
    turns_to_danger = (margin - FACTORY_CRITICAL_MARGIN) / (-net)
    return turns_to_danger <= horizon


# --- Basic geometry / wall helpers -----------------------------------------

def move_delta(direction):
    return DELTA[direction]


def apply_move(col, row, direction):
    dc, dr = DELTA[direction]
    return col + dc, row + dr


def _mirror_bits(v):
    """Mirror a wall bitfield across the vertical axis: keep N/S, swap E and W."""
    out = 0
    if v & WALL_BIT["NORTH"]:
        out |= WALL_BIT["NORTH"]
    if v & WALL_BIT["SOUTH"]:
        out |= WALL_BIT["SOUTH"]
    if v & WALL_BIT["EAST"]:
        out |= WALL_BIT["WEST"]
    if v & WALL_BIT["WEST"]:
        out |= WALL_BIT["EAST"]
    return out


def get_wall(obs, config, col, row):
    """Raw wall bitfield at (col,row); returns -1 if the cell is undiscovered/off-window.

    The maze is generated by mirroring the left half onto the right (with E/W
    swapped), so for an undiscovered cell we fall back to its mirror cell's walls
    when that side has been seen. This halves how much we must physically explore.
    Real (discovered) data always wins; the only inaccuracy is a wall a worker
    toggled on the unseen side, which is rare and no worse than the optimistic
    "unknown = open" assumption we'd otherwise make."""
    walls = _get(obs, "walls", []) or []
    width = _width(config)
    south = _south(obs)
    if not (0 <= col < width):
        return -1
    idx = (row - south) * width + col
    if 0 <= idx < len(walls):
        v = walls[idx]
        if v >= 0:
            return v
    # Undiscovered directly: infer from the mirror cell if it has been seen.
    midx = (row - south) * width + (width - 1 - col)
    if 0 <= midx < len(walls):
        mv = walls[midx]
        if mv >= 0:
            return _mirror_bits(mv)
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


def _enemy_threat_map(obs, config, me):
    """Map our-unit-type -> set of cells an enemy could occupy next turn that
    would destroy a unit of that type.

    Defensive combat awareness: enemies are only visible within our vision, so
    this is local and reactive. We assume (conservatively) that each visible
    enemy may stay put or step into any wall-open neighbor next turn."""
    danger = {FACTORY: set(), SCOUT: set(), WORKER: set(), MINER: set()}
    robots = parse_robots(obs)
    for e in robots.values():
        if e["owner"] == me:
            continue
        et = e["type"]
        cells = [(e["col"], e["row"])]
        for d in DIRS:
            if has_wall(obs, config, e["col"], e["row"], d):
                continue
            nc, nr = apply_move(e["col"], e["row"], d)
            if in_bounds(obs, config, nc, nr):
                cells.append((nc, nr))
        for my_type in (FACTORY, SCOUT, WORKER, MINER):
            if _loses_to(my_type, et):
                danger[my_type].update(cells)
    return danger


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


def _factory_half_center(config, factory):
    width = _width(config)
    half = width // 2
    if factory["col"] < half:
        return max(1, half // 2)
    return min(width - 2, half + half // 2)


def factory_corridor_cells(path, depth=FACTORY_CORRIDOR_DEPTH):
    """Cells on the factory's near-term route that helpers should avoid."""
    if not path:
        return set()
    return set(path[1:1 + depth])


def score_factory_target(obs, config, factory, target, dist, parent, danger=None, occupied=None):
    """Higher is better: north progress with penalties for slow, risky routes."""
    danger = danger or set()
    occupied = occupied or set()
    start = (factory["col"], factory["row"])
    path = _reconstruct(parent, start, target)
    if not path:
        return None

    frow = factory["row"]
    distance = dist[target]
    row_gain = target[1] - frow
    center = _factory_half_center(config, factory)
    lookahead = path[1:1 + FACTORY_ROUTE_LOOKAHEAD]
    backtrack = sum(max(0, frow - r) for _c, r in lookahead)
    max_dip = max([max(0, frow - r) for _c, r in lookahead] or [0])
    unknown = sum(1 for c, r in lookahead if not is_known(obs, config, c, r))
    danger_hits = sum(1 for cell in lookahead if cell in danger)
    blocked_hits = sum(1 for cell in lookahead if cell in occupied and cell != start)
    first_blocked = 1 if len(path) > 1 and path[1] in occupied else 0

    return (
        row_gain * 18
        - distance * 3
        - abs(target[0] - center)
        - backtrack * 8
        - max_dip * 6
        - unknown * 2
        - danger_hits * 80
        - blocked_hits * 50
        - first_blocked * 100
    )


def choose_factory_target(obs, config, factory, dist, parent, previous_goal=None,
                          danger=None, occupied=None):
    """Best reachable route from the precomputed BFS maps.

    The factory still values north progress most, but now accounts for route
    length, backtracking, fog confidence, danger, and near-term blockage. If the
    previous goal remains reachable, keep it unless the replacement is clearly
    better; this prevents path thrashing as scouts reveal new cells."""
    frow = factory["row"]
    best, best_score = None, None
    for cell, d in dist.items():
        c, r = cell
        if r <= frow:
            continue
        score = score_factory_target(obs, config, factory, cell, dist, parent, danger, occupied)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best, best_score = cell, score
    if previous_goal in dist and previous_goal[1] > frow:
        previous_score = score_factory_target(
            obs, config, factory, previous_goal, dist, parent, danger, occupied
        )
        if previous_score is None:
            return best
        if best is not None and best != previous_goal \
                and best_score > previous_score + FACTORY_GOAL_SWITCH_MARGIN:
            return best
        return previous_goal
    return best


def choose_crystal_target(obs, config, start, dist, max_dist=None):
    """Best visible crystal by score = energy - 2 * path_distance.

    When `max_dist` is set, only crystals reachable within that many steps are
    considered (used so scouts only divert for crystals that are a cheap detour)."""
    crystals = _get(obs, "crystals", {}) or {}
    best, best_score = None, None
    for key, energy in crystals.items():
        try:
            cell = _parse_cell(key)
        except Exception:
            continue
        if cell not in dist:
            continue
        if max_dist is not None and dist[cell] > max_dist:
            continue
        score = energy - 2 * dist[cell]
        if best_score is None or score > best_score:
            best, best_score = cell, score
    return best


def choose_frontier_target(obs, config, start, vision, dist, factory=None, corridor=None):
    """Reachable cell that maximizes information gain.

    Base score = 5*unknown_in_vision + 2*row_progress - distance. When a factory
    is given, add a bias to reveal terrain the factory will walk through: a bonus
    for cells north of the factory and a penalty for straying from its column.
    This keeps the fog cleared along the factory's intended path."""
    south = _south(obs)
    corridor = corridor or set()
    best, best_score = None, None
    for cell, d in dist.items():
        if cell == start:
            continue
        c, r = cell
        unknown = count_unknown_visible_cells(obs, config, c, r, vision)
        if unknown == 0:
            continue
        score = 5 * unknown + 2 * (r - south) - d
        if factory is not None:
            score += FRONTIER_AHEAD_WEIGHT * max(0, r - factory["row"])
            score -= FRONTIER_COL_WEIGHT * abs(c - factory["col"])
        if corridor:
            near_corridor = min(abs(c - cc) + abs(r - rr) for cc, rr in corridor)
            score += max(0, 4 - near_corridor) * 3
        if best_score is None or score > best_score:
            best, best_score = cell, score
    return best


def _first_reachable_progress_distance(factory, dist):
    """Shortest distance to any reachable cell north of the factory."""
    frow = factory["row"]
    best = None
    for _cell, d in dist.items():
        if _cell[1] <= frow:
            continue
        if best is None or d < best:
            best = d
    return best


def project_factory_margin(obs, config, factory, distance_to_progress=None):
    """Estimated scroll buffer after the factory reaches its next progress cell."""
    margin = factory["row"] - _south(obs)
    if distance_to_progress is None:
        return margin
    step = int(_get(obs, "step", 0))
    interval = max(1, _scroll_interval(step, config))
    move_period = max(1, int(_get(config, "factoryMovePeriod", 2)))
    turns = distance_to_progress * move_period
    scrolls = turns / interval
    return margin + 1 - scrolls


def build_value(kind, obs, config, factory, occupied, corridor, boxed, route_blocked):
    """Return whether this build is worth spending reserve under current pressure."""
    margin = factory["row"] - _south(obs)
    if kind in ("SCOUT", "MINER") and margin <= FACTORY_LOW_MARGIN:
        return False
    if kind == "WORKER":
        return boxed or route_blocked
    if kind == "MINER":
        nodes = _get(obs, "miningNodes", {}) or {}
        if not nodes:
            return False
        start = (factory["col"], factory["row"])
        reserved = set(occupied)
        reserved.discard(start)
        dist, _parent = _bfs_all(obs, config, start, reserved, max(_south(obs) + 2, factory["row"] - FACTORY_BACKTRACK_LIMIT))
        target = choose_mining_node_target(obs, config, start, dist)
        return target is not None and dist[target] <= 8 and (target[1] - _south(obs)) >= MINER_SAFE_MARGIN
    if kind == "SCOUT":
        return bool(_build_direction(obs, config, factory, occupied, corridor))
    return False


def _can_afford_worker_escape(factory, config):
    worker_cost = int(_get(config, "workerCost", 200))
    return factory["energy"] >= worker_cost + 25


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

def _move_action(obs, config, robot, goals, occupied, min_row=None, extra_blocked=None):
    """Plan a BFS move toward `goals`. Returns (action, new_cell) or (None, None).
    `extra_blocked` (e.g. enemy danger cells) is avoided in addition to friendlies."""
    if not _can_move_now(robot):
        return None, None
    start = (robot["col"], robot["row"])
    reserved = set(occupied)
    reserved.discard(start)
    if extra_blocked:
        reserved |= extra_blocked
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

def _factory_action(obs, config, factory, mine_robots, occupied, danger=None):
    """Prime directive: get the factory as far north as possible.

    Order of operations so north progress is never sacrificed for building:
      0. Lookahead jump: jump only if walking can't out-run the scroll (or boxed).
      1. If we can move this turn and a cell north of us is reachable, move there.
      2. Otherwise (move on cooldown, or boxed) build, reserve-gated.
      3. If genuinely boxed in, JUMP_NORTH when ready/safe, else IDLE.
    """
    danger = danger or set()
    south = _south(obs)
    start = (factory["col"], factory["row"])
    memory_key = factory["owner"]
    if int(_get(obs, "step", 0)) <= 1:
        _FACTORY_GOALS.pop(memory_key, None)
        _FACTORY_CORRIDORS.pop(memory_key, None)
    # Bound how far south the factory may dip: only a small detour below its
    # current row (never march back toward the scroll), and always stay clear of
    # the boundary. This is what stops the factory backtracking into death.
    margin = factory["row"] - south
    backtrack_limit = 0 if margin <= FACTORY_LOW_MARGIN else FACTORY_BACKTRACK_LIMIT
    min_row = max(south + 2, factory["row"] - backtrack_limit)

    def _try_jump():
        lc, lr = factory["col"], factory["row"] + 2
        if not in_bounds(obs, config, lc, lr):
            return None
        if (lc, lr) in occupied or (lc, lr) in danger:
            return None
        if (lr - south) < 1:
            return None
        _commit(occupied, start, (lc, lr))
        _FACTORY_CORRIDORS[memory_key] = set()
        return "JUMP_NORTH"

    # Optimistic BFS (fog treated as open) over reachable cells, avoiding danger.
    # Run it first so the jump decision knows whether we can walk north at all.
    reserved = set(occupied)
    reserved.discard(start)
    reserved |= danger
    dist, parent = _bfs_all(obs, config, start, reserved, min_row)
    north_target = choose_factory_target(
        obs, config, factory, dist, parent, _FACTORY_GOALS.get(memory_key),
        danger, occupied
    )
    if north_target is None:
        _FACTORY_GOALS.pop(memory_key, None)
    else:
        _FACTORY_GOALS[memory_key] = north_target
    corridor = set()
    if north_target is not None:
        planned_path = _reconstruct(parent, start, north_target)
        if planned_path and len(planned_path) > 1:
            corridor = factory_corridor_cells(planned_path)
    boxed = north_target is None  # no cell north of us is reachable -> walled in
    distance_to_progress = _first_reachable_progress_distance(factory, dist)
    route_blocked = boxed or has_wall(obs, config, factory["col"], factory["row"], "NORTH") \
        or (distance_to_progress is not None and distance_to_progress > 4)

    def _publish_corridor():
        _FACTORY_CORRIDORS[memory_key] = set(corridor)
        occupied.update(corridor)

    # 0) Lookahead jump: project whether walking out-runs the scroll; jump only
    # when it doesn't (or we're boxed), so the 20-turn jump is saved for need.
    can_build_escape_worker = (
        workers := sum(1 for r in mine_robots.values() if r["type"] == WORKER)
    ) < 1 and _can_build_now(factory) and _can_afford_worker_escape(factory, config)
    if _can_jump_now(factory) and _factory_should_jump(
            obs, config, factory, not boxed, boxed, distance_to_progress) \
            and (not boxed or margin <= FACTORY_JUMP_ROUTE_MARGIN or not can_build_escape_worker):
        j = _try_jump()
        if j:
            return j

    # 1) Advance north whenever possible.
    if _can_move_now(factory) and north_target is not None:
        action, new_cell = _move_action(obs, config, factory, {north_target}, occupied, min_row, danger)
        if action:
            _commit(occupied, start, new_cell)
            _publish_corridor()
            return action

    # 2) Build phase: only reached on a move-cooldown turn or when boxed, so it
    # never costs us a northward step. Reserve-gated to avoid bankruptcy.
    scouts = sum(1 for r in mine_robots.values() if r["type"] == SCOUT)
    miners = sum(1 for r in mine_robots.values() if r["type"] == MINER)
    mining_visible = len(_get(obs, "miningNodes", {}) or {}) > 0
    energy = factory["energy"]
    scout_cost = int(_get(config, "scoutCost", 50))
    miner_cost = int(_get(config, "minerCost", 300))

    if _can_build_now(factory):
        if route_blocked and workers < 1 and _can_afford_worker_escape(factory, config) \
                and build_value("WORKER", obs, config, factory, occupied, corridor, boxed, route_blocked):
            d = _build_direction(obs, config, factory, occupied, corridor)
            if d:
                occupied.add(apply_move(factory["col"], factory["row"], d))
                _publish_corridor()
                return "BUILD_WORKER_" + d
        if scouts < 2 and energy - scout_cost >= FACTORY_RESERVE \
                and build_value("SCOUT", obs, config, factory, occupied, corridor, boxed, route_blocked):
            d = _build_direction(obs, config, factory, occupied, corridor)
            if d:
                occupied.add(apply_move(factory["col"], factory["row"], d))
                _publish_corridor()
                return "BUILD_SCOUT_" + d
        elif mining_visible and miners < 1 and energy - miner_cost >= FACTORY_RESERVE \
                and build_value("MINER", obs, config, factory, occupied, corridor, boxed, route_blocked):
            d = _build_direction(obs, config, factory, occupied, corridor)
            if d:
                occupied.add(apply_move(factory["col"], factory["row"], d))
                _publish_corridor()
                return "BUILD_MINER_" + d
        elif boxed and workers < 1 and _can_afford_worker_escape(factory, config) \
                and build_value("WORKER", obs, config, factory, occupied, corridor, boxed, route_blocked):
            d = _build_direction(obs, config, factory, occupied, corridor)
            if d:
                occupied.add(apply_move(factory["col"], factory["row"], d))
                _publish_corridor()
                return "BUILD_WORKER_" + d

    # 3) Boxed in: jump two cells north (ignores walls) if it lands safely.
    if boxed and _can_jump_now(factory):
        j = _try_jump()
        if j:
            return j
    _publish_corridor()
    return "IDLE"


def _build_direction(obs, config, factory, occupied, avoid_cells=None):
    """Choose a spawn direction whose neighbor is open, in-window, and unoccupied."""
    avoid_cells = {cell for cell in (avoid_cells or set()) if cell is not None}
    width = _width(config)
    side_first = "EAST" if factory["col"] < width // 2 else "WEST"
    side_second = "WEST" if side_first == "EAST" else "EAST"
    for d in [side_first, side_second, "SOUTH", "NORTH"]:
        if has_wall(obs, config, factory["col"], factory["row"], d):
            continue
        nc, nr = apply_move(factory["col"], factory["row"], d)
        if not in_bounds(obs, config, nc, nr):
            continue
        if (nc, nr) in occupied:
            continue
        if (nc, nr) in avoid_cells:
            continue
        return d
    return None


def _scout_action(obs, config, scout, occupied, factory, danger=None, corridor=None):
    danger = danger or set()
    corridor = corridor or set()
    refuel = _refuel_factory_action(obs, config, scout, factory)
    if refuel:
        return refuel
    start = (scout["col"], scout["row"])
    min_row = _south(obs) + 1  # never step onto the lethal boundary row
    reserved = set(occupied)
    reserved.discard(start)
    reserved |= danger
    dist, _parent = _bfs_all(obs, config, start, reserved, min_row)
    vision = int(_get(config, "visionScout", 5))

    # Exploration-first: head to the highest-information frontier (biased toward
    # the factory's path). Only divert for a crystal if it's a cheap detour.
    cheap_crystal = choose_crystal_target(obs, config, start, dist, max_dist=CRYSTAL_DETOUR)
    if cheap_crystal is not None:
        target = cheap_crystal
    else:
        target = choose_frontier_target(obs, config, start, vision, dist, factory, corridor)

    if target is not None:
        action, new_cell = _move_action(obs, config, scout, {target}, occupied, min_row, danger)
        if action:
            _commit(occupied, start, new_cell)
            return action

    d, new_cell = safe_fallback_move(obs, config, scout, reserved, occupied, min_row)
    if d:
        _commit(occupied, start, new_cell)
        return d
    return "IDLE"


def _worker_action(obs, config, worker, factory, occupied, danger=None):
    """Shadow the factory; clear a removable wall that pens it in to the north."""
    danger = danger or set()
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
        action, new_cell = _move_action(obs, config, worker, goals, occupied, min_row, danger)
        if action:
            _commit(occupied, start, new_cell)
            return action

    reserved = set(occupied)
    reserved.discard(start)
    reserved |= danger
    d, new_cell = safe_fallback_move(obs, config, worker, reserved, occupied, min_row)
    if d:
        _commit(occupied, start, new_cell)
        return d
    return "IDLE"


def _miner_action(obs, config, miner, factory, occupied, danger=None, corridor=None):
    danger = danger or set()
    corridor = corridor or set()
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
    reserved |= danger
    dist, _parent = _bfs_all(obs, config, start, reserved, min_row)

    target = choose_mining_node_target(obs, config, start, dist)
    if target is None and factory is not None:
        # No nodes known: tag along north near the factory.
        target = (factory["col"], factory["row"] + 1)
        if target not in dist:
            target = None
    if target is None:
        # Fall back to frontier exploration (lower priority than scouts).
        target = choose_frontier_target(
            obs, config, start, int(_get(config, "visionMiner", 3)), dist, factory, corridor
        )

    if target is not None:
        action, new_cell = _move_action(obs, config, miner, {target}, occupied, min_row, danger)
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
    upkeep = int(_get(config, "factoryUpkeep", 1))
    projected_energy = factory["energy"] - FACTORY_REFUEL_LOOKAHEAD * upkeep
    if factory["energy"] >= FACTORY_REFUEL_BELOW and projected_energy >= FACTORY_REFUEL_BELOW:
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

    # Defensive combat awareness: cells a visible enemy could reach next turn
    # that would destroy each of our unit types.
    threat = _enemy_threat_map(obs, config, me)

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
        corridor = _FACTORY_CORRIDORS.get(robot["owner"], set())
        if rtype == FACTORY:
            actions[uid] = _factory_action(obs, config, robot, mine, occupied, threat[FACTORY])
        elif rtype == WORKER:
            actions[uid] = _worker_action(obs, config, robot, factory, occupied, threat[WORKER])
        elif rtype == MINER:
            actions[uid] = _miner_action(obs, config, robot, factory, occupied, threat[MINER], corridor)
        elif rtype == SCOUT:
            actions[uid] = _scout_action(obs, config, robot, occupied, factory, threat[SCOUT], corridor)
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
