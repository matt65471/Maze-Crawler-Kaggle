"""Evaluation harness for the Maze Crawler ("crawl") bot.

Measures real strength instead of just "did it beat random once". For each
opponent it plays a set of seeds on BOTH player sides (so a side-advantage in
the starting layout can't flatter the result) and reports, from our agent's
perspective:

  - win / tie / loss rate
  - survival rate (our factory still alive at the end)
  - average north progress (our factory's final row)
  - average final energy (our score proxy)

Opponents are just other agent functions in the second slot:
  - "random"  : the built-in random agent shipped with kaggle_environments
  - "self"    : a copy of our own bot (mirror match)
  - or any older snapshot you keep around (e.g. import from main_v1.py)

Usage:
    python eval.py            # default seed count
    python eval.py 24         # use 24 seeds per side
"""

import sys

from kaggle_environments import make

import main

# Frozen previous version, used for head-to-head measurement. `random` is
# saturated (we win 100%) and the self-mirror is symmetric (~50% by
# construction), so neither can detect small gains; playing the live bot against
# a frozen snapshot gives a real, asymmetric win-rate signal.
try:
    import agent_baseline
    _BASELINE = agent_baseline.agent
except Exception:
    _BASELINE = None


def _factory_of(global_robots, side):
    for d in global_robots.values():
        if d[0] == 0 and d[4] == side:  # type == FACTORY, owner == side
            return d
    return None


def _side_energy(global_robots, side):
    return sum(d[3] for d in global_robots.values() if d[4] == side)


def play_match(agent0, agent1, seed):
    """Play one game; return a per-side dict of final stats."""
    env = make("crawl", configuration={"randomSeed": seed}, debug=False)
    env.run([agent0, agent1])
    last = env.steps[-1]
    rewards = [last[0]["reward"], last[1]["reward"]]
    gr = last[0]["observation"].get("globalRobots", {}) or {}
    stats = {"steps": len(env.steps), "rewards": rewards}
    for side in (0, 1):
        fac = _factory_of(gr, side)
        stats[side] = {
            "alive": fac is not None,
            "factory_row": fac[2] if fac else None,
            "energy": _side_energy(gr, side),
        }
    return stats


def _outcome(rewards, our_side):
    opp = 1 - our_side
    if rewards[our_side] > rewards[opp]:
        return "win"
    if rewards[our_side] < rewards[opp]:
        return "loss"
    return "tie"


def _run_side(our_agent, opponent, seeds, our_side):
    """Play every seed with our agent on `our_side`; return raw per-game records."""
    records = []
    for seed in seeds:
        if our_side == 0:
            s = play_match(our_agent, opponent, seed)
        else:
            s = play_match(opponent, our_agent, seed)
        records.append((s, our_side))
    return records


def summarize(records):
    """Aggregate raw (stats, our_side) records from our perspective."""
    wins = ties = losses = survived = 0
    rows, energies = [], []
    for s, our_side in records:
        out = _outcome(s["rewards"], our_side)
        if out == "win":
            wins += 1
        elif out == "tie":
            ties += 1
        else:
            losses += 1
        mine = s[our_side]
        if mine["alive"]:
            survived += 1
            if mine["factory_row"] is not None:
                rows.append(mine["factory_row"])
        energies.append(mine["energy"])
    n = len(records)
    return {
        "n": n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "winrate": wins / n if n else 0.0,
        "survival": survived / n if n else 0.0,
        "avg_factory_row": (sum(rows) / len(rows)) if rows else 0.0,
        "avg_energy": (sum(energies) / len(energies)) if energies else 0.0,
    }


def evaluate_matchup(our_agent, opponent, seeds):
    """Evaluate against one opponent across both player sides."""
    records = _run_side(our_agent, opponent, seeds, our_side=0)
    records += _run_side(our_agent, opponent, seeds, our_side=1)
    return summarize(records)


def main_eval(seed_count=12):
    seeds = list(range(1, seed_count + 1))
    opponents = {
        "random": "random",
        "self": main.agent,
    }
    if _BASELINE is not None:
        # The head-to-head that actually measures progress: live bot vs frozen
        # snapshot. >50% here means the current changes genuinely beat the old bot.
        opponents["baseline"] = _BASELINE
    print(f"Evaluating main.agent over {len(seeds)} seeds x 2 sides "
          f"= {2 * len(seeds)} games per opponent.\n")
    header = f"{'opponent':<10} {'games':>6} {'W/T/L':>10} {'winrate':>8} {'survival':>9} {'avgRow':>7} {'avgEnergy':>10}"
    print(header)
    print("-" * len(header))
    for name, opp in opponents.items():
        r = evaluate_matchup(main.agent, opp, seeds)
        print(f"{name:<10} {r['n']:>6} "
              f"{r['wins']:>3}/{r['ties']:>1}/{r['losses']:<3} "
              f"{r['winrate']*100:>7.1f}% {r['survival']*100:>8.1f}% "
              f"{r['avg_factory_row']:>7.1f} {r['avg_energy']:>10.1f}")


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main_eval(count)
