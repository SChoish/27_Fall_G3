"""Recount first hits and cell/link vetoes from a saved 20-lane benchmark."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from tarfile import open as open_tar

import numpy as np

from attainment_report import aggregate
from audit import checks
from link_sizing_panel import LINK_ALLOWANCE, link_residual


def rows_from(root):
    directory = root / "trajectories"
    if directory.is_dir():
        for path in sorted(directory.glob("layout*.json")):
            yield json.loads(path.read_text())
        return
    with open_tar(root / "trajectories.tar.gz", "r:*") as records:
        for member in records:
            if member.isfile() and member.name.endswith(".json"):
                with records.extractfile(member) as stream:
                    yield json.load(stream)


def stage_status(raw, ref, fail, baseline, goal):
    valid, edp = checks(raw, ref)
    cell = valid and edp <= goal
    link = fail is not None and link_residual(fail, baseline) <= 0
    return cell, link, edp if valid else None


def trajectory(row, goal, cap):
    dev_ref, held_ref = row["ref"][:2], row["ref"][2:]
    base_dev, base_held = row["ref_link"]
    counts = Counter()
    calls = 4
    hit, verified_edp = None, None
    fine_charged = False
    for step in row["trace"]:
        if hit is not None:
            raise ValueError("trajectory continued after first success")
        calls += 2
        cell, link, _ = stage_status(step["dev"], dev_ref, step["dev_link"], base_dev, goal)
        counts["dev_tried"] += 1
        counts["dev_cell_pass"] += cell
        counts["dev_link_veto"] += cell and not link
        if (step["check"] is not None) != bool(cell and link and calls + 2 <= cap):
            raise ValueError("incorrect confirmation trigger")
        if step["check"] is not None:
            calls += 2
            cell, link, _ = stage_status(step["check"], held_ref, step["check_link"], base_held, goal)
            counts["held_tried"] += 1
            counts["held_cell_pass"] += cell
            counts["held_link_veto"] += cell and not link
            needed = 4 if fine_charged else 8
            if (step["fine"] is not None) != bool(cell and link and calls + needed <= cap):
                raise ValueError("incorrect fine trigger")
            if cell and link and calls + needed > cap:
                counts["fine_skipped_budget"] += 1
        if step["fine"] is not None:
            calls += 4 if fine_charged else 8
            fine_charged = True
            pairs = [stage_status(step["fine"][2*i:2*i+2], row["fine_ref"][2*i:2*i+2],
                                  step["fine_link"][i], row["fine_ref_link"][i], goal)
                     for i in (0, 1)]
            cell = all(item[0] for item in pairs)
            link = all(item[1] for item in pairs)
            counts["fine_tried"] += 1
            counts["fine_cell_pass"] += cell
            counts["fine_link_veto"] += cell and not link
            if cell and link:
                hit = calls
                verified_edp = max(item[2] for item in pairs)
        if calls != step["calls"] or calls > cap:
            raise ValueError("cost ledger mismatch")
    if (row["calls"] != calls or len(row["ledger"]) != calls or row["hit"] != hit
            or (hit is None and calls != cap)):
        raise ValueError("trajectory outcome or cost mismatch")
    if [item["call"] for item in row["ledger"]] != list(range(1, calls + 1)):
        raise ValueError("unordered ledger")
    if sum(item["cache_hit"] for item in row["ledger"]) != row["cache_hits"]:
        raise ValueError("wrong cache-hit count")
    if row["physical_transients"] != calls - row["cache_hits"]:
        raise ValueError("wrong physical-transient count")
    if sum("error" in entry["raw"] for entry in row["ledger"]) != row["invalid_attempts"]:
        raise ValueError("wrong invalid-attempt count")
    if hit is None:
        if not counts["dev_cell_pass"]:
            counts["unsolved_no_dev_cell_pass"] += 1
        elif counts["dev_cell_pass"] == counts["dev_link_veto"]:
            counts["unsolved_no_dev_link_pass"] += 1
        elif not counts["held_cell_pass"]:
            counts["unsolved_no_held_cell_pass"] += 1
        elif counts["held_cell_pass"] == counts["held_link_veto"]:
            counts["unsolved_no_held_link_pass"] += 1
        elif not counts["fine_tried"]:
            counts["unsolved_no_fine_called"] += 1
        elif not counts["fine_cell_pass"]:
            counts["unsolved_no_fine_cell_pass"] += 1
        else:
            counts["unsolved_no_fine_link_pass"] += 1
    warm = hit is not None and hit <= row["trace"][min(4, len(row["trace"])) - 1]["calls"]
    return hit, warm, verified_edp, counts


def seed_blocks(rows, seeds, budgets, methods):
    """Treat all layouts sharing an optimizer seed as one resampling block."""
    by_layout = {(key[0], method): hit for key, method, hit, _ in rows}
    blocks = defaultdict(list)
    for layout, seed in sorted(seeds.items()):
        blocks[seed].append(layout)
    sizes = np.array([len(group) for group in blocks.values()])
    samples = np.random.default_rng(816).integers(len(blocks), size=(10000, len(blocks)))
    result = {}
    for budget in budgets:
        methods_at_budget = {}
        baseline = {layout: min(by_layout[layout, "random_templates"] or budget, budget)
                    for layout in seeds}
        for method in methods:
            sums, seed_counts = [], {}
            for seed, group in blocks.items():
                hits = [by_layout[layout, method] for layout in group]
                effect = sum(min(hit or budget, budget) - baseline[layout]
                             for hit, layout in zip(hits, group))
                sums.append(effect)
                seed_counts[str(seed)] = dict(layouts=len(group),
                    successes=sum(hit is not None and hit <= budget for hit in hits),
                    mean_difference_from_random=effect / len(group))
            boots = np.array(sums)[samples].sum(axis=1) / sizes[samples].sum(axis=1)
            ci = np.percentile(boots, [2.5, 97.5]).tolist() if len(blocks) > 1 else None
            methods_at_budget[method] = dict(block_bootstrap_ci95=ci, by_seed=seed_counts)
        result[str(budget)] = methods_at_budget
    return result


def build_report(root):
    summary = json.loads((root / "summary.json").read_text())
    methods, cap, goal = summary["methods"], summary["cap"], summary["goal"]
    rows, seed_by_layout, warm_by_layout, witnessed = [], {}, {}, set()
    costs = Counter()
    counters = defaultdict(Counter)
    ten_by_method = Counter()
    verified = []
    for row in rows_from(root):
        layout, method = row["layout"], row["arm"]
        if method not in methods or not 0 <= layout < summary["layouts"]:
            raise ValueError(f"unexpected layout or method: {layout}, {method}")
        if seed_by_layout.setdefault(layout, row["seed"]) != row["seed"]:
            raise ValueError(f"unmatched seeds at layout {layout}")
        xs = [step["x"] for step in row["trace"][:4]]
        common = warm_by_layout.setdefault(layout, xs)
        n = min(len(xs), len(common))
        if xs[:n] != common[:n]:
            raise ValueError(f"unmatched starting proposals at layout {layout}")
        if len(xs) > len(common):
            warm_by_layout[layout] = xs
        hit, warm, verified_edp, counts = trajectory(row, goal, cap)
        rows.append(((layout,), method, hit, warm))
        counters[method].update(counts)
        if verified_edp is not None:
            verified.append(verified_edp)
            if verified_edp <= .90:
                ten_by_method[method] += 1
                witnessed.add(layout)
        for key in ("calls", "cache_hits", "physical_transients", "invalid_attempts"):
            costs[key] += row[key]
    for local, stored in (("calls", "logical_target_calls"), ("cache_hits", "cache_hits"),
                          ("physical_transients", "physical_target_transients"), ("invalid_attempts", "invalid_attempts")):
        if costs[local] != summary[stored]:
            raise ValueError(f"summary mismatch for {stored}")
    budgets = sorted({b for b in (20, 40, 60, cap) if b <= cap})
    result = aggregate(rows, budgets, cap, "random_templates", (summary["layouts"], len(methods)))
    for arm, data in result["budgets"][str(cap)]["methods"].items():
        if (data["successes"] != methods[arm]["link_qualified_hits"]
                or data["restricted_mean_calls"] != methods[arm]["mean_standalone_calls"]):
            raise ValueError(f"summary mismatch for method {arm}")
    result.update(stage_counts={method: dict(counters[method]) for method in methods},
                  verified_10pct_witnesses={method: ten_by_method[method] for method in methods},
                  layouts_with_10pct_witness=len(witnessed),
                  best_verified_edp_ratio=min(verified, default=None),
                  seed_block_sensitivity=seed_blocks(rows, seed_by_layout, budgets, methods),
                  archive_goal=goal, cap=cap)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="run or archived results directory")
    args = parser.parse_args()
    print(json.dumps(build_report(args.results), indent=2, allow_nan=False))
