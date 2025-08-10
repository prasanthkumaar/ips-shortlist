import argparse
import json
import os
import random
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Any

from matching_utils import (
    Stratum,
    Targets,
    load_targets,
    normalize_participants,
    allocate_largest_remainder,
    read_csv_dicts,
    write_csv_dicts,
    STATUS_CONFIRMED,
    STATUS_TO_CONTACT,
    STATUS_REGISTERED,
)


def stratum_of(sex: str, age_group: str, race: str, edu: str) -> Stratum:
    return Stratum(sex=sex, age_group=age_group, race=race, education_level=edu)


def compute_adjusted_desired_per_stratum(
    targets: Targets,
    group_size: int,
    pre_counts: Dict[Stratum, int],
) -> Dict[Stratum, float]:
    # initial desired = max(0, target_prop * group_size - pre_counts)
    desired_raw: Dict[Stratum, float] = {}
    total_pos = 0.0
    for st, prop in targets.by_stratum_prop.items():
        want = max(0.0, prop * group_size - pre_counts.get(st, 0))
        desired_raw[st] = want
        total_pos += want
    if total_pos <= 0:
        # fallback to plain target proportions
        return {st: targets.by_stratum_prop[st] * group_size for st in targets.by_stratum_prop}
    scale = group_size / total_pos
    return {st: desired_raw[st] * scale for st in desired_raw}


def allocate_with_spill(
    desired: Dict[Stratum, float],
    caps: Dict[Stratum, int],
    total_to_assign: int,
    rng: random.Random,
) -> Dict[Stratum, int]:
    alloc = allocate_largest_remainder(desired, total_to_assign, caps)
    assigned = sum(alloc.values())
    # Fill leftovers to strata with capacity proportionally to target desire among available
    while assigned < total_to_assign:
        candidates = [(st, max(0, caps.get(st, 0) - alloc.get(st, 0)), desired.get(st, 0.0)) for st in desired]
        candidates = [(st, cap_left, want) for st, cap_left, want in candidates if cap_left > 0]
        if not candidates:
            break
        total_weight = sum(want for _, _, want in candidates)
        if total_weight <= 0:
            # distribute uniformly
            candidates.sort(key=lambda x: (x[1]), reverse=True)
            st = candidates[0][0]
            alloc[st] = alloc.get(st, 0) + 1
            assigned += 1
            continue
        # pick one by probability proportional to remaining desired
        r = rng.random() * total_weight
        cum = 0.0
        chosen = candidates[0][0]
        for st, cap_left, want in candidates:
            cum += want
            if r <= cum:
                chosen = st
                break
        alloc[chosen] = alloc.get(chosen, 0) + 1
        assigned += 1
    return alloc


def build_pools(participants, targets: Targets):
    # Pool of eligible REGISTERED by stratum
    pool: Dict[Stratum, List[int]] = defaultdict(list)
    for idx, p in enumerate(participants):
        if p.eligible:
            st = stratum_of(p.sex, p.age_group, p.race, p.education_level)
            pool[st].append(idx)
    return pool


def preaccount_counts(participants, group_label: str, targets: Targets) -> Tuple[Dict[Stratum, int], Dict[str, Counter]]:
    # per stratum and per-dimension marginals for statuses locked in this group
    per_stratum: Dict[Stratum, int] = defaultdict(int)
    marginals: Dict[str, Counter] = {dim: Counter() for dim in ["sex", "age_group", "race", "education_level"]}
    for p in participants:
        if p.group == group_label and p.sex and p.age_group and p.status in {STATUS_CONFIRMED, STATUS_TO_CONTACT}:
            st = stratum_of(p.sex, p.age_group, p.race, p.education_level)
            per_stratum[st] += 1
            marginals["sex"][p.sex] += 1
            marginals["age_group"][p.age_group] += 1
            marginals["race"][p.race] += 1
            marginals["education_level"][p.education_level] += 1
    return per_stratum, marginals


def compute_adjusted_dim_targets(targets: Targets, group_size: int, pre_dim_counts: Dict[str, Counter]) -> Dict[str, Dict[str, float]]:
    # For each dimension, get desired counts for NEW PICKS after subtracting pre-existing, then scale to sum group_size
    result: Dict[str, Dict[str, float]] = {}
    for dim, target_prop in targets.marginals_prop.items():
        desired = {cat: max(0.0, target_prop.get(cat, 0.0) * group_size - pre_dim_counts.get(dim, Counter()).get(cat, 0)) for cat in target_prop}
        total_pos = sum(desired.values())
        if total_pos <= 0:
            # fallback to plain, proportionally
            desired = {cat: target_prop.get(cat, 0.0) * group_size for cat in target_prop}
            total_pos = sum(desired.values())
        scale = group_size / total_pos if total_pos > 0 else 0.0
        result[dim] = {cat: desired[cat] * scale for cat in desired}
    return result


def select_from_pools(
    pools: Dict[Stratum, List[int]],
    alloc: Dict[Stratum, int],
    rng: random.Random,
) -> List[int]:
    chosen: List[int] = []
    for st, k in alloc.items():
        if k <= 0:
            continue
        candidates = pools.get(st, [])
        rng.shuffle(candidates)
        picked = candidates[:k]
        pools[st] = candidates[k:]
        chosen.extend(picked)
    return chosen


def group_dim_counts(participants, indices: List[int]) -> Dict[str, Counter]:
    counts: Dict[str, Counter] = {dim: Counter() for dim in ["sex", "age_group", "race", "education_level"]}
    for idx in indices:
        p = participants[idx]
        counts["sex"][p.sex] += 1
        counts["age_group"][p.age_group] += 1
        counts["race"][p.race] += 1
        counts["education_level"][p.education_level] += 1
    return counts


def score_mad(dim_counts: Dict[str, Counter], dim_targets: Dict[str, Dict[str, float]], n: int) -> float:
    total = 0.0
    for dim, target in dim_targets.items():
        cats = list(target.keys())
        if not cats:
            continue
        mad = 0.0
        for cat in cats:
            obs = dim_counts.get(dim, Counter()).get(cat, 0) / max(1, n)
            tgt = target.get(cat, 0.0) / max(1, n)
            mad += abs(obs - tgt)
        mad /= len(cats)
        total += mad
    return total


def greedy_swap(
    participants,
    picks_g1: List[int],
    picks_g2: List[int],
    dim_targets_g1: Dict[str, Dict[str, float]],
    dim_targets_g2: Dict[str, Dict[str, float]],
    rng: random.Random,
    max_iter: int = 2000,
) -> Tuple[List[int], List[int]]:
    n1, n2 = len(picks_g1), len(picks_g2)
    counts1 = group_dim_counts(participants, picks_g1)
    counts2 = group_dim_counts(participants, picks_g2)
    best1 = score_mad(counts1, dim_targets_g1, n1)
    best2 = score_mad(counts2, dim_targets_g2, n2)
    best = best1 + best2

    for _ in range(max_iter):
        if not picks_g1 or not picks_g2:
            break
        i = rng.randrange(len(picks_g1))
        j = rng.randrange(len(picks_g2))
        a = picks_g1[i]
        b = picks_g2[j]
        pa = participants[a]
        pb = participants[b]
        # simulate swap
        for dim, get_val in [
            ("sex", lambda p: p.sex),
            ("age_group", lambda p: p.age_group),
            ("race", lambda p: p.race),
            ("education_level", lambda p: p.education_level),
        ]:
            counts1[dim][get_val(pa)] -= 1
            counts1[dim][get_val(pb)] += 1
            counts2[dim][get_val(pb)] -= 1
            counts2[dim][get_val(pa)] += 1
        new_score = score_mad(counts1, dim_targets_g1, n1) + score_mad(counts2, dim_targets_g2, n2)
        if new_score < best:
            best = new_score
            picks_g1[i], picks_g2[j] = b, a
        else:
            # revert
            for dim, get_val in [
                ("sex", lambda p: p.sex),
                ("age_group", lambda p: p.age_group),
                ("race", lambda p: p.race),
                ("education_level", lambda p: p.education_level),
            ]:
                counts1[dim][get_val(pa)] += 1
                counts1[dim][get_val(pb)] -= 1
                counts2[dim][get_val(pb)] += 1
                counts2[dim][get_val(pa)] -= 1
    return picks_g1, picks_g2


def ensure_status_group_headers(fieldnames: List[str], mapping: Dict[str, str]) -> List[str]:
    out = list(fieldnames)
    status_col = mapping.get("status", "Status")
    group_col = mapping.get("group", "Group")
    if status_col not in out:
        out.append(status_col)
    if group_col not in out:
        out.append(group_col)
    return out

# ------------------- Race-first allocation helpers -------------------

def allocate_float_keys_with_caps(desired: Dict[str, float], caps: Dict[str, int], total_to_assign: int, rng: random.Random) -> Dict[str, int]:
    # floor then largest remainder with caps
    keys = list(desired.keys())
    base = {k: min(int(desired.get(k, 0.0)), max(0, caps.get(k, 0))) for k in keys}
    assigned = sum(base.values())
    rema = []
    for k in keys:
        rem = max(0.0, desired.get(k, 0.0) - base[k]) if caps.get(k, 0) > base[k] else 0.0
        rema.append((rem, k))
    remaining = max(0, total_to_assign - assigned)
    rema.sort(key=lambda t: (t[0], t[1]), reverse=True)
    i = 0
    while remaining > 0 and i < len(rema):
        rem, k = rema[i]
        if base[k] < caps.get(k, 0):
            base[k] += 1
            remaining -= 1
        i += 1
        if i >= len(rema):
            i = 0
    return base


def allocate_race_first(targets: Targets, group_size: int, pre_dim_counts: Dict[str, Counter], caps_by_stratum: Dict[Stratum, int], rng: random.Random) -> Dict[Stratum, int]:
    # Desired per race after pre-account
    tgt_race_prop = targets.marginals_prop["race"]
    desired_race = {r: max(0.0, tgt_race_prop.get(r, 0.0) * group_size - pre_dim_counts.get("race", Counter()).get(r, 0)) for r in tgt_race_prop}
    total_pos = sum(desired_race.values())
    if total_pos <= 0:
        desired_race = {r: tgt_race_prop.get(r, 0.0) * group_size for r in tgt_race_prop}
        total_pos = sum(desired_race.values())
    scale = group_size / total_pos if total_pos > 0 else 0.0
    desired_race = {r: desired_race[r] * scale for r in desired_race}

    # Caps per race from pools
    caps_race: Dict[str, int] = {}
    for st, cap in caps_by_stratum.items():
        caps_race[st.race] = caps_race.get(st.race, 0) + cap

    # Allocate counts to races with caps
    race_counts = allocate_float_keys_with_caps(desired_race, caps_race, group_size, rng)

    # Within each race, allocate to strata proportional to target weights conditioned on race
    # Compute conditional weights w_st = prop(st) / sum_{st in race} prop(st)
    prop_by_race_sum: Dict[str, float] = {}
    for st, prop in targets.by_stratum_prop.items():
        prop_by_race_sum[st.race] = prop_by_race_sum.get(st.race, 0.0) + prop

    alloc: Dict[Stratum, int] = {st: 0 for st in targets.by_stratum_prop}
    for race, r_total in race_counts.items():
        # Build desired within this race
        desired_within: Dict[Stratum, float] = {}
        caps_within: Dict[Stratum, int] = {}
        denom = max(1e-9, prop_by_race_sum.get(race, 0.0))
        for st, prop in targets.by_stratum_prop.items():
            if st.race != race:
                continue
            weight = prop / denom
            desired_within[st] = weight * r_total
            caps_within[st] = max(0, caps_by_stratum.get(st, 0))
        # allocate within race
        if desired_within:
            alloc_within = allocate_largest_remainder(desired_within, r_total, caps_within)
            for st, k in alloc_within.items():
                alloc[st] = alloc.get(st, 0) + k

    # If due to caps we under-allocated overall, spread remaining by global caps
    assigned = sum(alloc.values())
    if assigned < group_size:
        # residual desired = targets.by_stratum_prop scaled to remaining
        rem = group_size - assigned
        desired_global = {st: targets.by_stratum_prop[st] * rem for st in targets.by_stratum_prop}
        caps_left = {st: max(0, caps_by_stratum.get(st, 0) - alloc.get(st, 0)) for st in targets.by_stratum_prop}
        add = allocate_largest_remainder(desired_global, rem, caps_left)
        for st, k in add.items():
            alloc[st] = alloc.get(st, 0) + k

    return alloc

# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shortlist participants into 2 groups with demographic matching")
    parser.add_argument("--participants", required=True, help="Path to participants CSV")
    parser.add_argument("--targets", required=True, help="Path to targets CSV")
    parser.add_argument("--group-size", type=int, default=100, help="New picks per group (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", help="Updated participants CSV path (default: <input>.updated.csv)")
    parser.add_argument("--log", help="JSON log output path (default: <input>.shortlist_log.json)")
    parser.add_argument("--solver", choices=["baseline", "mip"], default="baseline", help="Use baseline greedy or MIP (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write updated CSV/log; print summary and optionally save selection JSON")
    parser.add_argument("--selection-out", help="If set in dry-run, write selected indices JSON here for external analysis")
    parser.add_argument("--print-variance", action="store_true", help="In dry-run, also print variance summary for both groups")
    parser.add_argument("--top-k", type=int, default=5, help="Top K under-represented categories to highlight (default 5)")

    args = parser.parse_args()

    rng = random.Random(args.seed)

    targets = load_targets(args.targets)

    rows = read_csv_dicts(args.participants)
    if not rows:
        raise SystemExit("Participants CSV is empty")

    participants, mapping = normalize_participants(rows, targets, seed=args.seed)

    # Build pools and pre-account per group
    pools = build_pools(participants, targets)

    pre_g1_strata, pre_g1_dim = preaccount_counts(participants, "1", targets)
    pre_g2_strata, pre_g2_dim = preaccount_counts(participants, "2", targets)

    # Caps per stratum is total eligible in pool
    caps: Dict[Stratum, int] = {st: len(pools.get(st, [])) for st in targets.by_stratum_prop.keys()}

    # Race-first allocation per group
    alloc_g1 = allocate_race_first(targets, args.group_size, pre_g1_dim, caps, rng)
    if sum(alloc_g1.values()) < args.group_size:
        # Fallback to global spill to fill remaining
        rem = args.group_size - sum(alloc_g1.values())
        desired_global = {st: targets.by_stratum_prop[st] * rem for st in targets.by_stratum_prop}
        caps_left = {st: max(0, caps.get(st, 0) - alloc_g1.get(st, 0)) for st in targets.by_stratum_prop}
        add = allocate_largest_remainder(desired_global, rem, caps_left)
        for st, k in add.items():
            alloc_g1[st] = alloc_g1.get(st, 0) + k
    # Reduce caps by what was given to g1 to avoid duplicate picks
    used_g1: Dict[Stratum, int] = defaultdict(int)
    for st, k in alloc_g1.items():
        used_g1[st] += k
        caps[st] = max(0, caps.get(st, 0) - k)

    alloc_g2 = allocate_race_first(targets, args.group_size, pre_g2_dim, caps, rng)
    if sum(alloc_g2.values()) < args.group_size:
        rem = args.group_size - sum(alloc_g2.values())
        desired_global = {st: targets.by_stratum_prop[st] * rem for st in targets.by_stratum_prop}
        caps_left = {st: max(0, caps.get(st, 0) - alloc_g2.get(st, 0)) for st in targets.by_stratum_prop}
        add = allocate_largest_remainder(desired_global, rem, caps_left)
        for st, k in add.items():
            alloc_g2[st] = alloc_g2.get(st, 0) + k

    if sum(alloc_g1.values()) < args.group_size or sum(alloc_g2.values()) < args.group_size:
        raise SystemExit("Insufficient eligible REGISTERED candidates to fill both groups as requested.")

    # Select specific participants
    pools_for_g1 = {st: list(pools.get(st, [])) for st in pools}
    picks_g1 = select_from_pools(pools_for_g1, alloc_g1, rng)

    # Remove picked from original pools
    selected_set = set(picks_g1)
    for st in list(pools.keys()):
        pools[st] = [idx for idx in pools[st] if idx not in selected_set]

    picks_g2 = select_from_pools(pools, alloc_g2, rng)

    # Greedy swap pass to reduce MAD vs adjusted marginals (based on NEW PICKS only)
    dim_targets_g1 = compute_adjusted_dim_targets(targets, args.group_size, pre_g1_dim)
    dim_targets_g2 = compute_adjusted_dim_targets(targets, args.group_size, pre_g2_dim)
    picks_g1, picks_g2 = greedy_swap(participants, picks_g1, picks_g2, dim_targets_g1, dim_targets_g2, rng)

    if args.dry_run:
        # Print concise summary
        print(f"Selected (dry-run): group1={len(picks_g1)}, group2={len(picks_g2)}")
        if args.print_variance:
            # Precompute supply of available REGISTERED adults by stratum for guidance
            supply_by_stratum: Dict[Stratum, int] = Counter()
            for p in participants:
                if p.eligible and p.status == STATUS_REGISTERED:
                    st = Stratum(p.sex, p.age_group, p.race, p.education_level)
                    supply_by_stratum[st] += 1
            # print variance directly
            for g_label, picks, dim_targets in [("1", picks_g1, dim_targets_g1), ("2", picks_g2, dim_targets_g2)]:
                n = len(picks)
                obs = group_dim_counts(participants, picks)
                print(f"Group {g_label}: N={n}")
                overall_mad_sum = 0.0
                overall_mad_dims = 0
                dim_scores = {}
                diffs_all = []
                for dim in ["sex", "age_group", "race", "education_level"]:
                    tgt_prop = targets.marginals_prop.get(dim, {})
                    # convert counts to proportions
                    if dim == "education_level":
                        cats = [c for c in sorted(tgt_prop.keys()) if c != "no_info"]
                    else:
                        cats = sorted(tgt_prop.keys())
                    sum_abs = 0.0
                    print(f"  {dim}:")
                    print("    category | target_% | observed_% | diff_% | abs_diff_%")
                    # Denominators for renormalization
                    if dim == "education_level":
                        obs_den = sum((obs.get(dim, Counter()).get(c, 0) for c in cats))
                        tgt_den = sum((tgt_prop.get(c, 0.0) for c in cats)) or 1.0
                    else:
                        obs_den = n
                        tgt_den = 1.0
                    for c in cats:
                        o = (obs.get(dim, Counter()).get(c, 0) / max(1, obs_den)) * 100.0
                        t = (tgt_prop.get(c, 0.0) / max(1e-12, tgt_den)) * 100.0
                        diff = o - t
                        diffs_all.append((dim, c, diff/100.0))
                        sum_abs += abs(diff) / 100.0
                        print(f"    {c} | {t:.3f} | {o:.3f} | {diff:.3f} | {abs(diff):.3f}")
                    mad = sum_abs / max(1, len(cats))
                    tvd = 0.5 * sum_abs
                    dim_scores[dim] = (mad, tvd)
                    overall_mad_sum += mad
                    overall_mad_dims += 1
                    print(f"    MAD={mad:.4f}, TVD={tvd:.4f}")
                overall_mad = overall_mad_sum / max(1, overall_mad_dims)
                print(f"  overall MAD={overall_mad:.4f} (~{overall_mad*100:.2f}%)")
                # summary and focus
                diffs_all.sort(key=lambda x: x[2])
                under = diffs_all[: args.top_k]
                over = diffs_all[-args.top_k :][::-1]
                print("  Summary:")
                for dim in ["sex", "age_group", "race", "education_level"]:
                    mad, tvd = dim_scores[dim]
                    print(f"    {dim}: MAD={mad:.4f}, TVD={tvd:.4f}")
                print(f"    overall MAD={overall_mad:.4f} (~{overall_mad*100:.2f}%)")
                print("  Focus next picks on (under-represented):")
                for dim, cat, d in under:
                    print(f"    {dim}={cat} (deficit {abs(d)*100:.2f}% ~ {abs(d)*n:.0f} ppl)")

                # Full strata guidance
                obs_counts_stratum: Dict[Stratum, int] = Counter()
                for idx in picks:
                    p = participants[idx]
                    st = Stratum(p.sex, p.age_group, p.race, p.education_level)
                    obs_counts_stratum[st] += 1
                obs_prop_stratum = {st: c / n for st, c in obs_counts_stratum.items()}

                strata_diffs: List[Tuple[Stratum, float]] = []
                for st, tprop in targets.by_stratum_prop.items():
                    oprop = obs_prop_stratum.get(st, 0.0)
                    diff = oprop - tprop
                    strata_diffs.append((st, diff))
                strata_diffs.sort(key=lambda x: x[1])

                print("  Focus by strata (full combination):")
                count = 0
                for st, diff in strata_diffs:
                    if diff < 0 and count < args.top_k:
                        deficit_pct = -diff * 100.0
                        deficit_ppl = max(0, round(-diff * n))
                        supply = supply_by_stratum.get(st, 0)
                        print(f"    sex={st.sex} | age_group={st.age_group} | race={st.race} | education={st.education_level} -> deficit {deficit_pct:.2f}% ~ {deficit_ppl} ppl (available REGISTERED: {supply})")
                        count += 1
        # optional save selection
        if args.selection_out:
            sel = {"group1_indices": picks_g1, "group2_indices": picks_g2}
            with open(args.selection_out, "w", encoding="utf-8") as f:
                json.dump(sel, f)
            print(f"Wrote selection JSON: {os.path.abspath(args.selection_out)}")
        return

    # Apply updates
    status_col = mapping.get("status", "Status")
    group_col = mapping.get("group", "Group")

    updated_rows = list(rows)
    for idx in picks_g1:
        updated_rows[idx][status_col] = STATUS_TO_CONTACT
        updated_rows[idx][group_col] = "1"
    for idx in picks_g2:
        updated_rows[idx][status_col] = STATUS_TO_CONTACT
        updated_rows[idx][group_col] = "2"

    # Write outputs
    output_path = args.output or f"{os.path.splitext(args.participants)[0]}.updated.csv"
    fieldnames = ensure_status_group_headers(list(rows[0].keys()), mapping)
    write_csv_dicts(output_path, updated_rows, fieldnames)

    # Log summary
    def alloc_to_dict(alloc: Dict[Stratum, int]) -> Dict[str, int]:
        return {f"{st.sex}|{st.age_group}|{st.race}|{st.education_level}": n for st, n in alloc.items() if n}

    caps_dict = {f"{st.sex}|{st.age_group}|{st.race}|{st.education_level}": caps for st, caps in {**{st: len(build_pools(participants, targets).get(st, [])) for st in targets.by_stratum_prop},}.items()}

    log = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "group_size": args.group_size,
        "participants_input": os.path.abspath(args.participants),
        "targets_input": os.path.abspath(args.targets),
        "selected": {
            "group1": len(picks_g1),
            "group2": len(picks_g2),
        },
        "allocations": {
            "group1": alloc_to_dict(alloc_g1),
            "group2": alloc_to_dict(alloc_g2),
        },
        "caps": caps_dict,
        "preaccount_counts": {
            "group1": {k: v for k, v in pre_g1_dim.items()},
            "group2": {k: v for k, v in pre_g2_dim.items()},
        },
    }

    log_path = args.log or f"{os.path.splitext(args.participants)[0]}.shortlist_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=lambda o: dict(o))

    print(f"Wrote updated CSV: {output_path}")
    print(f"Wrote log: {log_path}")


if __name__ == "__main__":
    main()
