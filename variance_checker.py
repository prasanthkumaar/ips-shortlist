import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from matching_utils import (
    load_targets,
    normalize_participants,
    Stratum,
    read_csv_dicts,
    STATUS_REGISTERED,
)


def compute_group_memberships(participants):
    g_members = {"1": [], "2": []}
    for i, p in enumerate(participants):
        if p.group in {"1", "2"}:
            g_members[p.group].append(i)
    return g_members


def proportion_table(counts: Counter) -> Dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / total for k, v in counts.items()}


def observed_marginals(participants, indices: List[int]) -> Dict[str, Counter]:
    counts = {
        "sex": Counter(),
        "age_group": Counter(),
        "race": Counter(),
        "education_level": Counter(),
    }
    for idx in indices:
        p = participants[idx]
        counts["sex"][p.sex] += 1
        counts["age_group"][p.age_group] += 1
        counts["race"][p.race] += 1
        counts["education_level"][p.education_level] += 1
    return counts


def write_variance_csv(path: str, headers: List[str], rows: List[List]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Compute variance vs targets for shortlisting groups")
    parser.add_argument("--participants", required=True, help="Path to updated or source participants CSV")
    parser.add_argument("--targets", required=True, help="Path to targets CSV")
    parser.add_argument("--selection-json", help="Optional JSON with group1_indices and group2_indices from shortlist dry-run")
    parser.add_argument("--outdir", help="Output directory for variance CSVs (used only with --save)")
    parser.add_argument("--save", action="store_true", help="Also save variance CSVs (default: print only)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of most under-represented categories to highlight (default 5)")
    parser.add_argument("--strata-top-k", type=int, default=10, help="Number of most under-represented full strata to highlight (default 10)")

    args = parser.parse_args()

    targets = load_targets(args.targets)
    rows = read_csv_dicts(args.participants)
    if not rows:
        raise SystemExit("Participants CSV is empty")

    participants, mapping = normalize_participants(rows, targets)

    if args.selection_json:
        with open(args.selection_json, "r", encoding="utf-8") as f:
            sel = json.load(f)
        groups = {
            "1": sel.get("group1_indices", []),
            "2": sel.get("group2_indices", []),
        }
    else:
        groups = compute_group_memberships(participants)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.participants))
    if args.save:
        os.makedirs(outdir, exist_ok=True)

    # Precompute supply of available REGISTERED adults by stratum for guidance
    supply_by_stratum: Dict[Stratum, int] = Counter()
    for p in participants:
        if p.eligible and p.status == STATUS_REGISTERED:
            st = Stratum(p.sex, p.age_group, p.race, p.education_level)
            supply_by_stratum[st] += 1

    for g in ["1", "2"]:
        indices = groups.get(g, [])
        if not indices:
            print(f"Group {g}: no members found")
            continue
        n = len(indices)
        obs = observed_marginals(participants, indices)

        print(f"Group {g}: N={n}")
        # For each dimension, build variance table
        overall_mad_sum = 0.0
        overall_mad_dims = 0
        dim_scores = {}
        for dim in ["sex", "age_group", "race", "education_level"]:
            tgt_prop = targets.marginals_prop.get(dim, {})
            obs_prop = proportion_table(obs[dim])
            if dim == "education_level":
                cats = [c for c in sorted(tgt_prop.keys()) if c != "no_info"]
            else:
                cats = sorted(set(tgt_prop.keys()) | set(obs_prop.keys()))
            rows_out: List[List] = []
            sum_abs = 0.0
            # Renormalize when excluding no_info
            if dim == "education_level":
                obs_den = sum(obs_prop.get(c, 0.0) for c in cats) or 1.0
                tgt_den = sum(tgt_prop.get(c, 0.0) for c in cats) or 1.0
            else:
                obs_den = 1.0
                tgt_den = 1.0
            for c in cats:
                o = (obs_prop.get(c, 0.0) / obs_den)
                t = (tgt_prop.get(c, 0.0) / tgt_den)
                diff = o - t
                sum_abs += abs(diff)
                rows_out.append([c, round(t * 100, 3), round(o * 100, 3), round(diff * 100, 3), round(abs(diff) * 100, 3)])
            mad = sum_abs / max(1, len(cats))
            tvd = 0.5 * sum_abs
            overall_mad_sum += mad
            overall_mad_dims += 1
            dim_scores[dim] = (mad, tvd)

            print(f"  {dim}: MAD={mad:.4f}, TVD={tvd:.4f}")
            print("    category | target_% | observed_% | diff_% | abs_diff_%")
            for row in rows_out:
                print(f"    {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")

            if args.save:
                csv_path = os.path.join(outdir, f"variance_group{g}_{dim}.csv")
                write_variance_csv(csv_path, ["category", "target_%", "observed_%", "diff_%", "abs_diff_%"], rows_out)
                print(f"    saved: {csv_path}")

        overall_mad = overall_mad_sum / max(1, overall_mad_dims)
        print(f"  overall MAD={overall_mad:.4f} (~{overall_mad*100:.2f}%)")

        # Top under/over across all categories combined
        diffs: List[Tuple[str, str, float]] = []  # (dim, cat, diff)
        for dim in ["sex", "age_group", "race", "education_level"]:
            tgt_prop = targets.marginals_prop.get(dim, {})
            obs_prop = proportion_table(obs[dim])
            if dim == "education_level":
                cats = [c for c in sorted(tgt_prop.keys()) if c != "no_info"]
                obs_den = sum(obs_prop.get(c, 0.0) for c in cats) or 1.0
                tgt_den = sum(tgt_prop.get(c, 0.0) for c in cats) or 1.0
            else:
                cats = sorted(set(tgt_prop.keys()) | set(obs_prop.keys()))
                obs_den = 1.0
                tgt_den = 1.0
            for c in cats:
                diffs.append((dim, c, (obs_prop.get(c, 0.0) / obs_den) - (tgt_prop.get(c, 0.0) / tgt_den)))
        diffs.sort(key=lambda x: x[2])
        under = diffs[: args.top_k]
        over = diffs[-args.top_k :][::-1]
        print("  Top under-represented:")
        for dim, cat, d in under:
            print(f"    {dim}={cat}: {d*100:.2f}% (~{-d*n:.0f} ppl)")
        print("  Top over-represented:")
        for dim, cat, d in over:
            print(f"    {dim}={cat}: {d*100:.2f}% (~{d*n:.0f} ppl)")

        # Summary block
        print("  Summary:")
        for dim in ["sex", "age_group", "race", "education_level"]:
            mad, tvd = dim_scores[dim]
            print(f"    {dim}: MAD={mad:.4f}, TVD={tvd:.4f}")
        print(f"    overall MAD={overall_mad:.4f} (~{overall_mad*100:.2f}%)")
        print("  Focus next picks on (under-represented):")
        for dim, cat, d in under:
            print(f"    {dim}={cat} (deficit {abs(d)*100:.2f}% ~ {abs(d)*n:.0f} ppl)")

        # Full strata guidance (education excludes no_info implicitly via supply and obs)
        obs_counts_stratum: Dict[Stratum, int] = Counter()
        for idx in indices:
            p = participants[idx]
            st = Stratum(p.sex, p.age_group, p.race, p.education_level)
            obs_counts_stratum[st] += 1
        obs_prop_stratum = {st: c / n for st, c in obs_counts_stratum.items()}

        strata_diffs: List[Tuple[Stratum, float]] = []  # (stratum, diff)
        for st, tprop in targets.by_stratum_prop.items():
            if st.education_level == "no_info":
                continue
            oprop = obs_prop_stratum.get(st, 0.0)
            diff = oprop - tprop
            strata_diffs.append((st, diff))
        strata_diffs.sort(key=lambda x: x[1])

        print("  Focus by strata (full combination):")
        count = 0
        for st, diff in strata_diffs:
            if diff < 0 and count < args.strata_top_k:
                deficit_pct = -diff * 100.0
                deficit_ppl = max(0, round(-diff * n))
                supply = supply_by_stratum.get(st, 0)
                print(f"    sex={st.sex} | age_group={st.age_group} | race={st.race} | education={st.education_level} -> deficit {deficit_pct:.2f}% ~ {deficit_ppl} ppl (available REGISTERED: {supply})")
                count += 1


if __name__ == "__main__":
    main() 