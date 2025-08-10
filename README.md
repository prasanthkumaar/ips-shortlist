# IPS Shortlist

This repo shortlists participants into two groups while matching Singapore demographics (sex, age_group, race, education_level), and reports variance to targets.

## Data files

- `final_2024_agegroup_sex_race_education.csv`: target counts with header `age_group,sex,race,education_level,count`
- `data/GovernmentCallStudyAug2025_2025-08-09.csv`: source participants

## Global rules

- Only shortlist participants aged ≥ 18 (DOB parsed; age_group is bucketed from targets’ labels)
- Only shortlist rows where `Status == REGISTERED`
- Pre-account existing rows:
  - `Status == CONFIRMED` → locked-in toward quotas (kept in their `Group` if present)
  - `Status == TO_CONTACT` → pending; excluded from new selection but counted toward current composition (by `Group` if present)
- Special: education for 18–24 is set to `no_info`

## Install

Python 3.9+ recommended.

## Usage

1. Shortlist (creates/overwrites updated CSV and prints variance command)

```bash
python3 shortlist.py \
  --participants "/absolute/path/to/data/GovernmentCallStudyAug2025_2025-08-09.csv" \
  --targets "/absolute/path/to/final_2024_agegroup_sex_race_education.csv" \
  --seed 42 --per_group 100
```

- Output: `data/GovernmentCallStudyAug2025_2025-08-09.updated.csv`
- The script prints a one-line command to run the variance report.

2. Variance report (per group, with overall variance %; lower is better)

```bash
python3 variance_checker.py \
  --participants "/absolute/path/to/data/GovernmentCallStudyAug2025_2025-08-09.updated.csv" \
  --targets "/absolute/path/to/final_2024_agegroup_sex_race_education.csv"
```

## What the shortlist does

- Fuzzy column detection for headers: sex, race, education_level, DOB, Status, Group
- Value normalization for sex/race/education
- Buckets ages using age group labels found in the targets CSV
- Pre-accounts existing `CONFIRMED`/`TO_CONTACT`:
  - Uses their existing `Group` (if present) to seed each group’s composition
  - Excludes them from new picks
- Computes per-group “top-up” needed (to reach `--per_group` each). If some participants later reject and statuses change, re-running the shortlist tops up from remaining `REGISTERED`.
- Selects new participants from `REGISTERED`:
  - First, fills 4D deficits (sex × age_group × race × education_level)
  - Then, scores remaining candidates by marginal improvement vs targets
- Assigns new picks to `Group` 1 or 2 to minimize overall variance and keep groups similar:
  - Objective: sum of group variances plus a balance penalty between groups
  - Performs a swap pass over new picks only to further lower and balance variance
- Writes the updated CSV with `Status=TO_CONTACT` and `Group∈{1,2}` for new picks only

## Re-running after updates (top-up)

- If some `TO_CONTACT` are rejected and their status changes (e.g., removed or set not REGISTERED), simply re-run the shortlist command.
- The script will recompute per-group deficits, pre-account current `CONFIRMED`/`TO_CONTACT` (with their groups), and top up from remaining `REGISTERED` while keeping variance low and similar across groups.

## Notes

- Random tie-breaks are seeded (`--seed`) for reproducibility.
- If groups diverge in score, you can increase balancing by editing in `shortlist.py`:
  - `BALANCE_LAMBDA` (default 0.5)
  - `SWAP_TRIES_PER_ITER` and `MAX_SWAP_ITERS`

## Outputs

- `*.updated.csv`: participants with new picks marked `TO_CONTACT` and assigned `Group` 1 or 2
- Variance report: prints per-group proportions vs targets with overall variance (% MAD)
