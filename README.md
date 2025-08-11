# IPS Shortlist

This repo shortlists participants into two groups while matching Singapore demographics (sex, age_group, race, education_level), adds an OS ratio constraint (iOS vs Android), and reports variance to targets.

## Data files

- `final_2024_agegroup_sex_race_education.csv`: target counts with header `age_group,sex,race,education_level,count`
- `data/GovernmentCallStudyAug2025_2025-08-11.csv`: source participants

## Column headers (locked)

Participants CSV must include these exact headers:

- `Name`, `Gender`, `Race`, `DOB`, `Education`, `Number`, `OS`, `Group`, `Status`

Notes:

- `OS` values are normalized to two categories:
  - iOS: any value containing `Apple` (or `iOS`)
  - Android: any value containing `Android` or brands like `Samsung`, `Google`, `Oppo`, `Xiaomi`, `Huawei`, `OnePlus`, `Vivo`, `Realme`, `Sony`, `Motorola`, `Nothing`
- Unknown OS values default to Android. If you want strict validation instead (exclude unknowns), update `normalize_os`.

## Global rules

- Only shortlist participants aged ≥ 18 (DOB parsed; age_group is bucketed from targets’ labels)
- Only shortlist rows where `Status == REGISTERED`
- Pre-account existing rows:
  - `Status == CONFIRMED` → locked-in toward quotas (kept in their `Group` if present)
  - `Status == TO_CONTACT` or `CONTACTED` → pending; excluded from new selection but counted toward current composition (by `Group` if present)
- Special: education for 18–24 is set to `no_info`
- Device OS target share is fixed at 30% iOS / 70% Android across the final combined sample (existing + new picks), and group assignment also balances each group toward 30/70.

## Install

Python 3.9+ recommended.

## Usage (relative paths)

1. Shortlist (creates/overwrites updated CSV and prints variance command)

```bash
python3 shortlist.py \
  --participants data/GovernmentCallStudyAug2025_2025-08-11.csv \
  --targets final_2024_agegroup_sex_race_education.csv \
  --seed 42 --per_group 100
```

- Output: `data/GovernmentCallStudyAug2025_2025-08-11.updated.csv`
- The script prints a one-line command to run the variance report (it uses absolute paths for convenience; you can run an equivalent relative-path command shown below).

2. Variance report (per group, with overall variance %; lower is better)

```bash
python3 variance_checker.py \
  --participants data/GovernmentCallStudyAug2025_2025-08-11.updated.csv \
  --targets final_2024_agegroup_sex_race_education.csv
```

## What the shortlist does

- Fixed column detection for headers: `Gender`, `Race`, `Education`, `DOB`, `Status`, `Group`, `OS`
- Value normalization for sex/race/education and OS (iOS vs Android)
- Buckets ages using age group labels found in the targets CSV
- Pre-accounts existing `CONFIRMED`/`TO_CONTACT`/`CONTACTED`:
  - Uses their existing `Group` (if present) to seed each group’s composition
  - Excludes them from new picks
- Computes per-group “top-up” needed (to reach `--per_group` each). If some participants later reject and statuses change, re-running the shortlist tops up from remaining `REGISTERED`.
- Selects new participants from `REGISTERED`:
  - First, fills 4D deficits (sex × age_group × race × education_level)
  - Then, scores remaining candidates by marginal improvement vs targets, with an additional bias to hit the overall 30% iOS / 70% Android ratio across the combined sample
- Assigns new picks to `Group` 1 or 2 to minimize overall variance and keep groups similar:
  - Objective includes 4D demographics and Device OS; also applies a balance penalty between groups
  - Performs a swap pass over new picks only to further lower and balance variance
- Writes the updated CSV with `Status=TO_CONTACT` and `Group∈{1,2}` for new picks only

## Re-running after updates (top-up)

- If some `TO_CONTACT`/`CONTACTED` are rejected and their status changes (e.g., removed or set not REGISTERED), simply re-run the shortlist command.
- The script will recompute per-group deficits, pre-account current `CONFIRMED`/`TO_CONTACT`/`CONTACTED` (with their groups), and top up from remaining `REGISTERED` while keeping variance low and similar across groups.

## Notes

- Random tie-breaks are seeded (`--seed`) for reproducibility.
- If groups diverge in score, you can increase balancing by editing in `shortlist.py`:
  - `BALANCE_LAMBDA` (default 0.5)
  - `SWAP_TRIES_PER_ITER` and `MAX_SWAP_ITERS`

## Outputs

- `*.updated.csv`: participants with new picks marked `TO_CONTACT` and assigned `Group` 1 or 2
- Variance report: prints per-group proportions vs targets with overall variance (% MAD), plus a separate “Device OS” section (30% iOS / 70% Android)
