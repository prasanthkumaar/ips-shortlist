# IPS Shortlist

This repo shortlists participants into groups while matching Singapore demographics (sex, age_group, race, education_level), adds an OS ratio constraint (iOS vs Android), and reports variance to targets.

## Data files

- `final_2024_agegroup_sex_race_education.csv`: target counts with header `age_group,sex,race,education_level,count`
- `data/GovernmentCallStudyAug2025_2025-08-11.csv`: source participants

## Column headers (locked)

Participants CSV must include these headers:

- `UIN`, `Name`, `Gender`, `Race`, `DOB`, `Education`, `Number`, `OS`, `Group`, `Status`, `Preferred language of communication`

Notes:

- `Group` must exist as a column but may be blank prior to assignment; new picks will be written with configured labels.
- `OS` values are strictly validated:
  - iOS: exactly `Apple`
  - Android: exactly `Android (e.g., Samsung, Google, Oppo, Xiaomi, Huawei)`
  - Any other value (including lowercase or variants) will cause an error.
- `Preferred language of communication` values must be exactly one of: `Chinese`, `English`. Any other value will cause an error.

## Global rules

- Only shortlist participants aged ≥ 18 (DOB parsed; age_group is bucketed from targets’ labels)
- Only shortlist rows where `Status == REGISTERED`
- Language eligibility: only shortlist rows where `Preferred language of communication` is included in `config.SPEAKING_LANGUAGES_ELIGIBLE`. If a `REGISTERED` row has a non-eligible language, it will be marked in the output as `INELIGIBLE: [Preferred language of communication] is [<value>]` and excluded from selection.
- Pre-account existing rows:
  - `Status == CONFIRMED` → locked-in toward quotas (kept in their `Group` if present)
  - `Status == TO_CONTACT` or `CONTACTED` → pending; excluded from new selection but counted toward current composition (by `Group` if present)
- Special: education for 18–24 is set to `no_info`
- Device OS target share is fixed at 30% iOS / 70% Android across the final combined sample (existing + new picks), and group assignment also balances each group toward 30/70.
- All rows must include a non-empty `UIN`. The script errors if the `UIN` column is missing or any row has an empty `UIN`.
- Every row must have a non-empty `UIN` and `Status`; the script errors if either is missing.
- Duplicate check: if the same `UIN` appears more than once in the participants CSV, the script errors and lists duplicate UINs (no automatic merge).
- Blacklist handling: the script scans the `blacklist/` folder and does not abort.
  - Any `REGISTERED` participant whose UIN is found in any blacklist CSV will be marked in the output as `Status = BLACKLISTED: Found in {csvfile(s)}` and is excluded from selection.
  - Other statuses are left untouched.

## Dynamic groups (config.py)

- Configure any number of groups and their labels/totals in `config.py`:

```python
GROUP_LABELS = [
    "Group 1",
    "Group 2",
    # add more, e.g., "Group 3"
]

# Required per-group totals (including existing). Every label in GROUP_LABELS must have a total.
GROUP_TOTALS = {
    "Group 1": 120,
    "Group 2": 80,
    # "Group 3": 50,
}
```

- The `Group` column in the participants CSV can contain either the configured label (case-insensitive) or a 1-based index (e.g., `1` for `Group 1`). If blank or absent, it will be added during output for new picks.
- The shortlist writes the configured label string into the `Group` column for new picks.

## Language eligibility (config.py)

- Configure eligible speaking languages for shortlisting. Example:

```python
SPEAKING_LANGUAGES_ELIGIBLE = [
    "English",
]
```

- The participants CSV currently supports only `Chinese` and `English`. Any other value will cause a validation error.
- During output, any `REGISTERED` row with a non-eligible value is marked as:
  - `INELIGIBLE: [Preferred language of communication] is [<value>]`

## Blacklist

- Location: a folder named `blacklist` next to `shortlist.py` (i.e., in the repo root). The folder must exist; otherwise the script errors.
- Files: every `.csv` file in `blacklist/` is treated as a blacklist source and must include a `UIN` column. If any CSV is missing `UIN`, the script errors.
- Behavior:
  - The script prints which blacklist files were scanned, e.g., `Scanned blacklist files: studyA.csv, studyB.csv`. If the folder exists but contains no CSVs, it prints: `No blacklist CSV files found in 'blacklist' folder.`
  - If any participant UIN in the input participants CSV appears in any blacklist CSV and has `Status == REGISTERED`, the participant is excluded from selection and will be marked in the output as `Status = BLACKLISTED: Found in {csvfile(s)}`.

## Install

Python 3.9+ recommended.

## Usage

Set paths and group config in `config.py`, then run:

1. Shortlist (uses config paths and seed)

```bash
python3 shortlist.py
```

- Output: `<participants>.updated.csv`
- The script prints a one-line command for the variance checker with absolute paths; you can run it directly or the simple command below.
- If any `REGISTERED` participant is blacklisted, they will be marked as `BLACKLISTED: Found in {csvfile(s)}` in the output.

2. Variance report (uses config paths)

```bash
python3 variance_checker.py
```

## What the shortlist does

- Fixed column detection for headers: `Gender`, `Race`, `Education`, `DOB`, `Status`, `Group`, `OS`, `Preferred language of communication`
- Value normalization for sex/race/education and OS (iOS vs Android)
- Buckets ages using age group labels found in the targets CSV
- Pre-accounts existing `CONFIRMED`/`TO_CONTACT`/`CONTACTED`:
  - Uses their existing `Group` (if present) to seed each group’s composition
  - Excludes them from new picks
- Computes per-group “top-up” needed to reach the specified totals from `config.py`.
- Selects new participants from `REGISTERED`:
  - First, fills 4D deficits (sex × age_group × race × education_level)
  - Then, scores remaining candidates by marginal improvement vs targets, with an additional bias to hit the overall 30% iOS / 70% Android ratio across the combined sample
- Assigns new picks to configured groups to minimize overall variance and keep groups similar:
  - Objective includes 4D demographics and Device OS; also applies a balance penalty between groups
  - Performs a swap pass over new picks only to further lower and balance variance
- Writes the updated CSV:
  - New picks: `Status=TO_CONTACT` and configured group label in `Group`
  - Blacklisted REGISTERED participants: `Status=BLACKLISTED: Found in {csvfile(s)}`

## Notes

- Random tie-breaks are seeded via `config.SEED` for reproducibility (no CLI flags).
- If groups diverge in score, you can increase balancing by editing in `shortlist.py`:
  - `BALANCE_LAMBDA` (default 0.5)
  - `SWAP_TRIES_PER_ITER` and `MAX_SWAP_ITERS`

## Outputs

- `*.updated.csv`: participants with new picks marked `TO_CONTACT` and assigned to the configured group labels
- Variance report: prints per-group proportions vs targets with overall variance (% MAD), plus a separate “Device OS” section (30% iOS / 70% Android)
