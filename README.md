# IPS Shortlist

This repo shortlists participants into groups while matching Singapore demographics (sex, age_group, race, education_level), adds an OS ratio constraint (iOS vs Android), and reports variance to targets. **Features detailed selection reasons and fully configurable scoring weights.**

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
- `Preferred language of communication` values must be exactly one of: `Mandarin/Chinese`, `English`. Any other value will cause an error.

## Global rules

- Only shortlist participants aged ≥ 18 (DOB parsed; age_group is bucketed from targets' labels)
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

- The participants CSV currently supports only `Mandarin/Chinese` and `English`. Any other value will cause a validation error.
- During output, any `REGISTERED` row with a non-eligible value is marked as:
  - `INELIGIBLE: [Preferred language of communication] is [<value>]`

## **Configurable scoring weights (config.py)**

**NEW:** The selection algorithm now uses fully configurable scoring weights that you can adjust to prioritize different demographic goals:

```python
# Exact 4D cell deficit bonus
CELL_DEFICIT_BONUS = 3.0              # Bonus for exact sex/age/race/education deficits

# Special bonus for underrepresented races
UNDERREPRESENTED_RACE_BONUS = 2.0     # Extra flat bonus for minority races
UNDERREPRESENTED_RACES = ['Malay', 'Indian', 'Others']  # Races that get the bonus

# Marginal demographic weights
SEX_WEIGHT = 1.0          # Weight for sex (Male/Female) marginal targets
AGE_WEIGHT = 1.0          # Weight for age group marginal targets
RACE_WEIGHT = 1.0         # Weight for race marginal targets
EDUCATION_WEIGHT = 1.0    # Weight for education level marginal targets

# Device OS balance weights
OS_BONUS = 0.5           # Bonus for participants with needed OS type
OS_PENALTY = -0.5        # Penalty for participants with over-represented OS type

# Random tie-breaker
RANDOM_TIE_BREAKER_MAX = 0.01  # Small random value range for tie-breaking
```

### **Tuning recommendations:**

**To prioritize racial diversity over OS balance:**

- Increase `RACE_WEIGHT` from 1.0 to 1.5+
- Increase `UNDERREPRESENTED_RACE_BONUS` from 2.0 to 3.0+
- Reduce `OS_BONUS/OS_PENALTY` magnitude to ±0.25

**To fill specific demographic gaps more aggressively:**

- Increase `CELL_DEFICIT_BONUS` from 3.0 to 5.0+

**To better balance across all demographics:**

- Set all marginal weights equal: `SEX_WEIGHT = AGE_WEIGHT = RACE_WEIGHT = EDUCATION_WEIGHT = 1.0`

## **Detailed selection reasons**

**NEW:** The output CSV now includes a `Selection Reason` column with detailed explanations of why each participant was or wasn't selected, including score breakdowns:

### Examples:

**Selected participant:**

```
Selected (2nd pass) - ranked 24/301 (score: 2.559) for Female/50-54/Indian/Secondary.
Breakdown: 4D-cell deficit bonus: +0.0 | Underrepresented race bonus (Indian): +2.0 |
Sex (Female): +0.518 | Age (50-54): +0.088 | Race (Indian): +0.131 |
Education (Secondary): +0.068 | OS (iOS): -0.25 | Random: +0.003 -> assigned to Group 4
```

**Not selected participant:**

```
Not selected - ranked 145/301 (score: 2.616) for Male/20-24/Others/no_info, below cutoff.
Breakdown: 4D-cell deficit bonus: +0.0 | Race (Others): +0.017 |
OS (Android): +2.0 (need more Android, deficit: 75) | Random: +0.009
```

**Other reason examples:**

- `Blacklisted - found in phase1-uin.csv`
- `Not selected - preferred language 'Mandarin/Chinese' not eligible (only English allowed)`
- `Already confirmed` / `Already marked to contact`
- `Not selected - age 17 below 18`

## Blacklist

- Location: a folder named `blacklist` next to `shortlist.py` (i.e., in the repo root). The folder must exist; otherwise the script errors.
- Files: every `.csv` file in `blacklist/` is treated as a blacklist source and must include a `UIN` column. If any CSV is missing `UIN`, the script errors.
- Behavior:
  - The script prints which blacklist files were scanned, e.g., `Scanned blacklist files: studyA.csv, studyB.csv`. If the folder exists but contains no CSVs, it prints: `No blacklist CSV files found in 'blacklist' folder.`
  - If any participant UIN in the input participants CSV appears in any blacklist CSV and has `Status == REGISTERED`, the participant is excluded from selection and will be marked in the output as `Status = BLACKLISTED: Found in {csvfile(s)}`.

## Install

Python 3.9+ recommended.

## Usage

Set paths, group config, and **scoring weights** in `config.py`, then run:

1. Shortlist (uses config paths and seed)

```bash
python3 shortlist.py
```

- Output: `<participants>.updated.csv` **with detailed `Selection Reason` column**
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
  - Uses their existing `Group` (if present) to seed each group's composition
  - Excludes them from new picks
- Computes per-group "top-up" needed to reach the specified totals from `config.py`.
- **Selects new participants from `REGISTERED` using configurable scoring:**
  - **First pass:** fills exact 4D deficits (sex × age_group × race × education_level) with `CELL_DEFICIT_BONUS`
  - **Second pass:** scores remaining candidates by:
    - Marginal demographic contributions (weighted by `SEX_WEIGHT`, `AGE_WEIGHT`, `RACE_WEIGHT`, `EDUCATION_WEIGHT`)
    - Special `UNDERREPRESENTED_RACE_BONUS` for minority races
    - OS balance bias toward 30% iOS / 70% Android (weighted by `OS_BONUS`/`OS_PENALTY`)
    - Small random tie-breaker
- Assigns new picks to configured groups to minimize overall variance and keep groups similar:
  - Objective includes 4D demographics and Device OS; also applies a balance penalty between groups
  - Performs a swap pass over new picks only to further lower and balance variance
- Writes the updated CSV:
  - New picks: `Status=TO_CONTACT` and configured group label in `Group`
  - Blacklisted REGISTERED participants: `Status=BLACKLISTED: Found in {csvfile(s)}`
  - **All participants get detailed `Selection Reason` explaining their selection/rejection**

## Notes

- Random tie-breaks are seeded via `config.SEED` for reproducibility (no CLI flags).
- If groups diverge in score, you can increase balancing by editing in `shortlist.py`:
  - `BALANCE_LAMBDA` (default 0.5)
  - `SWAP_TRIES_PER_ITER` and `MAX_SWAP_ITERS`
- **Use the `Selection Reason` column to debug and understand selection decisions**
- **Adjust scoring weights in `config.py` to prioritize different demographic goals**

## Outputs

- `*.updated.csv`: participants with new picks marked `TO_CONTACT` and assigned to the configured group labels, **plus detailed `Selection Reason` column**
- Variance report: prints per-group proportions vs targets with overall variance (% MAD), plus a separate "Device OS" section (30% iOS / 70% Android)
