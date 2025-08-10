# IPS Shortlisting Toolkit

This repo shortlists participants into 2 groups of N each while matching Singapore’s demographics by sex, age_group, race, and education_level. It also prints detailed variance diagnostics to the terminal.

## Quick start

- Python 3.9+
- Files:
  - `data/GovernmentCallStudyAug2025_2025-08-09.csv` (participants)
  - `final_2024_agegroup_sex_race_education.csv` (targets)

### Dry-run (console only, no file writes)

This selects 100 per group (default) and prints variance + focus guidance to the terminal. It does not modify your CSV.

```bash
python3 shortlist.py \
  --participants data/GovernmentCallStudyAug2025_2025-08-09.csv \
  --targets final_2024_agegroup_sex_race_education.csv \
  --seed 123 \
  --dry-run \
  --print-variance \
  --top-k 8
```

What you’ll see per group (in the terminal):

- Per-dimension tables (sex, age_group, race, education_level)
  - target*% vs observed*% vs diff*% and abs_diff*%
  - MAD, TVD per dimension
- overall MAD (and as a percent)
- Focus next picks on: top-K under-represented categories with approx headcount
- Focus by strata (full combination sex×age_group×race×education) with deficits and available REGISTERED supply

Notes:

- Education `no_info` is treated as unknown (not a target bucket). Variance tables for education exclude `no_info` and renormalize the remaining categories.
- Reproducible selection via `--seed`.

### Write an updated CSV (optional)

If you are satisfied with the dry-run, you can write the shortlist to a new CSV. By default, the original file is NOT overwritten.

```bash
python3 shortlist.py \
  --participants data/GovernmentCallStudyAug2025_2025-08-09.csv \
  --targets final_2024_agegroup_sex_race_education.csv \
  --seed 123
```

Outputs:

- `data/GovernmentCallStudyAug2025_2025-08-09.updated.csv` (new picks have `Status=TO_CONTACT`, `Group=1/2`)
- `data/GovernmentCallStudyAug2025_2025-08-09.shortlist_log.json` (allocation summary)

Safety:

- To overwrite the working file, pass `--output data/GovernmentCallStudyAug2025_2025-08-09.csv` (not recommended until you’ve reviewed a dry-run).

### Variance checker (optional)

You can run the checker against either:

- An updated participants CSV (after running shortlist without `--dry-run`), or
- The source CSV that already has `Group` assignments.

```bash
python3 variance_checker.py \
  --participants data/GovernmentCallStudyAug2025_2025-08-09.csv \
  --targets final_2024_agegroup_sex_race_education.csv \
  --top-k 5 \
  --strata-top-k 10
```

It prints the same per-dimension variance tables (MAD/TVD), overall MAD%, single-dimension focus, and full strata focus with supply. Pass `--save` to also write CSVs (optional).

## How to interpret the printouts

- target\_%: Population share (from targets) for a category
- observed\_%: Share in your selected group
- diff\_% = observed − target (positive = over-represented; negative = under-represented)
- abs*diff*%: magnitude of deviation
- MAD (Mean Absolute Deviation): average abs(diff) across categories in a dimension (0 is perfect)
- TVD (Total Variation Distance): 0.5 × sum abs(diff) in a dimension; with N=100, TVD×100 ≈ minimal people to reassign
- overall MAD: average of the four dimension MADs; printed also as a percentage
- Focus next picks on: prioritized under-represented categories (with approximate headcount deficits)
- Focus by strata: prioritized under-represented full combinations sex×age_group×race×education with available REGISTERED supply in your pool

Education `no_info` handling:

- `no_info` is excluded from education variance and summary (renormalized over known educations). This treats missing education as unknown rather than a target bucket.

## Rules and normalization

- Eligibility: age ≥ 18, `Status=REGISTERED`
- Pre-accounting: existing `Status=CONFIRMED` and `Status=TO_CONTACT` count towards quotas (by group)
- Header fuzzy matching:
  - sex: `Gender|Sex` → values normalized to Male|Female
  - race: `Race|Ethnicity` → Chinese|Malay|Indian|Others
  - education_level: `Education|Highest Education|Edu` → known labels; unknown→no_info (excluded from variance)
  - dob: `DOB|DateOfBirth` → age bucketed by target age groups
  - status: `Status`, group: `Group`
- Date parsing: common formats (e.g. `dd-MMM-yy`, `dd-MMM-YYYY`, `YYYY-mm-dd`, etc.) with sensible century inference
- Age groups: read from the targets file and used for bucketing

## Allocation and optimization

- Race-first allocation with availability caps; then within-race proportional to target stratum weights
- Fallback global spill to fill remaining if a race is supply-constrained
- Greedy swap pass across groups to reduce overall MAD
- Random tie-breaks seeded by `--seed`

## Troubleshooting

- “Insufficient eligible REGISTERED candidates…”: you don’t have enough supply in some strata; use the “Focus by strata” list and the supply counts to adjust recruiting or relax constraints
- No file writes in dry-run: ensure you did not pass `--save` or omit `--dry-run`
- Column detection: check your header names match the synonyms above, or tweak the input file

## Reproducibility

- Use `--seed` to get deterministic selections and stable variance printouts.

## License

Internal research tooling.
