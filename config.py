# Dynamic group configuration

# Ordered list of group labels written into the participants CSV `Group` column.
# You can rename or add/remove labels here to change the number of groups.
GROUP_LABELS = [
	"Group 1",
	"Group 2",
]

# Required per-group totals (including existing). Every label in GROUP_LABELS must have a total.
GROUP_TOTALS = {
	"Group 1": 100,
	"Group 2": 100,
}

# Core run configuration (no CLI flags used)
# Absolute or relative paths are accepted.
PARTICIPANTS_CSV = "data/GovernmentCallStudyAug2025_2025-08-11.csv"
TARGETS_CSV = "final_2024_agegroup_sex_race_education.csv"
SEED = 42

# Optional: override the participants file used by variance checker.
# If left None, it will use PARTICIPANTS_CSV with ".csv" replaced by ".updated.csv".
VARIANCE_PARTICIPANTS_CSV = None 