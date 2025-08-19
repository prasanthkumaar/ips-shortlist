# Dynamic group configuration

# Ordered list of group labels written into the participants CSV `Group` column.
# You can rename or add/remove labels here to change the number of groups.
GROUP_LABELS = [
	"Group 3",
	"Group 4",
]

# Required per-group totals (including existing). Every label in GROUP_LABELS must have a total.
GROUP_TOTALS = {
	"Group 3": 100,
	"Group 4": 100,
}

# Core run configuration (no CLI flags used)
# Absolute or relative paths are accepted.
PARTICIPANTS_CSV = "data/reg-form-phase-2.csv"
TARGETS_CSV = "final_2024_agegroup_sex_race_education.csv"
SEED = 42

# Optional: override the participants file used by variance checker.
# If left None, it will use PARTICIPANTS_CSV with ".csv" replaced by ".updated.csv".
VARIANCE_PARTICIPANTS_CSV = None 

# Language eligibility
# Only participants whose `Preferred language of communication` value is in this list are eligible for shortlisting.
# The CSV currently only contains 'Chinese' and 'English' values; any other value will raise a validation error.
SPEAKING_LANGUAGES_ELIGIBLE = [
	"English",
]

# =============================================================================
# SCORING WEIGHTS - Adjust these to change selection algorithm priorities
# =============================================================================

# Exact 4D cell deficit bonus
# Participants in exact sex/age/race/education combinations that are under target
# get this flat bonus. Higher values prioritize filling specific demographic gaps.
CELL_DEFICIT_BONUS = 3.0

# Special bonus for highly underrepresented races
# Add an extra flat bonus for participants in severely underrepresented racial groups.
# This helps overcome 4D cell specificity when races are generally underrepresented.
UNDERREPRESENTED_RACE_BONUS = 2.0    # Extra bonus for Malay, Indian, Others
UNDERREPRESENTED_RACES = ['Malay', 'Indian', 'Others']  # Races that get the bonus

# Marginal demographic weights
# These multiply the target percentage for each demographic category.
# Higher values give more priority to underrepresented groups in that dimension.
SEX_WEIGHT = 1.0          # Weight for sex (Male/Female) marginal targets
AGE_WEIGHT = 1.0          # Weight for age group marginal targets  
RACE_WEIGHT = 1.0         # Weight for race marginal targets (increased to prioritize racial diversity)
EDUCATION_WEIGHT = 1.0    # Weight for education level marginal targets

# Device OS balance weights
# The algorithm tries to achieve 30% iOS / 70% Android across all selected participants.
# When iOS/Android is under target, participants get OS_BONUS points.
# When iOS/Android is over target, participants get OS_PENALTY points.
OS_BONUS = 0.5           # Bonus for participants with needed OS type (further reduced)
OS_PENALTY = -0.5       # Penalty for participants with over-represented OS type (further reduced)

# Random tie-breaker range
# Small random value (0 to this amount) added to break ties between equal candidates.
RANDOM_TIE_BREAKER_MAX = 0.01

# =============================================================================
# TUNING RECOMMENDATIONS:
# 
# To prioritize racial diversity over OS balance:
# - Increase RACE_WEIGHT from 0.5 to 1.0 or higher
# - Reduce OS_BONUS from 2.0 to 1.0
# - Reduce OS_PENALTY magnitude from -1.0 to -0.5
#
# To fill specific demographic gaps more aggressively:
# - Increase CELL_DEFICIT_BONUS from 3.0 to 5.0+
#
# To better balance across all demographics:
# - Set all marginal weights to 1.0: SEX_WEIGHT = AGE_WEIGHT = RACE_WEIGHT = EDUCATION_WEIGHT = 1.0
# ============================================================================= 