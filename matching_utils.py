import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Any, Iterable, Set

# ---------- Constants and types ----------

CANONICAL_SEX = ["Male", "Female"]  # Targets file uses Male/Female only
CANONICAL_RACES = ["Chinese", "Malay", "Indian", "Others"]
CANONICAL_EDU = [
    "Below Secondary",
    "Secondary",
    "Post Secondary (Non-Tertiary)",
    "Diploma & Professional Qualification",
    "University",
    "no_info",
]

STATUS_CONFIRMED = "CONFIRMED"
STATUS_TO_CONTACT = "TO_CONTACT"
STATUS_REGISTERED = "REGISTERED"

GROUP_VALUES = {"1", "2"}

@dataclass(frozen=True)
class Stratum:
    sex: str
    age_group: str
    race: str
    education_level: str

# ---------- Header / value normalization ----------

def _normalize_key(s: str) -> str:
    return ''.join(ch.lower() for ch in s if ch.isalnum())

HEADER_SYNONYMS = {
    "sex": ["sex", "gender"],
    "race": ["race", "ethnicity"],
    "education_level": ["education", "highesteducation", "edu"],
    "dob": ["dob", "dateofbirth", "birthdate", "dateofbirth(yyyy-mm-dd)"],
    "status": ["status"],
    "group": ["group"],
    "name": ["name", "fullname"],
}

VALUE_CLEAN_MAP_SEX = {
    "m": "Male", "male": "Male", "man": "Male",
    "f": "Female", "female": "Female", "woman": "Female",
}

VALUE_CLEAN_MAP_RACE = {
    "chinese": "Chinese",
    "malay": "Malay",
    "indian": "Indian",
    "others": "Others",
    "other": "Others",
}

# Education mapping: pass through known labels; map unknowns to no_info
KNOWN_EDU_NORMALIZED = {
    _normalize_key(x): x for x in [
        "Below Secondary",
        "Secondary",
        "Post Secondary (Non-Tertiary)",
        "Diploma & Professional Qualification",
        "University",
        "no_info",
    ]
}

DATE_INPUT_FORMATS = [
    "%d-%b-%y",  # 18-Sep-00
    "%d-%b-%Y",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d/%m/%y",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d %b %Y",
    "%d %b %y",
]


def find_header_mapping(header: List[str]) -> Dict[str, str]:
    """Return mapping canonical_key -> actual_header_name using fuzzy synonyms."""
    norm_to_actual: Dict[str, str] = {_normalize_key(h): h for h in header}
    mapping: Dict[str, str] = {}
    for canon, syns in HEADER_SYNONYMS.items():
        for syn in syns:
            key = _normalize_key(syn)
            if key in norm_to_actual:
                mapping[canon] = norm_to_actual[key]
                break
    return mapping


def clean_sex(value: str) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if v in VALUE_CLEAN_MAP_SEX:
        return VALUE_CLEAN_MAP_SEX[v]
    # direct match Male/Female
    if v == "male":
        return "Male"
    if v == "female":
        return "Female"
    # Any other values treated as Other/ignored (targets lack Other). Return None to exclude.
    return None


def clean_race(value: str) -> str:
    if not value:
        return "Others"
    v = value.strip().lower()
    if v in VALUE_CLEAN_MAP_RACE:
        return VALUE_CLEAN_MAP_RACE[v]
    # Map explicitly known primary races; else Others
    if v in {"ch", "cn", "zhongguo"}:
        return "Chinese"
    if v in {"my", "ms"}:
        return "Malay"
    if v in {"in", "ind"}:
        return "Indian"
    return "Others"


def clean_education(value: str) -> str:
    if not value:
        return "no_info"
    v_norm = _normalize_key(value)
    # Common light normalizations
    replacements = {
        "postsecondarynontertiary": "Post Secondary (Non-Tertiary)",
        "diplomaprofessionalqualification": "Diploma & Professional Qualification",
        "belowsecondary": "Below Secondary",
        "secondary": "Secondary",
        "university": "University",
        "na": "no_info",
        "nainfo": "no_info",
        "unknown": "no_info",
        "noinfo": "no_info",
        "none": "no_info",
    }
    if v_norm in KNOWN_EDU_NORMALIZED:
        return KNOWN_EDU_NORMALIZED[v_norm]
    if v_norm in replacements:
        return replacements[v_norm]
    return "no_info"


# ---------- Age parsing / bucketing ----------

@dataclass
class AgeGroup:
    label: str
    min_age: int
    max_age: Optional[int]  # None means open-ended upper bound


def parse_age_group_label(label: str) -> AgeGroup:
    label = label.strip()
    if label.endswith('+'):
        min_age = int(label[:-1])
        return AgeGroup(label=label, min_age=min_age, max_age=None)
    if '-' in label:
        parts = label.split('-')
        return AgeGroup(label=label, min_age=int(parts[0]), max_age=int(parts[1]))
    raise ValueError(f"Unrecognized age_group label: {label}")


def make_age_group_bucketer(age_group_labels: Iterable[str]):
    """Return a function that maps (age:int) -> age_group_label or None."""
    groups: List[AgeGroup] = [parse_age_group_label(l) for l in set(age_group_labels)]
    # sort by min_age
    groups.sort(key=lambda g: g.min_age)

    def bucket(age: int) -> Optional[str]:
        for g in groups:
            if g.max_age is None:
                if age >= g.min_age:
                    return g.label
            else:
                if g.min_age <= age <= g.max_age:
                    return g.label
        return None

    return bucket


def parse_date_with_century(value: str) -> Optional[date]:
    if not value or not value.strip():
        return None
    for fmt in DATE_INPUT_FORMATS:
        try:
            dt = datetime.strptime(value.strip(), fmt)
            # If two-digit year, datetime already infers, but ensure sensible mapping:
            # Rule: if year in [0..current_year%100] -> 2000s, else 1900s
            if "%y" in fmt and "%Y" not in fmt:
                year_two = dt.year % 100
                current_two = datetime.today().year % 100
                century = 2000 if year_two <= current_two else 1900
                dt = dt.replace(year=century + year_two)
            return dt.date()
        except Exception:
            continue
    return None


def compute_age(born: date, ref: Optional[date] = None) -> int:
    if ref is None:
        ref = date.today()
    years = ref.year - born.year
    if (ref.month, ref.day) < (born.month, born.day):
        years -= 1
    return years

# ---------- CSV I/O ----------


def read_csv_dicts(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_dicts(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ---------- Targets processing ----------

@dataclass
class Targets:
    total_count: int
    by_stratum_count: Dict[Stratum, int]
    by_stratum_prop: Dict[Stratum, float]
    categories: Dict[str, Set[str]]  # keys: sex, age_group, race, education_level
    marginals_prop: Dict[str, Dict[str, float]]  # dimension -> category -> prop


def load_targets(path: str) -> Targets:
    rows = read_csv_dicts(path)
    by_stratum_count: Dict[Stratum, int] = {}
    categories: Dict[str, Set[str]] = {
        "sex": set(),
        "age_group": set(),
        "race": set(),
        "education_level": set(),
    }
    total = 0
    for r in rows:
        sex = r["sex"].strip()
        age_group = r["age_group"].strip()
        race = r["race"].strip()
        edu = r["education_level"].strip()
        cnt = int(r["count"]) if r.get("count") not in (None, "") else 0
        st = Stratum(sex=sex, age_group=age_group, race=race, education_level=edu)
        by_stratum_count[st] = by_stratum_count.get(st, 0) + cnt
        total += cnt
        categories["sex"].add(sex)
        categories["age_group"].add(age_group)
        categories["race"].add(race)
        categories["education_level"].add(edu)
    by_stratum_prop: Dict[Stratum, float] = {st: c / total for st, c in by_stratum_count.items()}

    # marginals
    marginals_count: Dict[str, Dict[str, int]] = {
        dim: {} for dim in categories
    }
    for st, c in by_stratum_count.items():
        for dim in categories:
            key = getattr(st, dim)
            marginals_count[dim][key] = marginals_count[dim].get(key, 0) + c
    marginals_prop: Dict[str, Dict[str, float]] = {
        dim: {k: v / total for k, v in marginals_count[dim].items()} for dim in categories
    }

    return Targets(
        total_count=total,
        by_stratum_count=by_stratum_count,
        by_stratum_prop=by_stratum_prop,
        categories=categories,
        marginals_prop=marginals_prop,
    )

# ---------- Participants normalization ----------

@dataclass
class Participant:
    index: int  # index in original rows list
    row: Dict[str, Any]  # original row with original headers
    sex: Optional[str]
    race: str
    education_level: str
    dob: Optional[date]
    age: Optional[int]
    age_group: Optional[str]
    status: str
    group: Optional[str]  # "1" | "2" | None
    eligible: bool  # REGISTERED and age >= 18


def normalize_participants(rows: List[Dict[str, Any]], targets: Targets, seed: int = 42) -> Tuple[List[Participant], Dict[str, str]]:
    header = list(rows[0].keys()) if rows else []
    mapping = find_header_mapping(header)

    age_bucket = make_age_group_bucketer(targets.categories["age_group"])  # function

    rng = random.Random(seed)

    participants: List[Participant] = []
    for i, r in enumerate(rows):
        # Extract fields using mapping; if missing, use empty string
        sex_raw = r.get(mapping.get("sex", ""), "")
        race_raw = r.get(mapping.get("race", ""), "")
        edu_raw = r.get(mapping.get("education_level", ""), "")
        dob_raw = r.get(mapping.get("dob", ""), "")
        status_raw = r.get(mapping.get("status", ""), "")
        group_raw = r.get(mapping.get("group", ""), "")

        sex = clean_sex(sex_raw)
        race = clean_race(race_raw)
        edu = clean_education(edu_raw)
        dob = parse_date_with_century(dob_raw)
        age = compute_age(dob) if dob else None
        age_group = age_bucket(age) if (age is not None and age >= 0) else None

        status = (status_raw or "").strip().upper()
        if status not in {STATUS_CONFIRMED, STATUS_TO_CONTACT, STATUS_REGISTERED}:
            # Normalize noise to REGISTERED to avoid accidental inclusion unless explicitly requested otherwise
            # But safer to set as-is; we only shortlist REGISTERED anyway
            pass
        group = None
        if group_raw is not None and str(group_raw).strip() != "":
            gg = str(group_raw).strip()
            if gg in GROUP_VALUES:
                group = gg

        is_adult = (age is not None and age >= 18)
        eligible = (status == STATUS_REGISTERED and is_adult and sex in CANONICAL_SEX and age_group in targets.categories["age_group"])  # race/edu already coerced

        participants.append(Participant(
            index=i,
            row=r,
            sex=sex,
            race=race,
            education_level=edu,
            dob=dob,
            age=age,
            age_group=age_group,
            status=status,
            group=group,
            eligible=eligible,
        ))

    return participants, mapping


# ---------- Utility: largest remainder allocation ----------

def allocate_largest_remainder(desired: Dict[Stratum, float], total_to_assign: int, caps: Dict[Stratum, int]) -> Dict[Stratum, int]:
    """Given desired fractional counts per stratum and caps, allocate integers summing to total_to_assign.
    Uses largest remainder (Hamilton) method with caps.
    """
    # Floor respecting caps
    base: Dict[Stratum, int] = {}
    remainders: List[Tuple[float, Stratum]] = []
    sum_base = 0
    for st, want in desired.items():
        cap = max(0, caps.get(st, 0))
        x = min(int(math.floor(want)), cap)
        base[st] = x
        sum_base += x
        rem = max(0.0, min(1.0, want - x)) if cap > x else 0.0
        remainders.append((rem, st))

    remaining = max(0, total_to_assign - sum_base)
    # sort by remainder desc and then by a stable tuple order
    remainders.sort(key=lambda t: (t[0], t[1].sex, t[1].age_group, t[1].race, t[1].education_level), reverse=True)

    alloc = dict(base)
    for rem, st in remainders:
        if remaining <= 0:
            break
        if alloc[st] < caps.get(st, 0):
            alloc[st] += 1
            remaining -= 1
    return alloc


# ---------- Utility: spillover priorities ----------


def build_spill_priority(st: Stratum) -> List[Tuple[int, Stratum]]:
    """Generate priority levels for spillover search relative to a source stratum.
    Lower priority number = higher priority.
    We generate match masks in decreasing strictness.
    """
    levels: List[Tuple[int, Stratum]] = []
    # This utility only provides the relative priority logic shell; selection code will use this with available strata.
    # We return empty here; the shortlist module will implement the matching using this concept.
    return levels 