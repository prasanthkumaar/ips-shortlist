import csv
import sys
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Any, Set

# ==========================
# Configuration
# ==========================
DEFAULT_SEED = 42
NEW_PICKS_PER_GROUP = 100
GROUPS = [1, 2]
BALANCE_LAMBDA = 0.5
SWAP_TRIES_PER_ITER = 20
MAX_SWAP_ITERS = 500

# ==========================
# Utilities: string and header normalization
# ==========================

def normalize_header_name(header: str) -> str:
    return ''.join(ch.lower() for ch in header if ch.isalnum())


def best_header_match(headers: List[str], candidates: List[str]) -> Optional[str]:
    norm_headers = {normalize_header_name(h): h for h in headers}
    norm_candidates = [normalize_header_name(c) for c in candidates]
    # Exact normalized match
    for nc in norm_candidates:
        if nc in norm_headers:
            return norm_headers[nc]
    # Startswith/contains fallbacks
    for nh_key, orig in norm_headers.items():
        for nc in norm_candidates:
            if nh_key.startswith(nc) or nc in nh_key:
                return orig
    return None


def detect_columns(headers: List[str]) -> Dict[str, Optional[str]]:
    # Lock down to exact headers provided by the user
    header_set = set(headers)
    mapping: Dict[str, Optional[str]] = {
        'sex': 'Gender' if 'Gender' in header_set else None,
        'race': 'Race' if 'Race' in header_set else None,
        'education_level': 'Education' if 'Education' in header_set else None,
        'dob': 'DOB' if 'DOB' in header_set else None,
        'status': 'Status' if 'Status' in header_set else None,
        'group': 'Group' if 'Group' in header_set else None,
        'name': 'Name' if 'Name' in header_set else None,
        'email': 'Email' if 'Email' in header_set else None,
        'phone': 'Number' if 'Number' in header_set else None,
        'os': 'OS' if 'OS' in header_set else None,
    }
    return mapping

# ==========================
# Value normalization
# ==========================

def normalize_sex(value: str) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if v in {'m', 'male'}:
        return 'Male'
    if v in {'f', 'female'}:
        return 'Female'
    # If other values, return None to skip (targets only have Male/Female)
    return None


def normalize_race(value: str) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if 'chinese' in v:
        return 'Chinese'
    if 'malay' in v:
        return 'Malay'
    # Treat Indian and Sikh under Indian
    if 'indian' in v or 'sikh' in v or 'tamil' in v:
        return 'Indian'
    # Everything else → Others
    return 'Others'


def normalize_education(value: Optional[str]) -> str:
    if not value:
        return 'no_info'
    v = value.strip().lower()
    if v in {'below secondary', 'belowsecondary', 'primary', 'psle'}:
        return 'Below Secondary'
    if v in {'secondary', 'o level', 'olevel', 'n level', 'nlevel'}:
        return 'Secondary'
    if 'post' in v and 'secondary' in v:
        return 'Post Secondary (Non-Tertiary)'
    if 'diploma' in v or 'professional' in v:
        return 'Diploma & Professional Qualification'
    if 'university' in v or 'degree' in v or 'bachelor' in v or 'masters' in v or 'phd' in v:
        return 'University'
    # Any unknown → no_info
    return 'no_info'


def normalize_os(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip().lower()
    if 'apple' in v or v == 'ios':
        return 'iOS'
    # Treat anything else as Android per provided values
    if 'android' in v or any(b in v for b in ['samsung', 'google', 'oppo', 'xiaomi', 'huawei', 'oneplus', 'vivo', 'realme', 'sony', 'motorola', 'nothing']):
        return 'Android'
    # Default to Android if present but not recognized explicitly
    return 'Android'

# ==========================
# Dates and ages
# ==========================

def parse_date_maybe(value: str) -> Optional[date]:
    if not value:
        return None
    value = value.strip()
    # Try multiple formats
    fmts = [
        '%d-%b-%Y', '%d-%b-%y', '%d/%m/%Y', '%d/%m/%y',
        '%Y-%m-%d', '%d-%m-%Y', '%d-%m-%y', '%m/%d/%Y', '%m/%d/%y'
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(value, fmt).date()
            # two-digit year handling when using %y
            # strptime already handles 00-68 as 2000-2068 and 69-99 as 1969-1999 in some implementations
            # To be safe, if date is in the future by > 1 day, subtract 100 years
            if dt > date.today():
                try:
                    dt = dt.replace(year=dt.year - 100)
                except ValueError:
                    pass
            return dt
        except Exception:
            continue
    # As a last resort, try day-monthname-yy with varied cases
    try:
        parts = value.replace(',', ' ').replace('.', ' ').split()
        if len(parts) == 3:
            day = int(parts[0])
            mon_str = parts[1][:3].title()
            year = int(parts[2])
            if year < 100:
                year += 2000 if year <= 24 else 1900
            dt = datetime.strptime(f"{day}-{mon_str}-{year}", '%d-%b-%Y').date()
            if dt > date.today():
                dt = dt.replace(year=dt.year - 100)
            return dt
    except Exception:
        pass
    return None


def compute_age(birth_date: date, on_date: Optional[date] = None) -> int:
    if on_date is None:
        on_date = date.today()
    years = on_date.year - birth_date.year
    if (on_date.month, on_date.day) < (birth_date.month, birth_date.day):
        years -= 1
    return years

# ==========================
# Targets handling
# ==========================

def load_targets(targets_csv_path: str) -> Tuple[List[str], Set[Tuple[str, str, str, str]], Dict[Tuple[str, str, str, str], int]]:
    # Returns: (age_groups list), (valid 4D keys), (counts per 4D key)
    counts: Dict[Tuple[str, str, str, str], int] = {}
    age_groups_order: List[str] = []
    valid_keys: Set[Tuple[str, str, str, str]] = set()

    with open(targets_csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            age_group = row['age_group'].strip()
            sex = row['sex'].strip()
            race = row['race'].strip()
            edu = row['education_level'].strip()
            cnt = int(row['count'])
            key = (sex, age_group, race, edu)
            counts[key] = counts.get(key, 0) + cnt
            valid_keys.add(key)
            if age_group not in age_groups_order:
                age_groups_order.append(age_group)

    return age_groups_order, valid_keys, counts


def parse_age_group_label(label: str) -> Tuple[int, Optional[int]]:
    label = label.strip()
    if '+' in label:
        floor = int(label.replace('+', '').strip())
        return floor, None
    if '-' in label:
        a, b = label.split('-', 1)
        return int(a.strip()), int(b.strip())
    raise ValueError(f"Unrecognized age_group label: {label}")


def make_age_bucketer(age_groups_order: List[str]):
    parsed = []
    for lbl in age_groups_order:
        lo, hi = parse_age_group_label(lbl)
        parsed.append((lbl, lo, hi))

    def bucket(age: int) -> Optional[str]:
        if age < 0:
            return None
        for lbl, lo, hi in parsed:
            if hi is None:
                if age >= lo:
                    return lbl
            else:
                if lo <= age <= hi:
                    return lbl
        return None

    return bucket

# ==========================
# Selection helpers
# ==========================

def proportionize(counter: Dict[Any, int]) -> Dict[Any, float]:
    total = sum(counter.values())
    if total <= 0:
        return {k: 0.0 for k in counter.keys()}
    return {k: v / total for k, v in counter.items()}


def compute_marginal_targets(target_counts: Dict[Tuple[str, str, str, str], int]):
    sex_c = Counter()
    age_c = Counter()
    race_c = Counter()
    edu_c = Counter()
    for (sex, age, race, edu), cnt in target_counts.items():
        sex_c[sex] += cnt
        age_c[age] += cnt
        race_c[race] += cnt
        edu_c[edu] += cnt
    return (
        proportionize(sex_c),
        proportionize(age_c),
        proportionize(race_c),
        proportionize(edu_c),
    )


def mean_abs_dev_over_marginals(group_items: List[Tuple[str, str, str, str]], target_marginals):
    sex_t, age_t, race_t, edu_t = target_marginals
    sex_obs = Counter()
    age_obs = Counter()
    race_obs = Counter()
    edu_obs = Counter()
    for sex, age, race, edu in group_items:
        sex_obs[sex] += 1
        age_obs[age] += 1
        race_obs[race] += 1
        edu_obs[edu] += 1
    sex_p = proportionize(sex_obs)
    age_p = proportionize(age_obs)
    race_p = proportionize(race_obs)
    edu_p = proportionize(edu_obs)
    def mad(obs_p: Dict[str, float], tgt_p: Dict[str, float]) -> float:
        cats = set(obs_p.keys()) | set(tgt_p.keys())
        return sum(abs(obs_p.get(c, 0.0) - tgt_p.get(c, 0.0)) for c in cats) / max(len(cats), 1)
    return (mad(sex_p, sex_t) + mad(age_p, age_t) + mad(race_p, race_t) + mad(edu_p, edu_t)) / 4.0


def mean_abs_dev_for_os(os_list: List[str], target_os: Dict[str, float]) -> float:
    obs = Counter(os_list)
    obs_p = proportionize(obs)
    cats = set(obs_p.keys()) | set(target_os.keys())
    return sum(abs(obs_p.get(c, 0.0) - target_os.get(c, 0.0)) for c in cats) / max(len(cats), 1)

# ==========================
# Main selection pipeline
# ==========================

def load_participants(csv_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []
    colmap = detect_columns(headers)
    return rows, colmap


def normalize_row(row: Dict[str, Any], colmap: Dict[str, Optional[str]], age_bucket, eighteen_to_twentyfour_labels: Set[str]) -> Optional[Dict[str, Any]]:
    # Only keep rows that can be normalized
    status_val = row.get(colmap['status'], '') if colmap['status'] else ''
    status = (status_val or '').strip().upper()

    dob_raw = row.get(colmap['dob'], '') if colmap['dob'] else ''
    dob = parse_date_maybe(dob_raw) if dob_raw else None
    if not dob:
        return None
    age = compute_age(dob)
    if age < 18:
        return None
    age_group = age_bucket(age)
    if not age_group:
        return None

    sex_val = row.get(colmap['sex'], '') if colmap['sex'] else ''
    sex = normalize_sex(sex_val)
    if not sex:
        return None

    race_val = row.get(colmap['race'], '') if colmap['race'] else ''
    race = normalize_race(race_val)
    if not race:
        return None

    edu_val = row.get(colmap['education_level'], '') if colmap['education_level'] else ''
    edu_norm = normalize_education(edu_val)

    # Additional note: for 18-24, force education to no_info
    if age_group in eighteen_to_twentyfour_labels:
        edu_norm = 'no_info'

    os_val = row.get(colmap['os'], '') if colmap.get('os') else ''
    os_norm = normalize_os(os_val)
    if not os_norm:
        return None

    # Keep original for writing back
    norm = dict(row)
    norm['__norm_sex'] = sex
    norm['__norm_race'] = race
    norm['__norm_education_level'] = edu_norm
    norm['__norm_age_group'] = age_group
    norm['__norm_status'] = status
    norm['__norm_os'] = os_norm
    return norm


def build_target_quota(target_counts: Dict[Tuple[str, str, str, str], int], total_needed: int) -> Dict[Tuple[str, str, str, str], int]:
    total_pop = sum(target_counts.values())
    if total_pop <= 0:
        return defaultdict(int)
    # Floating desired counts
    desired_float = {k: (v / total_pop) * total_needed for k, v in target_counts.items()}
    # Round using largest remainder method
    base = {k: int(v) for k, v in desired_float.items()}
    remainder = {k: desired_float[k] - base[k] for k in desired_float}
    remaining = total_needed - sum(base.values())
    # Assign remaining to keys with largest remainders
    for k, _ in sorted(remainder.items(), key=lambda kv: kv[1], reverse=True)[:max(0, remaining)]:
        base[k] += 1
    return base


def shortlist(
    participants_csv: str,
    targets_csv: str,
    seed: int = DEFAULT_SEED,
    picks_per_group: int = NEW_PICKS_PER_GROUP,
) -> str:
    random.seed(seed)

    age_groups_order, valid_keys, target_counts = load_targets(targets_csv)
    age_bucket = make_age_bucketer(age_groups_order)
    eighteen_to_twentyfour_labels = {lbl for lbl in age_groups_order if ('18-19' in lbl or '20-24' in lbl)}

    rows, colmap = load_participants(participants_csv)

    # Normalize rows and separate by status
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        norm = normalize_row(row, colmap, age_bucket, eighteen_to_twentyfour_labels)
        if not norm:
            continue
        normalized_rows.append(norm)

    # Pre-account existing CONFIRMED and TO_CONTACT toward quotas
    confirmed_or_pending: List[Dict[str, Any]] = [
        r for r in normalized_rows if r['__norm_status'] in {'CONFIRMED', 'TO_CONTACT', 'CONTACTED'}
    ]
    existing_counts = Counter()
    # Track existing OS counts by group and overall for OS balancing
    existing_by_group_os: Dict[int, List[str]] = {g: [] for g in GROUPS}
    for r in confirmed_or_pending:
        key = (r['__norm_sex'], r['__norm_age_group'], r['__norm_race'], r['__norm_education_level'])
        # Only count toward 4D targets if key exists in targets
        if key in target_counts:
            existing_counts[key] += 1

    # Eligible pool is REGISTERED and not yet pending/confirmed
    eligible: List[Dict[str, Any]] = [r for r in normalized_rows if r['__norm_status'] == 'REGISTERED']

    # Per-group top-up: compute current group sizes and remaining capacity
    group_header_in = colmap['group'] if colmap['group'] else 'Group'
    existing_by_group_keys: Dict[int, List[Tuple[str, str, str, str]]] = {g: [] for g in GROUPS}
    current_group_size: Dict[int, int] = {g: 0 for g in GROUPS}
    for r in confirmed_or_pending:
        key = (r['__norm_sex'], r['__norm_age_group'], r['__norm_race'], r['__norm_education_level'])
        g_raw = r.get(group_header_in)
        try:
            g_val = int(str(g_raw).strip()) if g_raw is not None and str(g_raw).strip() != '' else None
        except Exception:
            g_val = None
        if g_val in GROUPS:
            existing_by_group_keys[g_val].append(key)
            existing_by_group_os[g_val].append(r['__norm_os'])
            current_group_size[g_val] += 1

    group_capacity_remaining: Dict[int, int] = {g: max(0, picks_per_group - current_group_size[g]) for g in GROUPS}
    total_new_needed = sum(group_capacity_remaining.values())

    # Build desired new quotas at 4D level accounting for existing locked/pending
    desired_total_counts = build_target_quota(target_counts, total_new_needed + sum(existing_counts.values()))
    # New picks quota = desired_total - existing
    new_quota = {k: max(0, desired_total_counts.get(k, 0) - existing_counts.get(k, 0)) for k in target_counts.keys()}

    # Greedy fill by exact 4D key
    by_key_pool: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    for idx, r in enumerate(eligible):
        key = (r['__norm_sex'], r['__norm_age_group'], r['__norm_race'], r['__norm_education_level'])
        if key in target_counts:
            by_key_pool[key].append(idx)

    # Shuffle indices for reproducibility but random tie-breaking
    for lst in by_key_pool.values():
        random.shuffle(lst)

    selected_indices: List[int] = []

    # First pass: fill deficits by key
    for key, deficit in sorted(new_quota.items(), key=lambda kv: kv[1], reverse=True):
        if deficit <= 0:
            continue
        pool = by_key_pool.get(key, [])
        take = min(deficit, len(pool))
        selected_indices.extend(pool[:take])
        by_key_pool[key] = pool[take:]

    # OS targets
    OS_TARGET = {'iOS': 0.3, 'Android': 0.7}
    total_existing_in_groups = sum(current_group_size.values())
    total_final_after_selection = total_existing_in_groups + total_new_needed
    desired_total_ios_count = int(round(OS_TARGET['iOS'] * total_final_after_selection))

    # If not enough, second pass: score by marginal improvement + OS balancing
    if len(selected_indices) < total_new_needed:
        # Build current observed counts (existing + selected so far)
        current_counts = Counter(existing_counts)
        current_os_counts = Counter()
        for g in GROUPS:
            for osv in existing_by_group_os[g]:
                current_os_counts[osv] += 1
        for i in selected_indices:
            r = eligible[i]
            key = (r['__norm_sex'], r['__norm_age_group'], r['__norm_race'], r['__norm_education_level'])
            current_counts[key] += 1
            current_os_counts[r['__norm_os']] += 1

        # Precompute marginals from targets
        target_marginals = compute_marginal_targets(target_counts)

        # Candidates not yet selected
        selected_set = set(selected_indices)
        remaining_indices = [i for i in range(len(eligible)) if i not in selected_set]

        def candidate_score(idx: int) -> float:
            r = eligible[idx]
            key = (r['__norm_sex'], r['__norm_age_group'], r['__norm_race'], r['__norm_education_level'])
            # Score higher if key is still under desired_total_counts vs current_counts
            deficit_key = desired_total_counts.get(key, 0) - current_counts.get(key, 0)
            sex, age, race, edu = key
            s = 0.0
            if deficit_key > 0:
                s += 3.0
            s += target_marginals[0].get(sex, 0) * 1.0
            s += target_marginals[1].get(age, 0) * 1.0
            s += target_marginals[2].get(race, 0) * 0.5
            s += target_marginals[3].get(edu, 0) * 0.5
            # OS bias toward reaching 30/70 overall
            if r['__norm_os'] == 'iOS':
                ios_deficit = desired_total_ios_count - current_os_counts.get('iOS', 0)
                s += 2.0 if ios_deficit > 0 else -1.0
            else:
                desired_android = total_final_after_selection - desired_total_ios_count
                android_deficit = desired_android - current_os_counts.get('Android', 0)
                s += 2.0 if android_deficit > 0 else -1.0
            s += random.random() * 0.01
            return s

        remaining_indices.sort(key=candidate_score, reverse=True)
        need = total_new_needed - len(selected_indices)
        selected_indices.extend(remaining_indices[:need])

    # Ensure exactly total_new_needed
    selected_indices = selected_indices[:total_new_needed]
    selected_rows = [eligible[i] for i in selected_indices]

    # Assign to groups with balancing objective, seeding with existing compositions
    target_marginals = compute_marginal_targets(target_counts)
    target_os_marginal = {'iOS': 0.3, 'Android': 0.7}

    fixed_allocations: Dict[int, List[Tuple[str, str, str, str]]] = {g: list(existing_by_group_keys[g]) for g in GROUPS}
    var_keys: Dict[int, List[Tuple[str, str, str, str]]] = {g: [] for g in GROUPS}
    var_rows_by_group: Dict[int, List[Dict[str, Any]]] = {g: [] for g in GROUPS}

    def build_groups_items() -> Dict[int, List[Tuple[str, str, str, str]]]:
        return {g: fixed_allocations[g] + var_keys[g] for g in GROUPS}

    def build_groups_os() -> Dict[int, List[str]]:
        return {g: existing_by_group_os[g] + [r['__norm_os'] for r in var_rows_by_group[g]] for g in GROUPS}

    def group_mads(groups_keys: Dict[int, List[Tuple[str, str, str, str]]], groups_os: Dict[int, List[str]]) -> Dict[int, float]:
        m: Dict[int, float] = {}
        for g in GROUPS:
            m4 = mean_abs_dev_over_marginals(groups_keys[g], target_marginals)
            mos = mean_abs_dev_for_os(groups_os[g], target_os_marginal)
            # Average across 5 dimensions: 4D + OS
            overall = (m4 * 4.0 + mos) / 5.0
            m[g] = overall
        return m

    def objective(groups_keys: Dict[int, List[Tuple[str, str, str, str]]], groups_os: Dict[int, List[str]]) -> float:
        m = group_mads(groups_keys, groups_os)
        total = sum(m.values())
        balance = abs(m.get(GROUPS[0], 0.0) - m.get(GROUPS[1], 0.0))
        return total + BALANCE_LAMBDA * balance

    capacity_remaining = dict(group_capacity_remaining)
    random.shuffle(selected_rows)

    for r in selected_rows:
        key = (r['__norm_sex'], r['__norm_age_group'], r['__norm_race'], r['__norm_education_level'])
        best_g = None
        best_obj = float('inf')
        base_keys = build_groups_items()
        base_os = build_groups_os()
        for g in GROUPS:
            if capacity_remaining[g] <= 0:
                continue
            trial_keys = {gg: list(base_keys[gg]) for gg in GROUPS}
            trial_os = {gg: list(base_os[gg]) for gg in GROUPS}
            trial_keys[g].append(key)
            trial_os[g].append(r['__norm_os'])
            obj = objective(trial_keys, trial_os)
            if obj < best_obj:
                best_obj = obj
                best_g = g
        if best_g is None:
            # Fallback to any group with capacity
            candidates = [g for g in GROUPS if capacity_remaining[g] > 0]
            best_g = candidates[0] if candidates else GROUPS[0]
        var_keys[best_g].append(key)
        var_rows_by_group[best_g].append(r)
        r['__assigned_group'] = best_g
        capacity_remaining[best_g] = max(0, capacity_remaining[best_g] - 1)

    # Swap pass over new picks only
    def build_current_obj() -> float:
        return objective(build_groups_items(), build_groups_os())

    current_obj = build_current_obj()
    iters = 0
    improved = True
    while improved and iters < MAX_SWAP_ITERS:
        iters += 1
        improved = False
        for _ in range(SWAP_TRIES_PER_ITER):
            a_group, b_group = GROUPS
            if not var_keys[a_group] or not var_keys[b_group]:
                break
            i = random.randrange(0, len(var_keys[a_group]))
            j = random.randrange(0, len(var_keys[b_group]))
            # Swap keys and rows
            var_keys[a_group][i], var_keys[b_group][j] = var_keys[b_group][j], var_keys[a_group][i]
            var_rows_by_group[a_group][i], var_rows_by_group[b_group][j] = var_rows_by_group[b_group][j], var_rows_by_group[a_group][i]
            new_obj = build_current_obj()
            if new_obj + 1e-9 < current_obj:
                current_obj = new_obj
                # Update assigned groups in rows
                var_rows_by_group[a_group][i]['__assigned_group'] = a_group
                var_rows_by_group[b_group][j]['__assigned_group'] = b_group
                improved = True
            else:
                # Revert swap
                var_keys[a_group][i], var_keys[b_group][j] = var_keys[b_group][j], var_keys[a_group][i]
                var_rows_by_group[a_group][i], var_rows_by_group[b_group][j] = var_rows_by_group[b_group][j], var_rows_by_group[a_group][i]
        # End tries loop

    # Write updated CSV
    updated_path = participants_csv.replace('.csv', '.updated.csv')

    # Ensure Group and Status columns exist in output header
    with open(participants_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        input_headers = reader.fieldnames or []

    out_headers = list(input_headers)
    if colmap['status'] is None:
        out_headers.append('Status')
        status_header = 'Status'
    else:
        status_header = colmap['status']
    if colmap['group'] is None:
        out_headers.append('Group')
        group_header = 'Group'
    else:
        group_header = colmap['group']

    # Build index mapping for writing
    selected_row_ids = set()
    # Use tuple of stable identifying fields; fallback to object id
    def row_identity(r: Dict[str, Any]) -> Tuple:
        return (
            r.get(colmap.get('name') or '', ''),
            r.get(colmap.get('email') or '', ''),
            r.get(colmap.get('phone') or '', ''),
            r.get(colmap.get('dob') or '', ''),
        )

    selected_id_to_group: Dict[Tuple, int] = {}
    for r in selected_rows:
        selected_row_ids.add(id(r))
        selected_id_to_group[row_identity(r)] = r['__assigned_group']

    # Now write rows, updating only newly selected REGISTERED rows
    with open(participants_csv, 'r', newline='', encoding='utf-8') as f_in, open(updated_path, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=out_headers)
        writer.writeheader()
        for raw in reader:
            row_out = dict(raw)
            # Normalize to check if this is a selected row
            norm = normalize_row(raw, colmap, age_bucket, eighteen_to_twentyfour_labels)
            if norm and norm['__norm_status'] == 'REGISTERED':
                rid = row_identity(norm)
                if rid in selected_id_to_group:
                    row_out[status_header] = 'TO_CONTACT'
                    row_out[group_header] = str(selected_id_to_group[rid])
            writer.writerow(row_out)

    return updated_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Shortlist participants into two groups while matching Singapore demographics and OS ratio (30% iOS / 70% Android).')
    parser.add_argument('--participants', '-p', type=str, required=True, help='Path to participants CSV (source of truth).')
    parser.add_argument('--targets', '-t', type=str, required=True, help='Path to targets CSV (ground truth counts).')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed for reproducibility.')
    parser.add_argument('--per_group', type=int, default=NEW_PICKS_PER_GROUP, help='New picks per group.')
    args = parser.parse_args()

    updated = shortlist(args.participants, args.targets, seed=args.seed, picks_per_group=args.per_group)
    print(f'Updated CSV written to: {updated}')
    # Also print a ready-to-run variance checker command (absolute paths)
    shortlist_dir = os.path.dirname(os.path.abspath(__file__))
    variance_path = os.path.join(shortlist_dir, 'variance_checker.py')
    updated_abs = os.path.abspath(updated)
    targets_abs = os.path.abspath(args.targets)
    print('Run this to view variance per group:')
    print(f'python3 {variance_path} --participants "{updated_abs}" --targets "{targets_abs}"')


if __name__ == '__main__':
    main()
