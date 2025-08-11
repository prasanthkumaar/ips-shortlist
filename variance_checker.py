import csv
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Any, Optional
from datetime import datetime, date


def proportionize(counter: Dict[Any, int]) -> Dict[Any, float]:
    total = sum(counter.values())
    if total <= 0:
        return {k: 0.0 for k in counter.keys()}
    return {k: v / total for k, v in counter.items()}


def load_targets(targets_csv_path: str) -> Tuple[Dict[Tuple[str, str, str, str], int], List[str]]:
    counts: Dict[Tuple[str, str, str, str], int] = {}
    age_groups_order: List[str] = []
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
            if age_group not in age_groups_order:
                age_groups_order.append(age_group)
    return counts, age_groups_order


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


def normalize_header_name(header: str) -> str:
    return ''.join(ch.lower() for ch in header if ch.isalnum())


def best_header_match(headers: List[str], candidates: List[str]) -> Optional[str]:
    norm_headers = {normalize_header_name(h): h for h in headers}
    norm_candidates = [normalize_header_name(c) for c in candidates]
    for nc in norm_candidates:
        if nc in norm_headers:
            return norm_headers[nc]
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
        'os': 'OS' if 'OS' in header_set else None,
    }
    return mapping


# Date parsing and age bucketing

def parse_date_maybe(value: str) -> Optional[date]:
    if not value:
        return None
    value = value.strip()
    fmts = [
        '%d-%b-%Y', '%d-%b-%y', '%d/%m/%Y', '%d/%m/%y',
        '%Y-%m-%d', '%d-%m-%Y', '%d-%m-%y', '%m/%d/%Y', '%m/%d/%y'
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(value, fmt).date()
            if dt > date.today():
                try:
                    dt = dt.replace(year=dt.year - 100)
                except ValueError:
                    pass
            return dt
        except Exception:
            continue
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


def report_for_group(rows: List[Dict[str, Any]], target_marginals) -> str:
    sex_t, age_t, race_t, edu_t = target_marginals
    sex_obs = Counter()
    age_obs = Counter()
    race_obs = Counter()
    edu_obs = Counter()
    os_obs = Counter()

    def norm_value(val: str) -> str:
        return (val or '').strip()

    for r in rows:
        sex_obs[norm_value(r['__norm_sex'])] += 1
        age_obs[norm_value(r['__norm_age_group'])] += 1
        race_obs[norm_value(r['__norm_race'])] += 1
        edu_obs[norm_value(r['__norm_education_level'])] += 1
        os_obs[norm_value(r['__norm_os'])] += 1

    sex_p = proportionize(sex_obs)
    age_p = proportionize(age_obs)
    race_p = proportionize(race_obs)
    edu_p = proportionize(edu_obs)
    os_p = proportionize(os_obs)

    def variance_table(obs_p: Dict[str, float], tgt_p: Dict[str, float]) -> List[Tuple[str, float, float, float]]:
        cats = set(obs_p.keys()) | set(tgt_p.keys())
        rows = []
        for c in sorted(cats):
            o = obs_p.get(c, 0.0)
            t = tgt_p.get(c, 0.0)
            diff = o - t
            rows.append((c, o, t, diff))
        return rows

    def mad(obs_p: Dict[str, float], tgt_p: Dict[str, float]) -> float:
        cats = set(obs_p.keys()) | set(tgt_p.keys())
        return sum(abs(obs_p.get(c, 0.0) - tgt_p.get(c, 0.0)) for c in cats) / max(len(cats), 1)

    def format_section(title: str, table: List[Tuple[str, float, float, float]]) -> str:
        lines = [f"{title} (category | observed% | target% | diff%)"]
        for cat, o, t, d in sorted(table, key=lambda x: x[3]):
            lines.append(f"  - {cat}: {o:.3f} | {t:.3f} | {d:+.3f}")
        return '\n'.join(lines)

    sex_tbl = variance_table(sex_p, sex_t)
    age_tbl = variance_table(age_p, age_t)
    race_tbl = variance_table(race_p, race_t)
    edu_tbl = variance_table(edu_p, edu_t)

    # OS targets per user's requirement
    os_target = {'iOS': 0.3, 'Android': 0.7}
    os_tbl = variance_table(os_p, os_target)

    # Overall single-number score (lower is better) across four original dimensions
    overall_mad = (mad(sex_p, sex_t) + mad(age_p, age_t) + mad(race_p, race_t) + mad(edu_p, edu_t)) / 4.0

    # Top under/over represented across all original dimensions
    all_rows = []
    for title, tbl in [('Sex', sex_tbl), ('Age', age_tbl), ('Race', race_tbl), ('Education', edu_tbl)]:
        for cat, o, t, d in tbl:
            all_rows.append((f"{title}:{cat}", o, t, d))

    under = sorted(all_rows, key=lambda x: x[3])[:10]
    over = sorted(all_rows, key=lambda x: x[3], reverse=True)[:10]

    lines: List[str] = []
    lines.append(f"Overall variance (mean absolute deviation across marginals, lower is better): {overall_mad*100:.2f}%")
    lines.append('')
    lines.append(format_section('Sex', sex_tbl))
    lines.append('')
    lines.append(format_section('Age Group', age_tbl))
    lines.append('')
    lines.append(format_section('Race', race_tbl))
    lines.append('')
    lines.append(format_section('Education Level', edu_tbl))
    lines.append('')
    # New Device OS section
    lines.append(format_section('Device OS', os_tbl))
    lines.append('')
    lines.append('Top under-represented (most negative diff%):')
    for label, o, t, d in under:
        lines.append(f"  - {label}: {o:.3f} | {t:.3f} | {d:+.3f}")
    lines.append('Top over-represented (most positive diff%):')
    for label, o, t, d in over:
        lines.append(f"  - {label}: {o:.3f} | {t:.3f} | {d:+.3f}")

    return '\n'.join(lines)


def normalize_race(value: str) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if 'chinese' in v:
        return 'Chinese'
    if 'malay' in v:
        return 'Malay'
    if 'indian' in v or 'sikh' in v or 'tamil' in v:
        return 'Indian'
    return 'Others'


def normalize_sex(value: str) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if v in {'m', 'male'}:
        return 'Male'
    if v in {'f', 'female'}:
        return 'Female'
    return None


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
    return 'no_info'


def normalize_os(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip().lower()
    if 'apple' in v or v == 'ios':
        return 'iOS'
    if 'android' in v or any(b in v for b in ['samsung', 'google', 'oppo', 'xiaomi', 'huawei', 'oneplus', 'vivo', 'realme', 'sony', 'motorola', 'nothing']):
        return 'Android'
    return 'Android'


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute variance report against targets per group, with Device OS section (30% iOS / 70% Android).')
    parser.add_argument('--participants', '-p', type=str, required=True, help='Path to updated participants CSV.')
    parser.add_argument('--targets', '-t', type=str, required=True, help='Path to targets CSV.')
    parser.add_argument('--group-size', type=int, default=100, help='Expected group size (for sanity checks).')
    args = parser.parse_args()

    targets, age_groups_order = load_targets(args.targets)
    target_marginals = compute_marginal_targets(targets)
    age_bucket = make_age_bucketer(age_groups_order)
    eighteen_to_twentyfour_labels = {lbl for lbl in age_groups_order if ('18-19' in lbl or '20-24' in lbl)}

    with open(args.participants, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        colmap = detect_columns(headers)
        rows = list(reader)

    # Determine group column
    gcol = colmap['group'] or 'Group'
    scol = colmap['status'] or 'Status'

    # Include rows with Status in {'TO_CONTACT', 'CONFIRMED'}
    def norm_row(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        status = (r.get(scol, '') or '').strip().upper()
        if status not in {'TO_CONTACT', 'CONTACTED', 'CONFIRMED'}:
            return None
        sex = normalize_sex(r.get(colmap['sex'] or '', ''))
        race = normalize_race(r.get(colmap['race'] or '', ''))
        edu = normalize_education(r.get(colmap['education_level'] or '', ''))
        dob_raw = r.get(colmap['dob'] or '', '')
        dob = parse_date_maybe(dob_raw) if dob_raw else None
        if not dob:
            return None
        age = compute_age(dob)
        if age < 18:
            return None
        age_group = age_bucket(age)
        if not age_group:
            return None
        if age_group in eighteen_to_twentyfour_labels:
            edu = 'no_info'
        try:
            grp = int((r.get(gcol, '') or '').strip())
        except Exception:
            return None
        os_norm = normalize_os(r.get(colmap['os'] or '', ''))
        return {
            '__norm_sex': sex or 'Male',
            '__norm_race': race or 'Others',
            '__norm_education_level': edu,
            '__norm_age_group': age_group,
            '__norm_os': os_norm or 'Android',
            '__group': grp,
        }

    normalized = [nr for nr in (norm_row(r) for r in rows) if nr]

    by_group: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in normalized:
        by_group[r['__group']].append(r)

    for g in sorted(by_group.keys()):
        print(f"===== Group {g} (n={len(by_group[g])}) =====")
        print(report_for_group(by_group[g], target_marginals))
        print()


if __name__ == '__main__':
    main()
