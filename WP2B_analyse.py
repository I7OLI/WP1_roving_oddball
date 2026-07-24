"""
WP2B — offline analysis of saved pre/post JSON
==============================================

Edit the CONFIG block at the top, then just hit Run.

The conditioned dimension is NOT stored in the JSON, so you have to set it
here yourself.


WHY THIS SEPARATES TWO EFFECTS
------------------------------
A pre->post change in PSE can come from two quite different things, and the
naive readout confuses them.

  LATERAL BIAS   — everything sounds shifted toward one side. Adds the SAME
                   signed constant to every PSE.
  CUE WEIGHTING  — one cue counts for more. Changes the MAGNITUDE of the
                   rove needed to cancel the anchor, so it moves the
                   anchor_LEFT and anchor_RIGHT PSEs in OPPOSITE signed
                   directions.

With anchor_LEFT giving PSE_L and anchor_RIGHT giving PSE_R:

    bias  b = (PSE_L + PSE_R) / 2        <- the "shifted to a side" part
    half  k = (PSE_L - PSE_R) / 2        <- the cue-weighting part

k is bias-free by construction, so weights computed from k are immune to a
lateral shift. That is the key improvement over using raw PSEs.
"""

import glob
import json
import os

import numpy as np

# ════════════════════════════════════════════════════════════════════
#  CONFIG  —  edit these, then Run
# ════════════════════════════════════════════════════════════════════

# Where the result files live.
#   ''  (empty)  ->  the folder this script is in. Works on any machine.
#   otherwise    ->  an absolute path, e.g.
#                    Mac      '/Users/oliver/PycharmProjects/WP1_roving_oddball'
#                    Windows  r'C:\Users\oliver\WP1_roving_oddball'
#                    (note the r'...' on Windows, or backslashes get eaten)
DATA_DIR = ''

# Which participant to analyse.
#   None  ->  every pre/post pair in DATA_DIR, plus a group summary
#   '1'   ->  just sub-1
#   ['1', '3']  ->  only those
SUBJECT = None

# Which cue was CS+ during conditioning: 'ITD' or 'ILD'.
# NOTE: new-format sub-*_WP2B.json files record this themselves, and the
# value in the file always wins. This setting is only a fallback for the
# OLD split sub-*_pre.json / sub-*_post.json files, which stored no such
# field — for those you have to supply it from your lab notes.
COND_DIM = 'ITD'

# Per-subject overrides for that fallback, if participants differed.
#   e.g.  SUBJECT_DIMS = {'2': 'ILD', '5': 'ILD'}
SUBJECT_DIMS = {}

# Filename patterns. Both layouts are read automatically.
COMBINED_SUFFIX = '_WP2B.json'      # new: one file, metadata included
PRE_SUFFIX      = '_pre.json'       # legacy split files
POST_SUFFIX     = '_post.json'
SUBJECT_PREFIX  = 'sub-'

# ════════════════════════════════════════════════════════════════════

ANCHOR_DEG = 10
CONDITIONS = ['ITD_anchor_LEFT', 'ITD_anchor_RIGHT',
              'ILD_anchor_LEFT', 'ILD_anchor_RIGHT']


# ════════════════════════════════════════════════════════════════════
# PSYCHOMETRIC FIT  (kept identical to WP2B_stimuli_updated.py)
# ════════════════════════════════════════════════════════════════════

def fit_psychometric(results, condition):
    data = [r for r in results if r['condition'] == condition
            and r['response'] in ('left', 'right')]
    if not data:
        return None
    xs = sorted(set(r['rove_val'] for r in data))
    p_right = []
    for x in xs:
        sub = [r for r in data if r['rove_val'] == x]
        p_right.append(sum(r['response'] == 'right' for r in sub) / len(sub))
    return _pse_from_curve(np.array(xs, float), np.array(p_right, float))


def _pse_from_curve(x, p):
    def logistic(x, alpha, beta):
        return 1.0 / (1.0 + np.exp(-(x - alpha) / beta))
    try:
        from scipy.optimize import curve_fit
        (alpha, _beta), _ = curve_fit(
            logistic, x, p, p0=[float(np.mean(x)), 5.0],
            bounds=([x.min() - 20, 1e-3], [x.max() + 20, 100.0]),
            maxfev=10000)
        return float(alpha)
    except Exception:
        for i in range(len(x) - 1):
            if (p[i] - 0.5) * (p[i + 1] - 0.5) <= 0 and p[i] != p[i + 1]:
                frac = (0.5 - p[i]) / (p[i + 1] - p[i])
                return float(x[i] + frac * (x[i + 1] - x[i]))
        return float(np.interp(0.5, p, x)) if p[0] < p[-1] else None


# ════════════════════════════════════════════════════════════════════
# BIAS / WEIGHT DECOMPOSITION
# ════════════════════════════════════════════════════════════════════

def decompose(results, anchored_cue):
    """For one anchored-cue family, return (bias, half_range, w_rove_norm).

    anchored_cue : 'ITD' or 'ILD' — which cue is held fixed.
    """
    pse_l = fit_psychometric(results, f'{anchored_cue}_anchor_LEFT')
    pse_r = fit_psychometric(results, f'{anchored_cue}_anchor_RIGHT')
    if pse_l is None or pse_r is None:
        return None, None, None

    bias = (pse_l + pse_r) / 2.0        # lateral shift, cue-weight-free
    half = (pse_l - pse_r) / 2.0        # cue weighting, bias-free

    if half <= 0:
        return bias, half, None         # cues never cancelled in range

    ratio = ANCHOR_DEG / half           # w_rove / w_anchor
    return bias, half, ratio / (1.0 + ratio)


def analyse(pre, post, cond_dim, label=''):
    other = 'ILD' if cond_dim == 'ITD' else 'ITD'

    print(f"\n{'=' * 68}")
    print(f"  {label}   conditioned cue (CS+) = {cond_dim}")
    print(f"{'=' * 68}")

    # ── raw PSEs, for eyeballing ──
    print("\n  raw PSEs (sign flips between LEFT/RIGHT anchors by design):")
    for cond in CONDITIONS:
        a, b = fit_psychometric(pre, cond), fit_psychometric(post, cond)
        if a is None or b is None:
            print(f"    {cond:>18s}:  (insufficient data)")
        else:
            print(f"    {cond:>18s}:  pre={a:+7.2f}  post={b:+7.2f}"
                  f"   Δ={b - a:+6.2f}°")

    # ── decomposition ──
    print(f"\n  {'':18s}  {'LATERAL BIAS (°)':>26s}   {'CUE WEIGHTING':>26s}")
    print(f"  {'anchored cue':18s}  {'pre':>8s}{'post':>9s}{'Δ':>9s}   "
          f"{'pre':>8s}{'post':>9s}{'Δ':>9s}")

    bias_deltas, w_deltas = [], []

    for anchored in ('ITD', 'ILD'):
        b0, k0, w0 = decompose(pre,  anchored)
        b1, k1, w1 = decompose(post, anchored)
        if b0 is None or b1 is None:
            print(f"  {anchored + ' anchored':18s}  (insufficient data)")
            continue

        # weight of the CONDITIONED cue in this family
        roving = 'ILD' if anchored == 'ITD' else 'ITD'
        if w0 is None or w1 is None:
            wc0 = wc1 = None
        elif cond_dim == roving:
            wc0, wc1 = w0, w1
        else:
            wc0, wc1 = 1 - w0, 1 - w1

        bias_deltas.append(b1 - b0)
        wtxt = "     n/a     n/a      n/a"
        if wc0 is not None:
            w_deltas.append(wc1 - wc0)
            wtxt = f"{wc0:8.3f}{wc1:9.3f}{wc1 - wc0:+9.3f}"

        print(f"  {anchored + ' anchored':18s}  "
              f"{b0:8.2f}{b1:9.2f}{b1 - b0:+9.2f}   {wtxt}")

    # ── verdict ──
    print(f"\n  {'-' * 64}")
    if not w_deltas:
        print("  No interpretable weighting estimate.")
        return None

    mean_bias = float(np.mean(bias_deltas))
    mean_w    = float(np.mean(w_deltas))

    print(f"  lateral bias shift   Δb = {mean_bias:+6.2f}°   "
          f"(pure side effect — NOT cue weighting)")
    print(f"  weight of {cond_dim:>3s}        Δw = {mean_w:+6.3f}    "
          f"(0.5 = {cond_dim} and {other} equally weighted)")

    print()
    if mean_w > 0:
        print(f"  → weighting shifted TOWARD the conditioned cue ({cond_dim}).")
    else:
        print(f"  → weighting shifted AWAY from {cond_dim} (toward {other}).")

    # is the "shift to a side" just bias?
    if abs(mean_bias) > 2 * abs(mean_w) * ANCHOR_DEG:
        print(f"  ⚠  The lateral bias ({mean_bias:+.2f}°) dominates the "
              f"weighting change.")
        print(f"     What looks like 'a shift to one side' is mostly a "
              f"lateral bias,")
        print(f"     not a change in cue weighting. Bias direction that "
              f"varies randomly")
        print(f"     across participants is what noise looks like — check "
              f"consistency")
        print(f"     of the SIGN of Δw across your sample, not Δb.")

    return {'label': label, 'cond_dim': cond_dim,
            'delta_bias': mean_bias, 'delta_weight': mean_w}


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def _wanted_subjects():
    if SUBJECT is None:
        return None
    if isinstance(SUBJECT, str):
        return [SUBJECT]
    return list(SUBJECT)


def find_sessions():
    """Return [(sid, label, pre, post, cs_plus_dim, dim_source)].

    Reads the new combined sub-*_WP2B.json AND the legacy split
    sub-*_pre.json / sub-*_post.json, so old data still works.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.expanduser(DATA_DIR) if DATA_DIR else here

    if not os.path.isdir(folder):
        msg = [f"DATA_DIR does not exist:", f"    {folder}", ""]
        msg.append(f"This script lives in:")
        msg.append(f"    {here}")
        msg.append(f"Working directory is:")
        msg.append(f"    {os.getcwd()}")
        # where ARE the result files? look in the obvious places
        found = []
        for cand in {here, os.getcwd(), os.path.expanduser('~')}:
            for pat in (COMBINED_SUFFIX, PRE_SUFFIX):
                hits = glob.glob(os.path.join(cand, '*' + pat))
                if hits:
                    found.append((cand, len(hits)))
                    break
        if found:
            msg.append("")
            msg.append("Result files were found in:")
            for cand, n in found:
                msg.append(f"    {cand}   ({n} file(s))")
            msg.append("")
            msg.append("Set DATA_DIR to one of those, or DATA_DIR = '' "
                       "to use the script's own folder.")
        else:
            msg.append("")
            msg.append("No WP2B result files found nearby either. Search for "
                       "them with:")
            msg.append("    find ~ -name 'sub-*_WP2B.json' -o -name "
                       "'sub-*_pre.json'")
        raise SystemExit("\n".join(msg))

    wanted = _wanted_subjects()
    sessions, seen = [], set()

    def sid_of(stem):
        return stem[len(SUBJECT_PREFIX):] if stem.startswith(SUBJECT_PREFIX) else stem

    # ── new combined format (preferred) ──
    for path in sorted(glob.glob(os.path.join(folder, '*' + COMBINED_SUFFIX))):
        stem = os.path.basename(path)[:-len(COMBINED_SUFFIX)]
        sid = sid_of(stem)
        if wanted is not None and sid not in wanted and stem not in wanted:
            continue
        with open(path) as f:
            rec = json.load(f)
        dim = rec.get('cs_plus_dim')
        if dim not in ('ITD', 'ILD'):
            print(f"  ! {os.path.basename(path)}: no valid 'cs_plus_dim' "
                  f"— falling back to CONFIG")
            dim, src = SUBJECT_DIMS.get(sid, COND_DIM), 'CONFIG'
        else:
            src = 'file'
        sessions.append((sid, stem, rec['pre'], rec['post'], dim, src))
        seen.add(sid)

    # ── legacy split format ──
    for pre_path in sorted(glob.glob(os.path.join(folder, '*' + PRE_SUFFIX))):
        stem = os.path.basename(pre_path)[:-len(PRE_SUFFIX)]
        sid = sid_of(stem)
        if wanted is not None and sid not in wanted and stem not in wanted:
            continue
        if sid in seen:
            print(f"  ! {stem}: combined file already loaded, "
                  f"ignoring legacy {PRE_SUFFIX}/{POST_SUFFIX}")
            continue
        post_path = pre_path[:-len(PRE_SUFFIX)] + POST_SUFFIX
        if not os.path.exists(post_path):
            print(f"  ! {os.path.basename(pre_path)} has no "
                  f"{POST_SUFFIX} partner — skipping")
            continue
        with open(pre_path) as f:
            pre = json.load(f)
        with open(post_path) as f:
            post = json.load(f)
        sessions.append((sid, stem, pre, post,
                         SUBJECT_DIMS.get(sid, COND_DIM), 'CONFIG'))
        seen.add(sid)

    for post_path in sorted(glob.glob(os.path.join(folder, '*' + POST_SUFFIX))):
        stem = os.path.basename(post_path)[:-len(POST_SUFFIX)]
        if sid_of(stem) not in seen and not os.path.exists(
                post_path[:-len(POST_SUFFIX)] + PRE_SUFFIX):
            print(f"  ! {os.path.basename(post_path)} has no "
                  f"{PRE_SUFFIX} partner — skipping")

    if not sessions:
        if wanted is not None:
            raise SystemExit(
                f"No files matching SUBJECT = {SUBJECT!r} in:\n    {folder}\n"
                f"Expected {SUBJECT_PREFIX}{wanted[0]}{COMBINED_SUFFIX} "
                f"or {SUBJECT_PREFIX}{wanted[0]}{PRE_SUFFIX}")
        raise SystemExit(f"No WP2B result files found in:\n    {folder}")

    return sessions


def main():
    sessions = find_sessions()
    print(f"\nDATA_DIR : {DATA_DIR}")
    print(f"SUBJECT  : {'all' if SUBJECT is None else SUBJECT}")
    print(f"Found {len(sessions)} session(s).")

    if any(src == 'CONFIG' for *_, src in sessions):
        print(f"\n  NOTE: some sessions have no CS+ recorded in the file.\n"
              f"  For those, CS+ is assumed to be COND_DIM = {COND_DIM!r} "
              f"(see SUBJECT_DIMS).\n"
              f"  If that is wrong, the sign of every weighting result "
              f"for them is wrong.")

    summary = []
    for sid, label, pre, post, dim, src in sessions:
        out = analyse(pre, post, dim, f"{label}   [CS+ from {src}]")
        if out:
            summary.append(out)

    # ── across participants ──
    if len(summary) > 1:
        print(f"\n{'=' * 68}")
        print(f"  ACROSS {len(summary)} PARTICIPANTS")
        print(f"{'=' * 68}")
        dw = np.array([s['delta_weight'] for s in summary])
        db = np.array([s['delta_bias'] for s in summary])
        print(f"\n  Δweight(CS+):   mean={dw.mean():+.3f}  "
              f"SD={dw.std(ddof=1):.3f}  "
              f"| same sign in {max((dw > 0).sum(), (dw < 0).sum())}/{len(dw)}")
        print(f"  Δbias:          mean={db.mean():+.2f}°  "
              f"SD={db.std(ddof=1):.2f}°  "
              f"| same sign in {max((db > 0).sum(), (db < 0).sum())}/{len(db)}")
        print("\n  A real conditioning effect shows CONSISTENT SIGN in Δweight.")
        print("  Δbias flipping sign across participants is consistent with noise.")


if __name__ == '__main__':
    main()
