"""
WP2B — offline analysis of saved pre/post JSON
==============================================

    python WP2B_analyse.py sub-1_pre.json  --dim ITD   # partner found automatically
    python WP2B_analyse.py sub-1_post.json --dim ITD   # either half works
    python WP2B_analyse.py ./data          --dim ITD   # every pair in a folder
    python WP2B_analyse.py --pre a.json --post b.json --dim ITD   # explicit

The conditioned dimension is NOT stored in the JSON, so --dim is required.


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

import argparse
import glob
import json
import os

import numpy as np

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

def resolve_pairs(path):
    """Accept a folder OR either half of a pre/post pair, return [(pre, post)].

    Pointing at sub-1_pre.json finds sub-1_post.json automatically, and
    vice versa — so it does not matter which half you name.
    """
    path = os.path.expanduser(path)

    if os.path.isdir(path):
        pairs = []
        for pre_path in sorted(glob.glob(os.path.join(path, '*_pre.json'))):
            post_path = pre_path[:-len('_pre.json')] + '_post.json'
            if os.path.exists(post_path):
                pairs.append((pre_path, post_path))
            else:
                print(f"  ! {os.path.basename(pre_path)} has no _post partner "
                      f"— skipping")
        # warn about orphaned post files too, rather than silently ignoring
        for post_path in sorted(glob.glob(os.path.join(path, '*_post.json'))):
            if not os.path.exists(post_path[:-len('_post.json')] + '_pre.json'):
                print(f"  ! {os.path.basename(post_path)} has no _pre partner "
                      f"— skipping")
        return pairs

    if os.path.isfile(path):
        if path.endswith('_pre.json'):
            pre_path = path
            post_path = path[:-len('_pre.json')] + '_post.json'
        elif path.endswith('_post.json'):
            post_path = path
            pre_path = path[:-len('_post.json')] + '_pre.json'
        else:
            raise SystemExit(f"'{path}' is not a *_pre.json or *_post.json file.")

        missing = [p for p in (pre_path, post_path) if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"Missing partner file: {missing[0]}")
        return [(pre_path, post_path)]

    raise SystemExit(f"No such file or directory: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', nargs='?',
                    help='a folder of results, OR either half of a pre/post '
                         'pair (the partner is found automatically)')
    ap.add_argument('--pre', help='explicit pre file (overrides positional)')
    ap.add_argument('--post', help='explicit post file (overrides positional)')
    ap.add_argument('--dim', required=True, choices=['ITD', 'ILD'],
                    help='conditioned dimension (not stored in the JSON)')
    args = ap.parse_args()

    if args.pre and args.post:
        pairs = [(args.pre, args.post)]
    elif args.path:
        pairs = resolve_pairs(args.path)
    else:
        ap.error('give a folder or file path, or both --pre and --post')

    if not pairs:
        print("No pre/post pairs found.")
        return

    print(f"Found {len(pairs)} pre/post pair(s).")

    summary = []
    for pre_path, post_path in pairs:
        with open(pre_path) as f:
            pre = json.load(f)
        with open(post_path) as f:
            post = json.load(f)
        label = os.path.basename(pre_path).replace('_pre.json', '')
        out = analyse(pre, post, args.dim, label)
        if out:
            summary.append(out)

    # ── across participants ──
    if len(summary) > 1:
        print(f"\n{'=' * 68}")
        print(f"  ACROSS {len(summary)} PARTICIPANTS")
        print(f"{'=' * 68}")
        dw = np.array([s['delta_weight'] for s in summary])
        db = np.array([s['delta_bias'] for s in summary])
        print(f"\n  Δweight({args.dim}):  mean={dw.mean():+.3f}  "
              f"SD={dw.std(ddof=1):.3f}  "
              f"| same sign in {max((dw > 0).sum(), (dw < 0).sum())}/{len(dw)}")
        print(f"  Δbias:          mean={db.mean():+.2f}°  "
              f"SD={db.std(ddof=1):.2f}°  "
              f"| same sign in {max((db > 0).sum(), (db < 0).sum())}/{len(db)}")
        print("\n  A real conditioning effect shows CONSISTENT SIGN in Δweight.")
        print("  Δbias flipping sign across participants is consistent with noise.")


if __name__ == '__main__':
    main()
