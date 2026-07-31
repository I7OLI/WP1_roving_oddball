"""
WP2B — affective cue weighting (ITD vs ILD)
===========================================
 Method of constant stimuli with a two-tone judgment:

  Every trial plays a REFERENCE tone (centre, 0°) then a TEST tone.
  Participant answers: was the second tone LEFT or RIGHT of the first?

The test tone carries two conflicting spatial cues (ITD and ILD). One cue
is anchored (fixed); the other roves across a fixed set of probe values.
Fitting P("right") against the roving value gives the PSE — the point
where the two cues cancel. That PSE is the cue-weighting readout.

Three phases:
  1. Pre  conflict block   → PSE per condition
  2. Conditioning          → one cue dimension paired with shock (CS+/CS-)
  3. Post conflict block    → same conditions, look for a PSE shift
"""
# %% 
import time
import json
import os
import datetime
import random
import numpy as np
import slab
import freefield as ff
from psychopy import event, core
from psychopy.hardware import keyboard


# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════

FS        = 48828.125     # TDT sample rate
RCX       = 'shock.rcx'
PROC      = 'RM1'

TONE_MS   = 150           # tone duration
RAMP_MS   = 10            # on/off ramp
F0        = 700           # centre frequency of the noise band (Hz)
BANDWIDTH_OCT = 1 / 3     # one-third-octave noise band
HEAD_RADIUS_CM = 8.75     # slab default; used for natural ITD mapping
LEVEL     = 65            # dB SPL (nominal; requires playback calibration)
SILENCE   = slab.Sound.silence(0.1,FS,2)
ANCHOR_DEG = 10           # fixed anchor position (1
# degrees)
# Rove only on the OPPOSITE side of the anchor:
#   anchor LEFT  (−20°) → rove 0°..+30° (right side)
#   anchor RIGHT (+20°) → rove 0°..−30° (left side)
PROBE_VALS     = [0, 2.5, 5, 7.5, 10, 12.5, 15, 20]   # roving cue magnitudes
REPS_PER_PROBE = 2       # trials per probe value per condition

# Conditioning ITI is JITTERED so the US is not perfectly predictable in
# time. (The old code used a single ITI_MS = 1500 and then slept to
# `t0 + ITI_MS/1000/2`, i.e. 750 ms from tone onset — half the advertised
# value, fixed, with 1500 written into the saved config regardless.)
ITI_MIN_MS     = 1500
ITI_MAX_MS     = 2500
RESP_TIMEOUT_MS = 2000    # response window
POST_RESP_MS   = 200      # conflict blocks: gap AFTER response before next trial
                          # (self-paced — fast responders get through faster)

# Conditioning
# N_COND_TRIALS must be divisible by 4 * len(PROBE_VALS[1:]) = 28 so that
# side × magnitude is fully crossed within BOTH CS+ and CS-. 56 gives
# 28 per CS type = 14 per side, of which 12/12 are reinforced (85.7%).
N_COND_TRIALS = 56        # total conditioning trials
SHOCK_RATE    = 0.85      # CS+ reinforcement rate (realised: 12/14 = 85.7%)
CS_US_INTERVAL = 250      # ms, tone onset → shock
MAX_SAME_SIDE_RUN = 4     # sequence constraint: consecutive same-side trials
MAX_UNREINF_RUN   = 2     # sequence constraint: consecutive unreinforced CS+
ORDER_MAX_TRIES   = 5000  # rejection-sampling budget for the above

CONDITIONS = ['ITD_anchor_LEFT', 'ITD_anchor_RIGHT',
              'ILD_anchor_LEFT', 'ILD_anchor_RIGHT']
kb = keyboard.Keyboard()
# %%
# ════════════════════════════════════════════════════════════════════
# STIMULI
# ════════════════════════════════════════════════════════════════════
# slab's natural ILD mapping is based on a KEMAR interaural-level
# spectrum. Cache it because loading/recomputing it on every trial would
# be unnecessarily slow.
_KEMAR_ILS = None


def _get_kemar_ils():
    """Load slab's bundled KEMAR interaural-level spectrum once."""
    global _KEMAR_ILS
    if _KEMAR_ILS is None:
        _KEMAR_ILS = slab.Binaural.make_interaural_level_spectrum()
    return _KEMAR_ILS


def third_octave_edges(center_hz):
    """Return the lower and upper edge of a 1/3-octave band."""
    half_band = 2 ** (BANDWIDTH_OCT / 2)
    return center_hz / half_band, center_hz * half_band


def make_noise_token():
    """Generate one monaural, 1/3-octave noise token.

    The same returned token should be passed to make_ref_sound() and
    make_stimulus() within a trial. A new token is generated on each
    trial, but reference and test therefore have identical waveforms.
    """
    low_hz, high_hz = third_octave_edges(F0)
    token = slab.Sound.whitenoise(
        duration=TONE_MS / 1000,
        samplerate=FS,
    )
    token = token.filter(kind='bp', frequency=(low_hz, high_hz))
    token = token.ramp('both', RAMP_MS / 1000)
    token.level = LEVEL
    return token


def make_ref_sound(noise_token=None):
    """Create the centred reference (ITD = 0, ILD = 0).

    Pass the same noise_token to make_stimulus() to obtain a matched
    reference/test pair. If omitted, a fresh token is generated.
    """
    if noise_token is None:
        noise_token = make_noise_token()
    return slab.Binaural(noise_token)


def itd_deg_to_seconds(itd_deg):
    """Map nominal ITD azimuth (degrees) to natural-head ITD seconds."""
    return float(slab.Binaural.azimuth_to_itd(
        azimuth=float(itd_deg),
        frequency=F0,
        head_radius=HEAD_RADIUS_CM,
    ))


def ild_deg_to_db(ild_deg):
    """Map nominal ILD azimuth (degrees) to a KEMAR-derived ILD in dB.

    slab.azimuth_to_ild() returns separate level changes for the left
    and right ears. Their difference is the physical ILD (right minus
    left). Applying that scalar through Binaural.ild() preserves the
    mean binaural level, avoiding an overall-level cue.
    """
    left_db, right_db = slab.Binaural.azimuth_to_ild(
        azimuth=float(ild_deg),
        frequency=F0,
        ils=_get_kemar_ils(),
    )
    return float(right_db - left_db)


def make_stimulus(itd_deg, ild_deg, noise_token=None):
    """Create a test sound with independently specified ITD and ILD.

    Parameters
    ----------
    itd_deg : float
        Nominal azimuth used only to obtain the natural ITD. Negative
        values lateralize left; positive values lateralize right.
    ild_deg : float
        Nominal azimuth used only to obtain the natural KEMAR ILD.
        Negative values lateralize left; positive values lateralize right.
    noise_token : slab.Sound or None
        Monaural carrier token. Pass the same token used for the reference
        so the two intervals differ only in their binaural cues.
    """
    if noise_token is None:
        noise_token = make_noise_token()

    itd_seconds = itd_deg_to_seconds(itd_deg)
    ild_db = ild_deg_to_db(ild_deg)

    stimulus = slab.Binaural(noise_token)
    stimulus = stimulus.itd(duration=itd_seconds)
    stimulus = stimulus.ild(dB=ild_db)
    return stimulus

def get_response():
    """Return 'left', 'right', or None (timeout)."""
    kb.clearEvents()
    keys = kb.waitKeys(maxWait=RESP_TIMEOUT_MS / 1000,
                       keyList=['1', '4'], waitRelease=False)
    if keys:
        return 'left' if keys[0].name == '1' else 'right'
    return None

# ════════════════════════════════════════════════════════════════════
# PLAY ONE TWO-TONE TRIAL  (reference, then test → judgement)
# ════════════════════════════════════════════════════════════════════
def precise_sleep_until(target_time, busy_wait_threshold=0.002):
    """Sleep until target_time, busy-waiting the last ~2 ms for accuracy."""
    remaining = target_time - time.time()
    if remaining > busy_wait_threshold:
        time.sleep(remaining - busy_wait_threshold)
    while time.time() < target_time:
        pass


def _makeSound(itd, ild):
    """Reference + ISI silence + test as ONE buffer.

    Concatenating makes the reference->test interval sample-exact, instead
    of it being whatever the USB write between them happened to take.
    """
    token = make_noise_token()
    reference = make_ref_sound(token)
    test = make_stimulus(itd, ild, token)
    return slab.Binaural(slab.Sound.sequence(reference, SILENCE, test))


def trial_to_itd_ild(t):
    """Map a trial dict onto (itd_deg, ild_deg)."""
    if t['anchor_cue'] == 'ITD':
        return t['anchor_val'], t['rove_val']
    return t['rove_val'], t['anchor_val']


def _write(sound):
    """Transfer a buffer to the RM1. Must complete before _play()."""
    ff.write('playbuflen', len(sound), PROC)
    ff.write('data_l', sound.left.data, PROC)
    ff.write('chan_l', 1, PROC)
    ff.write('data_r', sound.right.data, PROC)
    ff.write('chan_r', 2, PROC)


def _play():
    """Trigger the buffer already loaded on the RM1, then collect a response.

    Takes no argument on purpose: writing and playing are decoupled so the
    write for trial i+1 can happen in trial i's dead time.
    Returns (response, onset_timestamp).
    """
    t_onset = time.time()
    ff.play(1, [PROC])
    ff.wait_to_finish_playing()
    resp = get_response()
    t_resp = time.time()   # when the key landed (or the response window timed out)
    return resp, t_onset, t_resp

# ════════════════════════════════════════════════════════════════════
# TRIAL LIST  — every condition × probe × repetition, shuffled
# ════════════════════════════════════════════════════════════════════
def generate_conflict_trials(phase='pre'):
    trials = []
    for cond in CONDITIONS:
        is_right_anchor = 'RIGHT' in cond
        anchor_cue = 'ITD' if 'ITD' in cond else 'ILD'
        rove_cue = 'ILD' if anchor_cue == 'ITD' else 'ITD'
        anchor_val = ANCHOR_DEG if is_right_anchor else -ANCHOR_DEG
        for probe in PROBE_VALS:
            rove_val = -probe if is_right_anchor else probe  # opposite side
            for _ in range(REPS_PER_PROBE):
                trials.append({
                    'condition':  cond,
                    'anchor_cue': anchor_cue,
                    'rove_cue':   rove_cue,
                    'anchor_val': anchor_val,
                    'rove_val':   rove_val,
                    'phase':      phase,
                })
    random.shuffle(trials)
    return trials


# ════════════════════════════════════════════════════════════════════
# RUN ONE CONFLICT BLOCK
# ════════════════════════════════════════════════════════════════════
def run_conflict_block(trials, block_label):
    """Play one block with all stimulus generation hoisted out of the loop
    and the RM1 write for trial i+1 pipelined into trial i's ITI."""
    print(f"\n--- Block: {block_label} ({len(trials)} trials) ---")

    # ── generation is the expensive part (noise, bandpass, 2048-tap ITD
    #    delay). The trial list is fully known up front, so build every
    #    buffer before the block starts. Each still gets a fresh token.
    print(f"    building {len(trials)} stimuli...", end='', flush=True)
    sounds = [_makeSound(*trial_to_itd_ild(t)) for t in trials]
    print(" done.")

    # ── trial 0 has no preceding ITI to hide its write in, so write it here
    _write(sounds[0])

    results = []
    for i, t in enumerate(trials):
        resp, t_onset, t_resp = _play()

        # ── dead time starts here. Load the next buffer now so that the
        #    next ff.play() is preceded by zero USB transfer. The USB write
        #    is hidden inside the post-response gap below.
        if i + 1 < len(sounds):
            _write(sounds[i + 1])

        # NB: no 'block' key — it duplicated 'phase', which generate_conflict_
        # trials already set. Two fields that must agree is one field too many.
        results.append(dict(t, response=resp, onset=t_onset, trial_num=i + 1))

        # ── self-paced: fixed short gap measured from the RESPONSE, not onset.
        #    A fast responder moves on sooner; a slow one still gets the same
        #    200 ms breather. (Timeouts pace off the end of the response window.)
        precise_sleep_until(t_resp + POST_RESP_MS / 1000)

    return results


# ════════════════════════════════════════════════════════════════════
# CONDITIONING  (one cue dimension → CS+ shocked / CS- safe)
# ════════════════════════════════════════════════════════════════════
def _balanced_cs_set(cue, is_plus, n):
    """Fully crossed side × magnitude trial set for ONE CS type.

    Every (side, magnitude) cell gets an equal number of trials, so
    `n` must be divisible by 2 * len(PROBE_VALS[1:]).

    For CS+, reinforcement is allocated as a FIXED count split evenly
    across sides rather than an independent coin flip per trial.
    """
    mags    = PROBE_VALS[1:]
    n_cells = 2 * len(mags)
    if n % n_cells:
        raise ValueError(
            f"{n} trials per CS type does not divide into {n_cells} "
            f"(side × magnitude) cells. Set N_COND_TRIALS to a multiple "
            f"of {2 * n_cells}.")
    reps = n // n_cells

    trials = []
    for side in (-1, +1):
        for m in mags:
            for _ in range(reps):
                trials.append({'is_plus': is_plus, 'cue': cue, 'side': side,
                               'pos': side * m, 'shocked': False})
    if not is_plus:
        return trials

    per_side  = n // 2
    n_sh_side = int(round(SHOCK_RATE * per_side))
    for side in (-1, +1):
        idx = [i for i, t in enumerate(trials) if t['side'] == side]
        random.shuffle(idx)
        for i in idx[:n_sh_side]:          # which magnitudes go unreinforced
            trials[i]['shocked'] = True    # is left random — only 2/side
    return trials


def _order_ok(order):
    """Sequence constraints: no long laterality runs, no extinction streaks."""
    run_side = run_unreinf = 1
    if order[0]['is_plus'] and not order[0]['shocked']:
        return False                       # don't open on an unreinforced CS+
    for a, b in zip(order, order[1:]):
        run_side = run_side + 1 if a['side'] == b['side'] else 1
        if run_side > MAX_SAME_SIDE_RUN:
            return False
        b_unreinf = b['is_plus'] and not b['shocked']
        a_unreinf = a['is_plus'] and not a['shocked']
        run_unreinf = run_unreinf + 1 if (a_unreinf and b_unreinf) else 1
        if run_unreinf > MAX_UNREINF_RUN:
            return False
    return True


def run_conditioning(cond_dim):
    """Condition a cue TYPE, not a side.

    cond_dim : 'ITD' or 'ILD' — the cue type that is CS+ (shocked).
    The other cue type is CS- (safe).

    Laterality is COUNTERBALANCED, not merely randomised. Previously side
    was drawn per trial with random.choice([-1, +1]) and shock with an
    independent Bernoulli(SHOCK_RATE); over 20k simulated participants that
    left the *shocked* subset lateralised by ≥4 trials in 44% of runs and
    ≥6 in 23%. Because the DV is a signed PSE shift, a lateralised US
    schedule induces side-specific threat learning that loads unequally on
    the anchor_LEFT vs anchor_RIGHT conditions — precisely the confound the
    cue-weight measure assumes away. Side and magnitude are now fully
    crossed within each CS type, and the reinforced subset is split evenly
    across sides by construction.
    """
    other_dim = 'ILD' if cond_dim == 'ITD' else 'ITD'
    n_each    = N_COND_TRIALS // 2

    trials = (_balanced_cs_set(cond_dim,  True,  n_each) +
              _balanced_cs_set(other_dim, False, n_each))

    for _ in range(ORDER_MAX_TRIES):
        random.shuffle(trials)
        if _order_ok(trials):
            break
    else:
        print(f"  ! no ordering met the run-length constraints in "
              f"{ORDER_MAX_TRIES} tries — using an unconstrained shuffle")
        random.shuffle(trials)

    for t in trials:                       # jittered ITI, drawn up front
        t['iti_ms'] = random.uniform(ITI_MIN_MS, ITI_MAX_MS)

    # realised balance — printed and saved so it can be checked post hoc
    plus    = [t for t in trials if t['is_plus']]
    shocked = [t for t in plus if t['shocked']]
    print(f"  balance check — CS+ L/R: "
          f"{sum(t['side'] < 0 for t in plus)}/{sum(t['side'] > 0 for t in plus)}"
          f" | shocked L/R: "
          f"{sum(t['side'] < 0 for t in shocked)}/{sum(t['side'] > 0 for t in shocked)}"
          f" | reinforcement: {len(shocked)}/{len(plus)} "
          f"({len(shocked) / len(plus):.1%})")

    log = []
    print(f"\n--- Conditioning (CS+ = {cond_dim}, CS- = {other_dim}, "
          f"{len(trials)} trials) ---")
    # single-interval stimuli here (no reference, no ISI), built up front
    stims = [make_stimulus(t['pos'], 0) if t['cue'] == 'ITD'
             else make_stimulus(0, t['pos']) for t in trials]

    _write(stims[0])

    for i, t in enumerate(trials):
        # NOTE: previously this called _play(stim) with no _write(), so every
        # conditioning trial replayed whatever buffer the last pre-block
        # trial had left on the RM1. The CS+/CS- manipulation never reached
        # the hardware.
        t0 = time.time()
        ff.play(1, [PROC])

        label = 'CS+ ⚡' if t['shocked'] else ('CS+' if t['is_plus'] else 'CS-')
        print(f"  {label:>6s}  {t['cue']} {t['pos']:+5.1f}°")

        if t['shocked']:
            # arm the shock BEFORE the busy-wait so the USB write is not
            # itself part of the CS-US interval
            ff.write('num_shock', 4, PROC)
            precise_sleep_until(t0 + CS_US_INTERVAL / 1000)
            ff.play(2, [PROC])

        ff.wait_to_finish_playing()

        if i + 1 < len(stims):
            _write(stims[i + 1])

        log.append(dict(t, cond_dim=cond_dim, onset=t0, trial_num=i + 1))
        precise_sleep_until(t0 + t['iti_ms'] / 1000)

    return log

# ════════════════════════════════════════════════════════════════════
# ANALYSIS  —  P("right") vs roving value  →  PSE
# ════════════════════════════════════════════════════════════════════
def fit_psychometric(results, condition):
    """Fit P('right') ~ rove_val for one condition. Returns PSE (or None)."""
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
    """PSE = value where P('right') = 0.5. Logistic fit, linear-interp fallback."""
    def logistic(x, alpha, beta):
        return 1.0 / (1.0 + np.exp(-(x - alpha) / beta))
    try:
        from scipy.optimize import curve_fit
        # beta is bounded positive: unbounded, the fit can converge on a
        # NEGATIVE slope (an inverted psychometric function), report success,
        # and hand back a meaningless PSE with no warning.
        (alpha, _beta), _ = curve_fit(
            logistic, x, p, p0=[float(np.mean(x)), 5.0],
            bounds=([x.min() - 20, 1e-3], [x.max() + 20, 100.0]),
            maxfev=10000)
        return float(alpha)
    except Exception:
        # linear interpolation to the 0.5 crossing
        for i in range(len(x) - 1):
            if (p[i] - 0.5) * (p[i + 1] - 0.5) <= 0 and p[i] != p[i + 1]:
                frac = (0.5 - p[i]) / (p[i + 1] - p[i])
                return float(x[i] + frac * (x[i + 1] - x[i]))
        return float(np.interp(0.5, p, x)) if p[0] < p[-1] else None


# ════════════════════════════════════════════════════════════════════
# CUE WEIGHTS  —  PSE  →  relative weight of the CONDITIONED cue
# ════════════════════════════════════════════════════════════════════
# At the PSE the two cues cancel, so the percept is centred:
#
#       w_anchor * anchor_val  +  w_rove * PSE  =  0
#   =>  w_rove / w_anchor  =  -anchor_val / PSE
#
# That ratio is side-invariant (anchor_val and PSE always have opposite
# signs), which raw PSE is NOT — anchor_LEFT gives positive PSEs and
# anchor_RIGHT negative ones, so averaging raw deltas across the two would
# cancel a real effect out.

def _condition_meta(results, condition):
    """Recover (anchor_cue, rove_cue, anchor_val) for a condition."""
    for r in results:
        if r['condition'] == condition:
            return r['anchor_cue'], r['rove_cue'], r['anchor_val']
    return None


def conditioned_cue_weight(results, condition, cond_dim):
    """Normalised weight (0-1) of the CONDITIONED cue in this condition.

    0.5 = both cues weighted equally. >0.5 = the conditioned cue dominates.
    Returns (weight, pse) or (None, pse) if the fit is uninterpretable.
    """
    pse = fit_psychometric(results, condition)
    meta = _condition_meta(results, condition)
    if pse is None or meta is None or abs(pse) < 1e-6:
        return None, pse

    anchor_cue, rove_cue, anchor_val = meta
    ratio = -anchor_val / pse            # w_rove / w_anchor

    # a negative ratio means the PSE fell on the same side as the anchor:
    # the cues did not cancel anywhere in the tested range
    if ratio <= 0:
        return None, pse

    if cond_dim == rove_cue:
        return ratio / (1.0 + ratio), pse
    return 1.0 / (1.0 + ratio), pse


def report_conditioning_effect(pre, post, cond_dim):
    """Print whether conditioning shifted weighting toward the conditioned CUE.

    The predicted signature is a DOUBLE DISSOCIATION, because the conditioned
    cue is the anchor in half the conditions and the rover in the other half:

      cond_dim is the ANCHOR  -> anchor gets stronger -> |PSE| INCREASES
      cond_dim is the ROVER   -> less rove needed     -> |PSE| DECREASES

    Both map onto the same thing once converted to weights: w_conditioned up.
    """
    other_dim = 'ILD' if cond_dim == 'ITD' else 'ITD'

    print(f"\n=== CUE WEIGHTING: shift toward {cond_dim} (CS+) ===")
    print(f"    weight of {cond_dim}, 0.5 = equal weighting with {other_dim}\n")

    deltas = []
    for cond in CONDITIONS:
        w_pre,  pse_pre  = conditioned_cue_weight(pre,  cond, cond_dim)
        w_post, pse_post = conditioned_cue_weight(post, cond, cond_dim)

        role = 'anchor' if cond_dim in cond.split('_')[0] else 'rove  '

        if w_pre is None or w_post is None:
            print(f"  {cond:>18s} [{role}]:  (uninterpretable fit — "
                  f"pse pre={pse_pre}, post={pse_post})")
            continue

        d = w_post - w_pre
        deltas.append(d)
        arrow = '+' if d > 0 else '-'
        print(f"  {cond:>18s} [{role}]:  w_pre={w_pre:.3f}  w_post={w_post:.3f}"
              f"   Δ={d:+.3f} {arrow}"
              f"   (PSE {pse_pre:+.2f}° → {pse_post:+.2f}°)")

    if not deltas:
        print("\n  No interpretable conditions. Cannot assess the shift.")
        return

    mean_d = float(np.mean(deltas))
    n_pos  = sum(d > 0 for d in deltas)

    print(f"\n  mean Δw({cond_dim}) = {mean_d:+.3f}   "
          f"({n_pos}/{len(deltas)} conditions in the predicted direction)")

    if mean_d > 0:
        print(f"  → weighting shifted TOWARD the conditioned cue ({cond_dim}).")
    else:
        print(f"  → weighting shifted AWAY from the conditioned cue "
              f"(toward {other_dim}).")

    print("\n  CAVEAT: with REPS_PER_PROBE = "
          f"{REPS_PER_PROBE}, each PSE rests on {REPS_PER_PROBE} trials per "
          "probe point.\n  Treat a single participant's Δ as descriptive "
          "only — this is not a test.")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== WP2B — affective cue weighting ===")
    PID = input("Participant ID: ").strip()
    DIM = input("Conditioned dimension (ITD/ILD): ").strip().upper()
    if DIM not in ('ITD', 'ILD'):
        raise SystemExit(f"Conditioned dimension must be ITD or ILD, got {DIM!r}")
    # (the old `SIDE = hash(PID) % 2` is gone: Python salts string hashing per
    #  process, so it gave a different answer every run. It was never used.)

    slab.set_default_samplerate(FS)
    ff.initialize(setup='headphones', device=[['RM1', PROC, RCX]],
                  zbus=False, connection='usb')

    pre_trials = generate_conflict_trials('pre')
    post_trials = generate_conflict_trials('post')

    input("Enter → PRE block")
    pre = run_conflict_block(pre_trials, 'pre')

    input("Enter → CONDITIONING")
    conditioning = run_conditioning(DIM)   # was discarded — the whole
                                           # conditioning log was being lost

    input("Enter → POST block")
    post = run_conflict_block(post_trials, 'post')

    # ── results ──
    # raw PSEs first (diagnostic — note the sign flips between LEFT/RIGHT
    # anchors, which is why these must not be averaged directly)
    print("\n=== PSE COMPARISON (pre → post) ===")
    for cond in CONDITIONS:
        pre_pse = fit_psychometric(pre, cond)
        post_pse = fit_psychometric(post, cond)
        if pre_pse is None or post_pse is None:
            print(f"  {cond:>18s}:  (insufficient data)")
        else:
            print(f"  {cond:>18s}:  pre={pre_pse:+6.2f}  post={post_pse:+6.2f}"
                  f"  Δ={post_pse - pre_pse:+6.2f}°")

    # the actual readout
    report_conditioning_effect(pre, post, DIM)

    # ════════════════════════════════════════════════════════════════
    # SAVE — one file per participant, with a metadata header
    # ════════════════════════════════════════════════════════════════
    # Everything needed to interpret the data now lives IN the file. The
    # conditioned dimension in particular was previously recorded nowhere,
    # so the analysis had no way to know which cue was CS+.
    record = {
        'participant_id': PID,
        'cs_plus_dim':    DIM,
        'cs_minus_dim':   'ILD' if DIM == 'ITD' else 'ITD',
        'datetime':       datetime.datetime.now().isoformat(timespec='seconds'),
        'script':         os.path.basename(__file__),
        'config': {
            'FS': FS, 'TONE_MS': TONE_MS, 'RAMP_MS': RAMP_MS, 'F0': F0,
            'BANDWIDTH_OCT': BANDWIDTH_OCT, 'HEAD_RADIUS_CM': HEAD_RADIUS_CM,
            'LEVEL': LEVEL, 'ANCHOR_DEG': ANCHOR_DEG,
            'PROBE_VALS': PROBE_VALS, 'REPS_PER_PROBE': REPS_PER_PROBE,
            'ITI_MIN_MS': ITI_MIN_MS, 'ITI_MAX_MS': ITI_MAX_MS,
            'RESP_TIMEOUT_MS': RESP_TIMEOUT_MS,
            'POST_RESP_MS': POST_RESP_MS,
            'ISI_MS': len(SILENCE) / FS * 1000,
            'N_COND_TRIALS': N_COND_TRIALS, 'SHOCK_RATE': SHOCK_RATE,
            'CS_US_INTERVAL': CS_US_INTERVAL,
            'MAX_SAME_SIDE_RUN': MAX_SAME_SIDE_RUN,
            'MAX_UNREINF_RUN': MAX_UNREINF_RUN,
        },
        'pre':          pre,
        'conditioning': conditioning,
        'post':         post,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"sub-{PID}_WP2B.json")
    with open(out_path, 'w') as f:
        json.dump(record, f, indent=2)

    n_shocked = sum(t['shocked'] for t in conditioning)
    print(f"\nSaved {out_path}")
    print(f"  pre={len(pre)} trials | conditioning={len(conditioning)} "
          f"({n_shocked} shocked) | post={len(post)} trials")
