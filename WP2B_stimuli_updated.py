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

import time
import json
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

ITI_MS         = 1500     # inter-trial interval
RESP_TIMEOUT_MS = 2000    # response window

# Conditioning
N_COND_TRIALS = 50        # total conditioning trials
SHOCK_RATE    = 0.85       # CS+ reinforcement rate
CS_US_INTERVAL = 250      # ms, tone onset → shock

CONDITIONS = ['ITD_anchor_LEFT', 'ITD_anchor_RIGHT',
              'ILD_anchor_LEFT', 'ILD_anchor_RIGHT']
kb = keyboard.Keyboard()

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
    return resp, t_onset

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
        resp, t_onset = _play()

        # ── dead time starts here. Load the next buffer now so that the
        #    next ff.play() is preceded by zero USB transfer.
        if i + 1 < len(sounds):
            _write(sounds[i + 1])

        results.append(dict(t, response=resp, block=block_label,
                            onset=t_onset, trial_num=i + 1))

        # ── pad out to a constant trial period, measured from onset
        precise_sleep_until(t_onset + ITI_MS / 1000)

    return results


# ════════════════════════════════════════════════════════════════════
# CONDITIONING  (one cue dimension → CS+ shocked / CS- safe)
# ════════════════════════════════════════════════════════════════════
def run_conditioning(cond_dim):
    """Condition a cue TYPE, not a side.

    cond_dim : 'ITD' or 'ILD' — the cue type that is CS+ (shocked).
    The other cue type is CS- (safe). Direction (left/right) is
    randomised within BOTH CS+ and CS-, so laterality is not the
    conditioned feature: the participant learns which *cue type*
    predicts shock, not which side.
    """
    other_dim = 'ILD' if cond_dim == 'ITD' else 'ITD'

    trials = []
    for i in range(N_COND_TRIALS):
        is_plus = i < N_COND_TRIALS // 2
        cue     = cond_dim if is_plus else other_dim
        side    = random.choice([-1, +1])          # ← direction randomised
        pos     = side * random.choice(PROBE_VALS[1:])
        shocked = is_plus and random.random() < SHOCK_RATE
        trials.append({'is_plus': is_plus, 'cue': cue,
                       'side': side, 'pos': pos, 'shocked': shocked})
    random.shuffle(trials)

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
        precise_sleep_until(t0 + ITI_MS / 1000 / 2)

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
        (alpha, _beta), _ = curve_fit(
            logistic, x, p, p0=[float(np.mean(x)), 5.0], maxfev=10000)
        return float(alpha)
    except Exception:
        # linear interpolation to the 0.5 crossing
        for i in range(len(x) - 1):
            if (p[i] - 0.5) * (p[i + 1] - 0.5) <= 0 and p[i] != p[i + 1]:
                frac = (0.5 - p[i]) / (p[i + 1] - p[i])
                return float(x[i] + frac * (x[i + 1] - x[i]))
        return float(np.interp(0.5, p, x)) if p[0] < p[-1] else None


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=== WP2B — affective cue weighting ===")
    PID = input("Participant ID: ").strip()
    DIM = input("Conditioned dimension (ITD/ILD): ").strip().upper()
    SIDE = 1 if hash(PID) % 2 == 0 else -1     # which side is CS+

    slab.set_default_samplerate(FS)
    ff.initialize(setup='headphones', device=[['RM1', PROC, RCX]],
                  zbus=False, connection='usb')

    pre_trials = generate_conflict_trials('pre')
    post_trials = generate_conflict_trials('post')

    input("Enter → PRE block")
    pre = run_conflict_block(pre_trials, 'pre')

    input("Enter → CONDITIONING")
    run_conditioning(DIM)

    input("Enter → POST block")
    post = run_conflict_block(post_trials, 'post')

    # ── results ──
    print("\n=== PSE COMPARISON (pre → post) ===")
    for cond in CONDITIONS:
        pre_pse = fit_psychometric(pre, cond)
        post_pse = fit_psychometric(post, cond)
        if pre_pse is None or post_pse is None:
            print(f"  {cond:>18s}:  (insufficient data)")
        else:
            print(f"  {cond:>18s}:  pre={pre_pse:+6.2f}  post={post_pse:+6.2f}"
                  f"  Δ={post_pse - pre_pse:+6.2f}°")

    json.dump(pre, open(f"sub-{PID}_pre.json", 'w'), indent=2)
    json.dump(post, open(f"sub-{PID}_post.json", 'w'), indent=2)
    print(f"\nSaved sub-{PID}_pre.json and sub-{PID}_post.json")
