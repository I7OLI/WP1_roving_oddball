"""
Generate WP1 experiment sequences — VERSION 3 (unified array format).
SIMPLIFIED build: same behaviour and outputs as WP1_generate_seq_v3.py, with dead
code removed and single-use helpers inlined for readability.

SELF-CONTAINED: this file has no dependency on WP1_generate_seq.py. All the
building blocks (roving, balancing, reinforcement, counterbalancing) are
inlined below so v3 can be run and version-controlled on its own.

Difference from v2 (the balance fix)
------------------------------------
v2 balanced three things but only PAIRWISE / IN EXPECTATION:
  * train lengths were ~equal overall,
  * +1/-1 deviant signs were exactly equal,
  * reinforcement was applied AFTER sequence construction (early-biased),
but the CROSS of length x direction x reinforcement was left to chance, so a
given train length could precede more CS+ than CS- deviants, and reinforced vs
clean deviants could have mismatched preceding-train-lengths.

v3 makes that cross EXACT by decomposing construction into two independent steps:
  1. SIGNS. Build the +1/-1 order with the same bounded-walk / rejection method
     as v2 (cumsum stays in +/-MAX_CUMSUM). Only the deviant SIGN touches the
     roving constraint — standards (0s) and shock labels do not — so this step
     alone fully determines the roving.
  2. PAINT. Onto each direction's slots, assign a fully balanced pool of
     (train_length x reinforcement) tags, drawn without replacement:
         CS+ : n_reinf cascade + n_reinf-complement clean, 25 of each length in each
         CS- : n_reinf tap     + clean,                    25 of each length in each
     Train length and reinforcement never affect the +/-MAX_CUMSUM constraint,
     so they can be assigned freely — no reshuffle needed for them.

Result: in single-modality conditioning blocks, the 2 (dir) x 2 (reinf) x 6
(length) = 24 cells are exactly equal. Signs stay exactly balanced and cumsum
stays in +/-MAX_CUMSUM, as in v2. In bf/bp the mover (control) stream is nudged
by enforce_min_separation, so a few of ITS train lengths drift by a trial or two
(signs/counts stay exact); the CS+ anchor stream is untouched and stays exact.

Reinforcement timing is controlled by REINF_TIME_BIAS below ('early' keeps v2's
front-loaded acquisition bias; 'uniform' spreads shocks evenly in time).

Difference from v1:
    v1 wrote a separate 1-D sequence (-1/0/+1) per experiment type (f / p / a).
    v2 encodes everything in ONE (n_trials, 5) integer array:

        column 0 : frequency step    (-1 down / 0 stay / +1 up)   -- roving
        column 1 : azimuth  step     (-1 down / 0 stay / +1 up)   -- roving
        column 2 : shock             (0 none / 1 tap CS- / 5 cascade CS+)
        column 3 : pattern identity  (-1 down / 0 standard / +1 up)  -- NOT roving
        column 4 : base-freq index   (index into frequency_list, or -1 if unused)

    Columns 0-2 are unchanged from the original 3-column v2 format, so any run
    script that indexes rows by COL_FREQ / COL_AZI / COL_SHOCK keeps working.

    Columns 0, 1 and 3 are mutually exclusive within a row: a trial only ever
    carries a deviant in ONE stimulus column at a time.

    IMPORTANT semantic difference for column 3: freq and azimuth steps ROVE
    (the run script accumulates them into an index and clamps). Pattern steps
    do NOT rove — the value IS the pattern identity for that trial, resolved
    against PATTERN_STRUCTURES. Column 4 gives the base frequency that the
    pattern is built on, pre-drawn here so the .npy fully determines the
    session (no run-time randomness).

Session types (--type)
----------------------
    f   frequency only            -> deviants in col 0; freq is CS+ (shocked)
    p   azimuth only              -> deviants in col 1; azimuth is CS+ (shocked)
    a   abstract pattern only     -> deviants in col 3 (+ col 4 base freq);
                                     pattern is CS+ (shocked)
    bf  both freq + azimuth       -> deviants in BOTH cols 0 and 1, but only
                                     FREQ is CS+ (only freq deviants are ever
                                     shocked; azimuth deviants are an
                                     unreinforced control)
    bp  both freq + azimuth       -> both cols, only AZIMUTH is CS+ (shocked)

    Every type is a 2-block session: BASELINE (no shock) + CONDITIONING (shock).
    Think of each type as its own session / day.

    In the "both" modes the two deviant streams are interleaved while preserving
    each stream's internal order, so the +/-4 cumsum balance (and therefore the
    roving) is preserved for each dimension independently. Type 'a' is
    standalone — it is never interleaved with f or p.

Usage
-----
Batch (N participants) — also writes the allocation sheet:
    python WP1_generate_seq_v2.py 24
    python WP1_generate_seq_v2.py 24 --type a --seed 42

Single:
    python WP1_generate_seq_v2.py --participant 1 --type bp

Allocation sheet only (who does which session on which day, and their CS+):
    python WP1_generate_seq_v2.py --allocate 24

Everything is written into STUDY_DIR (set just below this docstring):
    <STUDY_DIR>/allocation.csv
    <STUDY_DIR>/sequences/WP1_sub001_bf_v2_seq.npy    the (n_trials, 5) int array
    <STUDY_DIR>/sequences/WP1_sub001_bf_v2_seq.json   metadata (blocks, timing, CS+)
    <STUDY_DIR>/sequences/WP1_sub001_bf_v2_seq.csv    long form (inspection only)
"""
import argparse
import json
import os
import random
import csv
from datetime import datetime

import numpy as np


# ============================================================================
#  >>>  STUDY FOLDER — SET THIS  <<<
# ============================================================================
# Everything for one study lives in here. Point it at a piloting folder now,
# a real-experiment folder later. Created automatically if it doesn't exist.
#
#     <STUDY_DIR>/allocation.csv    who gets which order + which CS+
#     <STUDY_DIR>/sequences/        the .npy / .json / .csv this script writes
#     <STUDY_DIR>/data/             behavioural CSVs from the run script
#     <STUDY_DIR>/pupil/            pupil recordings from the run script
#
# NOTE: the same line exists at the top of WP1_run_Exp_v2.py. Change both when
# you switch studies. Each script prints the folder it is using when it starts,
# so a mismatch is visible immediately.

STUDY_DIR = 'studies/pilot'

# ----------------------------------------------------------------------------
SEQ_DIR = os.path.join(STUDY_DIR, 'sequences')
ALLOCATION_FILE = os.path.join(STUDY_DIR, 'allocation.csv')


# ============================================================================
# EXPERIMENT CONFIG (shared across all participants)
# ============================================================================
ITI = 0.2
TONE_DURATION = 0.1
ITI_WITHIN_PATTERN = 0.1
MAX_CUMSUM = 4
N_DEVIANTS_BASELINE = 400
N_DEVIANTS_CONDITIONING = 600
N_SHOCK = 150
TRAIN_LENGTHS = (5, 6, 7, 8, 9, 10)   # standards-run lengths (single source of truth)
REINF_PROB_START = 0.7
REINF_PROB_END = 0.3
# v3: how reinforced (shocked) deviants are distributed over time within a block.
#   'early'   -> front-loaded like v2 (linear decay REINF_PROB_START->REINF_PROB_END),
#                drives acquisition but shocks are not temporally uniform.
#   'uniform' -> reinforced slots spread evenly over the block (cleanest factorial).
# Either way the length x direction x reinforcement CROSS is exactly balanced;
# this flag only sets WHERE in the timeline the reinforced deviants sit.
REINF_TIME_BIAS = 'early'
SHOCK_ONSET_IN_ITI = 0.1
SOA = ITI + TONE_DURATION

# ---------------------------------------------------------------------------
# CONSTANT SESSION LENGTH
# ---------------------------------------------------------------------------
# Every block is built as [ ...standards, deviant ] trains, so its LAST trial is
# always a deviant with no post-stimulus standards -> that trial can't be scored.
# We therefore append a run of standards ("tail") after each block's last deviant.
#
# The block CORE length also isn't constant: N_DEVIANTS_BASELINE (400) doesn't
# divide evenly by len(TRAIN_LENGTHS) (6), so a couple of leftover trains get
# random lengths; and in bf/bp the min-separation nudge adds a trial or two.
# To pin every session to the SAME total, we pad each block to a fixed target by
# making its tail VARIABLE: tail = target - core (>= TAIL_STANDARDS). The tail is
# pure filler standards, so absorbing the wobble there costs nothing analytically.
TAIL_STANDARDS = 5     # minimum standards after a block's last deviant
SEP_MARGIN = 10        # headroom for the bf/bp concurrent-stream nudge


def _max_core(n_deviants):
    """Worst-case core length of a block with `n_deviants` deviants: the balanced
    'base per length' trains in both directions, plus the longest possible
    leftover trains, plus the deviants themselves."""
    per_dir = n_deviants // 2
    base, rem = divmod(per_dir, len(TRAIN_LENGTHS))
    return n_deviants + 2 * base * sum(TRAIN_LENGTHS) + 2 * rem * max(TRAIN_LENGTHS)


# Fixed per-block targets -> constant total tones for EVERY session type.
TARGET_BASELINE_TONES = _max_core(N_DEVIANTS_BASELINE) + SEP_MARGIN + TAIL_STANDARDS
TARGET_CONDITIONING_TONES = _max_core(N_DEVIANTS_CONDITIONING) + SEP_MARGIN + TAIL_STANDARDS
TARGET_TOTAL_TONES = TARGET_BASELINE_TONES + TARGET_CONDITIONING_TONES

# Abstract-pattern condition: each trial is a 2-tone pattern built on a base
# frequency. The deviant is a step in the SECOND tone.
PATTERN_STEP = 0.15
PATTERN_STRUCTURES = {
    0: [0, 0],      # standard:    stay-stay
    1: [0, 1],      # deviant up:  stay-up
    -1: [0, -1],    # deviant down: stay-down
}

# Each pattern tone is HALF a normal tone, so the two tones together deliver the
# same total acoustic energy as one f/p tone (inherited from WP1_run_Exp_ff.py
# and friends). Change this one constant if you want full-length pattern tones.
PATTERN_TONE_DURATION = TONE_DURATION / 2

# Total on-air span of one pattern trial: tone + gap + tone.
PATTERN_SPAN = (len(PATTERN_STRUCTURES[0]) * PATTERN_TONE_DURATION
                + (len(PATTERN_STRUCTURES[0]) - 1) * ITI_WITHIN_PATTERN)   # 0.2 s

# A pattern trial is longer than a single tone, so it needs its own SOA.
A_SOA = 0.65

# Shock timing is defined RELATIVE TO STIMULUS OFFSET so the rule is identical
# for single tones and patterns:
#     shock_time = trial_onset + stimulus_span + SHOCK_DELAY_AFTER_OFFSET
# For f/p that reproduces the existing behaviour exactly (0.1 s tone, shock at
# 0.1 s). For 'a' it puts the shock at 0.2 s, i.e. at pattern offset.
SHOCK_DELAY_AFTER_OFFSET = SHOCK_ONSET_IN_ITI - TONE_DURATION   # 0.0 s


# In the "both" session types the two stimulus dimensions run as INDEPENDENT
# concurrent streams over one shared trial timeline — each keeps its own train
# structure and its own full deviant budget. A frequency change may therefore be
# followed a trial or two later by an azimuth change.
#
# MIN_MODALITY_SEPARATION is the minimum number of trials between a deviant in
# one dimension and a deviant in the other. 2 means: never simultaneous, never
# adjacent. Set to 1 to allow adjacent-but-not-simultaneous, or 0 to allow
# genuine compound deviants.
MIN_MODALITY_SEPARATION = 2


# ============================================================================
# STIMULUS LISTS
# ============================================================================
def macke_feqlist(base=900, max_cumsum=4, step=0.1):
    """Log-spaced frequency ladder centred on `base`, +/- max_cumsum steps."""
    step = 1 + step
    feqlist = [base / (step ** i) for i in range(1, max_cumsum + 1)]
    feqlist.append(base)
    feqlist += [base * (step ** i) for i in range(1, max_cumsum + 1)]
    feqlist.sort()
    return feqlist


def macke_azilist(max_cumsum=4, step=10):
    """Linear azimuth ladder centred on 0 degrees, +/- max_cumsum steps."""
    azilist = [-step * i for i in range(1, max_cumsum + 1)]
    azilist.append(0)
    azilist += [step * i for i in range(1, max_cumsum + 1)]
    azilist.sort()
    return azilist


# ============================================================================
# v3 SEQUENCE GENERATION  (exact length x direction x reinforcement balance)
# ============================================================================
def bounded_signs(n_deviants, max_cumsum=4, block_size=50, max_attempts=1000):
    """
    The one fragile invariant, kept as a named function on purpose.

    Return a list of +1/-1 deviant SIGNS (no standards), exactly balanced (equal
    +1 and -1), whose running cumsum never leaves +/-max_cumsum -- so the roving
    index never walks off the stimulus ladder. Built in balanced sub-blocks of
    `block_size` (each block's cumsum returns to 0 at its boundary, so blocks
    compose without drift); each block is found by rejection sampling: reshuffle
    until the walk stays in range. Train length and reinforcement are painted on
    later and never touch this constraint -- which is what keeps the rest simple.
    """
    if n_deviants % 2:
        n_deviants += 1
    n_blocks = int(np.ceil(n_deviants / block_size))
    per_block = [n_deviants // n_blocks] * n_blocks
    for i in range(n_deviants % n_blocks):
        per_block[i] += 1
    per_block = [nb + (nb % 2) for nb in per_block]   # each sub-block even

    signs = []
    for nb in per_block:
        block = [1] * (nb // 2) + [-1] * (nb // 2)
        for _ in range(max_attempts):
            random.shuffle(block)
            if np.abs(np.cumsum(block)).max() <= max_cumsum:
                break
        else:
            raise RuntimeError(
                f"No valid sign block of {nb} within +/-{max_cumsum} after "
                f"{max_attempts} attempts.")
        signs.extend(block)
    return signs

def _balanced_lengths(n_slots, train_lengths):
    """A shuffled list of `n_slots` train lengths, as equal across lengths as
    possible (remainder scattered at random)."""
    base = n_slots // len(train_lengths)
    rem = n_slots % len(train_lengths)
    pool = []
    for L in train_lengths:
        pool.extend([L] * base)
    if rem:
        pool.extend(random.sample(list(train_lengths), rem))
    random.shuffle(pool)
    return pool


def assign_tags(n_slots, train_lengths, n_reinf, reinf_timing):
    """
    For one DIRECTION group (all the CS+ slots, or all the CS- slots), decide
    each slot's train length and whether it is reinforced, with the cross of
    (reinforcement x length) exactly balanced:

        * `n_reinf` slots are reinforced, the rest clean;
        * within the reinforced subset, lengths are balanced across train_lengths;
        * within the clean subset, lengths are balanced across train_lengths.

    Returns (lengths_per_slot, reinf_per_slot) — two lists of length n_slots,
    indexed by the slot's temporal rank within its direction group.
    """
    n_reinf = max(0, min(n_reinf, n_slots))

    if n_reinf == 0:
        reinf_ranks = set()
    elif reinf_timing == 'uniform':
        reinf_ranks = set(random.sample(range(n_slots), n_reinf))
    else:  # 'early' — front-load reinforced slots (linear-decay weights)
        w = np.array([REINF_PROB_START - (REINF_PROB_START - REINF_PROB_END)
                      * (r / max(n_slots - 1, 1)) for r in range(n_slots)])
        reinf_ranks = set(np.random.choice(n_slots, n_reinf, replace=False,
                                           p=w / w.sum()))

    reinf_flags = [r in reinf_ranks for r in range(n_slots)]
    reinf_idx = [r for r in range(n_slots) if reinf_flags[r]]
    clean_idx = [r for r in range(n_slots) if not reinf_flags[r]]

    lengths_per = [0] * n_slots
    for idx_list in (reinf_idx, clean_idx):
        for slot, L in zip(idx_list, _balanced_lengths(len(idx_list), train_lengths)):
            lengths_per[slot] = L
    return lengths_per, reinf_flags


def make_stream(n_deviants, cs_plus_value, n_reinf,
                train_lengths=TRAIN_LENGTHS, max_cumsum=4,
                reinf_timing=REINF_TIME_BIAS,
                cs_plus_shocks=5, cs_minus_shocks=1):
    """
    Build ONE roving stream as (seq_1d, shock_1d), with the length x direction x
    reinforcement cross exactly balanced.

    Step 1 (signs): bounded_signs -> the +1/-1 order, cumsum in +/-max_cumsum.
    Step 2 (paint): for each direction group assign balanced (length, reinf) tags.
        CS+ direction (sign == cs_plus_value): reinforced -> cascade (cs_plus_shocks).
        CS- direction (sign == -cs_plus_value): reinforced -> tap (cs_minus_shocks).
        Both directions get n_reinf reinforced slots (balanced tap control).

    n_reinf = 0 gives a baseline stream (all clean, length x direction balanced).
    """
    signs = bounded_signs(n_deviants, max_cumsum=max_cumsum)
    n = len(signs)

    plus_slots = [i for i, s in enumerate(signs) if s == cs_plus_value]    # CS+
    minus_slots = [i for i, s in enumerate(signs) if s == -cs_plus_value]  # CS-

    lp_plus, rf_plus = assign_tags(len(plus_slots), train_lengths, n_reinf, reinf_timing)
    lp_minus, rf_minus = assign_tags(len(minus_slots), train_lengths, n_reinf, reinf_timing)

    length_of = {}
    shock_of = {}
    for k, slot in enumerate(plus_slots):
        length_of[slot] = lp_plus[k]
        shock_of[slot] = cs_plus_shocks if rf_plus[k] else 0
    for k, slot in enumerate(minus_slots):
        length_of[slot] = lp_minus[k]
        shock_of[slot] = cs_minus_shocks if rf_minus[k] else 0

    seq, shock = [], []
    for slot in range(n):
        seq.extend([0] * length_of[slot])
        shock.extend([0] * length_of[slot])
        seq.append(signs[slot])
        shock.append(shock_of[slot])

    return np.asarray(seq, dtype=int), np.asarray(shock, dtype=int)


# ============================================================================
# COUNTERBALANCING
# ============================================================================
# Which hierarchical level runs on day 1 / 2 / 3. Cycles every 6 participants.
SESSION_ORDERS = ['fpa', 'fap', 'pfa', 'paf', 'afp', 'apf']

CS_PLUS_LABELS = {'f': {1: 'high', -1: 'low'},
                  'p': {1: 'right', -1: 'left'},
                  'a': {1: 'up', -1: 'down'}}


def get_session_order(participant_id):
    """
    Session order (e.g. 'fpa' = frequency day 1, position day 2, abstract day 3).
    Cycles every 6 participants, so all 6 orders are equally often used at any
    multiple of 6.
    """
    return SESSION_ORDERS[(participant_id - 1) % 6]


def get_cs_plus_assignment(participant_id):
    """
    Returns the CS+ direction (+1 or -1) for all 3 stimulus dimensions
    to ensure perfect counterbalancing every 8 participants.

    Mappings:
    f: +1 (High) / -1 (Low)
    p: +1 (Right) / -1 (Left)
    a: +1 (Up)    / -1 (Down)
    """
    combos = [
        {'f': 1, 'p': 1, 'a': 1},     # 1: High, Right, Up
        {'f': 1, 'p': 1, 'a': -1},    # 2: High, Right, Down
        {'f': 1, 'p': -1, 'a': 1},    # 3: High, Left,  Up
        {'f': 1, 'p': -1, 'a': -1},   # 4: High, Left,  Down
        {'f': -1, 'p': 1, 'a': 1},    # 5: Low,  Right, Up
        {'f': -1, 'p': 1, 'a': -1},   # 6: Low,  Right, Down
        {'f': -1, 'p': -1, 'a': 1},   # 7: Low,  Left,  Up
        {'f': -1, 'p': -1, 'a': -1},  # 8: Low,  Left,  Down
    ]
    return combos[(participant_id - 1) % 8]


def write_allocation(n_participants):
    """
    Write <STUDY_DIR>/allocation.csv — the schedule sheet: who does which
    session on which day, and which direction is CS+ for them.

    Order cycles every 6 and CS+ every 8, so both are exactly balanced at any
    multiple of 24. Their COMBINATION repeats every 24 (24 of the 48 possible
    order x CS+ cells are used), which is fine unless you expect an
    order x CS+ interaction.

    Deterministic: participant 7 always gets the same allocation, so rerunning
    this with a bigger n never changes anyone already tested.
    """
    os.makedirs(STUDY_DIR, exist_ok=True)
    stamp = datetime.now().isoformat(timespec='seconds')

    with open(ALLOCATION_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['participant_id', 'order_code',
                    'session_1_type', 'session_2_type', 'session_3_type',
                    'cs_plus_f', 'cs_plus_p', 'cs_plus_a',
                    'cs_plus_f_label', 'cs_plus_p_label', 'cs_plus_a_label',
                    'generated_at'])
        for pid in range(1, n_participants + 1):
            order = get_session_order(pid)
            cs = get_cs_plus_assignment(pid)
            w.writerow([pid, order, order[0], order[1], order[2],
                        cs['f'], cs['p'], cs['a'],
                        CS_PLUS_LABELS['f'][cs['f']],
                        CS_PLUS_LABELS['p'][cs['p']],
                        CS_PLUS_LABELS['a'][cs['a']],
                        stamp])

    print(f"\n{'=' * 70}")
    print(f"ALLOCATION  (n = {n_participants})")
    print(f"{'=' * 70}")
    print(f"{'sub':>4s} | {'order':>5s} | {'CS+ f':>6s} {'CS+ p':>6s} {'CS+ a':>6s}")
    for pid in range(1, min(n_participants, 24) + 1):
        order = get_session_order(pid)
        cs = get_cs_plus_assignment(pid)
        print(f"{pid:>4d} | {order:>5s} | "
              f"{CS_PLUS_LABELS['f'][cs['f']]:>6s} "
              f"{CS_PLUS_LABELS['p'][cs['p']]:>6s} "
              f"{CS_PLUS_LABELS['a'][cs['a']]:>6s}")
    if n_participants > 24:
        print(f"  ... {n_participants - 24} more")
    if n_participants % 24:
        print(f"\n  NOTE: order and CS+ are only exactly balanced at multiples "
              f"of 24 (24, 48, ...).")
    print(f"\nSaved: {ALLOCATION_FILE}")
    print(f"{'=' * 70}")


# ============================================================================
# JSON HELPERS
# ============================================================================
def sanitize_for_json(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


# ============================================================================
# COLUMN LAYOUT  (single source of truth for the array shape)
# ============================================================================
COL_FREQ = 0        # frequency step     (-1/0/+1)  roving
COL_AZI = 1         # azimuth  step      (-1/0/+1)  roving
COL_SHOCK = 2       # shock amount       (0/1/5)
COL_PATTERN = 3     # pattern identity   (-1/0/+1)  NOT roving
COL_BASEFREQ = 4    # index into frequency_list for the pattern base (-1 = unused)
N_COLS = 5

NO_BASEFREQ = -1    # sentinel for column 4 when the type isn't 'a'

# Which stimulus column each modality writes its deviant steps into.
MODALITY_COL = {'f': COL_FREQ, 'p': COL_AZI, 'a': COL_PATTERN}

# Base stimulus lists. The centre index is the "resting" value.
FREQUENCY_LIST = macke_feqlist(700, MAX_CUMSUM, 0.1)
POSITION_LIST = macke_azilist(MAX_CUMSUM, 10)

# Session-type table: which modalities are present, and which one is CS+.
TYPE_CONFIG = {
    'f':  {'modalities': ['f'],       'cs_plus_modality': 'f'},
    'p':  {'modalities': ['p'],       'cs_plus_modality': 'p'},
    'a':  {'modalities': ['a'],       'cs_plus_modality': 'a'},
    'bf': {'modalities': ['f', 'p'],  'cs_plus_modality': 'f'},
    'bp': {'modalities': ['f', 'p'],  'cs_plus_modality': 'p'},
}
TYPES = list(TYPE_CONFIG.keys())

TYPE_HELP = ("f = freq,  p = azimuth,  a = abstract pattern,  "
             "bf = both (freq CS+),  bp = both (azimuth CS+)")

# Every session is 2 blocks: baseline (no shock) + conditioning (shock).
BLOCK_PLAN_V2 = [
    {'label': 'BASELINE',     'n_deviants': N_DEVIANTS_BASELINE,     'use_reinforcement': False},
    {'label': 'CONDITIONING', 'n_deviants': N_DEVIANTS_CONDITIONING, 'use_reinforcement': True},
]


# ============================================================================
# BLOCK CONSTRUCTION
# ============================================================================
def enforce_min_separation(anchor, mover, min_sep=MIN_MODALITY_SEPARATION, pad=500):
    """
    Move `mover`'s deviants so no deviant sits within `min_sep - 1` trials of
    any `anchor` deviant. (min_sep=2 -> never on the same trial, never adjacent.)

    The anchor stream is untouched: it is the CS+ modality, whose timing drives
    conditioning, so it must not be perturbed.

    What is preserved EXACTLY:
      * the number of deviants in `mover`
      * their VALUES and their ORDER  -> the +/-MAX_CUMSUM balance is unaffected
    What changes:
      * deviant POSITIONS shift by a trial or two, so a few of `mover`'s train
        lengths fall outside the nominal train_lengths range.

    Returns (anchor, mover, n_moved, shifts).
    """
    n = max(len(anchor), len(mover)) + pad
    a_full = np.zeros(n, dtype=int); a_full[:len(anchor)] = anchor
    m_full = np.zeros(n, dtype=int); m_full[:len(mover)] = mover

    forbidden = np.zeros(n, dtype=bool)
    a_pos = np.flatnonzero(a_full)
    for off in range(-(min_sep - 1), min_sep):
        idx = a_pos + off
        forbidden[idx[(idx >= 0) & (idx < n)]] = True

    old_pos = list(np.flatnonzero(m_full))
    values = [int(m_full[i]) for i in old_pos]

    new_pos, taken, prev, n_moved = [], set(), -1, 0
    for p in old_pos:
        chosen = None
        for d in range(n):                     # search outward from p
            for cand in ([p] if d == 0 else [p - d, p + d]):
                if cand <= prev or cand < 0 or cand >= n:
                    continue
                if forbidden[cand] or cand in taken:
                    continue
                chosen = cand
                break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(
                f"Could not place deviant {len(new_pos) + 1}/{len(old_pos)} with "
                f"min_sep={min_sep}. The timeline is too crowded — reduce "
                f"n_deviants or min_sep.")
        if chosen != p:
            n_moved += 1
        new_pos.append(chosen)
        taken.add(chosen)
        prev = chosen

    out = np.zeros(n, dtype=int)
    for pos, val in zip(new_pos, values):
        out[pos] = val

    used = max(np.flatnonzero(a_full).max(), np.flatnonzero(out).max()) + 1
    shifts = [b - a for a, b in zip(old_pos, new_pos)]
    return a_full[:used], out[:used], n_moved, shifts


def build_block(n_deviants, modalities, cs_plus_modality, cs_plus_value,
                use_reinforcement, soa=SOA):
    """
    Build one block's (n, N_COLS) array.

    Single modality  -> one deviant stream in that modality's column.
    Two modalities   -> two roving streams (n_deviants split evenly),
                        interleaved with order preserved; each deviant lands in
                        its own column. Only the CS+ modality is ever shocked.

    For modality 'a' the sequence is still generated by make_stream
    (so up/down deviants stay balanced and evenly spaced), but the values are
    written as pattern IDENTITIES rather than roving steps, and every row gets
    a pre-drawn base-frequency index in COL_BASEFREQ.
    """
    # v3: reinforcement is baked into stream construction (crossed with length),
    # not applied afterwards. n_reinf per direction = N_SHOCK in a shocked block.
    n_reinf = N_SHOCK if use_reinforcement else 0

    if len(modalities) == 1:
        m = modalities[0]
        seq, shock = make_stream(n_deviants, cs_plus_value, n_reinf)
        block = np.zeros((len(seq), N_COLS), dtype=int)
        block[:, COL_BASEFREQ] = NO_BASEFREQ
        block[:, MODALITY_COL[m]] = seq
        block[:, COL_SHOCK] = shock
    else:
        # TWO INDEPENDENT CONCURRENT STREAMS.
        #
        # Each modality gets the FULL deviant budget and its own train
        # structure, then both are overlaid on one shared trial timeline. This
        # is deliberately NOT an interleave of whole trains: the dimensions are
        # independent, so a frequency change can be followed a trial or two
        # later by an azimuth change.
        #
        # Because the streams are concurrent rather than concatenated, the
        # block is the length of ONE stream, not the sum — deviant DENSITY
        # doubles, session duration does not.
        other = [m for m in modalities if m != cs_plus_modality]
        if len(modalities) != 2 or not other:
            raise ValueError(f"expected exactly 2 modalities incl. the CS+, got {modalities}")
        mover_mod = other[0]

        # CS+ anchor carries the (balanced) reinforcement; the mover is a
        # never-shocked control, so it is built with n_reinf=0.
        anchor_seq, anchor_shock = make_stream(n_deviants, cs_plus_value, n_reinf)
        mover_seq, _ = make_stream(n_deviants, cs_plus_value=1, n_reinf=0)

        # The CS+ modality anchors the timeline; the other one yields, so
        # conditioning timing is never perturbed.
        anchor, mover, n_moved, shifts = enforce_min_separation(anchor_seq, mover_seq)

        block = np.zeros((len(anchor), N_COLS), dtype=int)
        block[:, COL_BASEFREQ] = NO_BASEFREQ
        block[:, MODALITY_COL[cs_plus_modality]] = anchor
        block[:, MODALITY_COL[mover_mod]] = mover

        # Reapply the anchor's shock labels. enforce_min_separation leaves the
        # anchor deviants in place and in order, so map shock across by rank.
        anchor_dev_shocks = anchor_shock[np.flatnonzero(anchor_seq)]
        for pos, s in zip(np.flatnonzero(anchor), anchor_dev_shocks):
            block[pos, COL_SHOCK] = int(s)

        max_shift = max((abs(s) for s in shifts), default=0)
        print(f"  Concurrent streams: {cs_plus_modality} (anchor, CS+) + "
              f"{mover_mod} (moved to keep >= {MIN_MODALITY_SEPARATION} trials apart)")
        print(f"    {n_moved}/{len(shifts)} {mover_mod}-deviants shifted "
              f"(mean |shift| {np.mean(np.abs(shifts)) if shifts else 0:.2f}, max {max_shift})")

    # Abstract-pattern trials need a base frequency drawn for EVERY row
    # (standards included), pre-drawn here so the .npy is fully deterministic.
    if 'a' in modalities:
        block[:, COL_BASEFREQ] = np.random.randint(0, len(FREQUENCY_LIST),
                                                   size=block.shape[0])

    return block


def make_tail(n_rows, modalities):
    """A run of `n_rows` STANDARD trials (no deviant, no shock) appended after a
    block's last deviant, so that final deviant has a post-stimulus window and is
    scorable. For type 'a' each standard still needs its own base frequency."""
    tail = np.zeros((max(n_rows, 0), N_COLS), dtype=int)
    tail[:, COL_BASEFREQ] = NO_BASEFREQ
    if 'a' in modalities:
        tail[:, COL_BASEFREQ] = np.random.randint(0, len(FREQUENCY_LIST),
                                                  size=tail.shape[0])
    return tail


# ============================================================================
# BUILD ONE PARTICIPANT
# ============================================================================
def generate_one(participant_id, exp_type, seed, out_dir=SEQ_DIR):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    tcfg = TYPE_CONFIG[exp_type]
    modalities = tcfg['modalities']
    cs_plus_modality = tcfg['cs_plus_modality']

    # CS+ direction per modality (counterbalanced every 8 participants).
    assignment = get_cs_plus_assignment(participant_id)  # {'f':±1,'p':±1,'a':±1}
    cs_plus_value = assignment[cs_plus_modality]
    order_code = get_session_order(participant_id)

    # Timing depends on the session type: pattern trials are longer on air and
    # therefore need a longer SOA.
    soa = A_SOA if exp_type == 'a' else SOA
    stim_span = PATTERN_SPAN if exp_type == 'a' else TONE_DURATION

    block_rows = []          # list of (n_i, N_COLS) arrays, concatenated at the end
    block_table = []         # metadata describing each block's row span
    cursor = 0               # running row index into the final array

    for cfg in BLOCK_PLAN_V2:
        label = f"{exp_type.upper()}_{cfg['label']}"
        print(f"\n--- {label}  (modalities={modalities}, "
              f"CS+={cs_plus_modality}{cs_plus_value:+d}) ---")

        block = build_block(
            n_deviants=cfg['n_deviants'],
            modalities=modalities,
            cs_plus_modality=cs_plus_modality,
            cs_plus_value=cs_plus_value,
            use_reinforcement=cfg['use_reinforcement'],
            soa=soa,
        )
        core_n = block.shape[0]

        # Pad to a fixed per-block target with a standards tail, so the total
        # tone count is identical for every participant and every session type.
        target = (TARGET_CONDITIONING_TONES if cfg['use_reinforcement']
                  else TARGET_BASELINE_TONES)
        tail_n = target - core_n
        if tail_n < TAIL_STANDARDS:
            raise RuntimeError(
                f"{label}: core {core_n} exceeds target {target} - "
                f"TAIL_STANDARDS ({TAIL_STANDARDS}). Increase SEP_MARGIN.")
        block = np.vstack([block, make_tail(tail_n, modalities)])
        n = block.shape[0]                    # == target
        block_rows.append(block)
        print(f"  Core {core_n} + tail {tail_n} standards -> {n} tones (target {target})")

        block_table.append({
            'label': label,
            'modalities': modalities,
            'cs_plus_modality': cs_plus_modality,
            'cs_plus_column': MODALITY_COL[cs_plus_modality],
            'cs_plus_value': int(cs_plus_value),
            'cs_minus_value': int(-cs_plus_value),
            'use_reinforcement': cfg['use_reinforcement'],
            'row_start': cursor,              # inclusive
            'row_end': cursor + n,            # exclusive
            'n_trials': int(n),
            'n_core_trials': int(core_n),     # trials up to and incl. last deviant
            'n_tail_standards': int(tail_n),  # trailing filler standards
            'n_freq_deviants': int(np.count_nonzero(block[:, COL_FREQ])),
            'n_azi_deviants': int(np.count_nonzero(block[:, COL_AZI])),
            'n_pattern_deviants': int(np.count_nonzero(block[:, COL_PATTERN])),
            'n_shock_trials': int(np.count_nonzero(block[:, COL_SHOCK])),
        })
        cursor += n

    seq_array = np.vstack(block_rows).astype(int)  # (total_trials, N_COLS)

    tag = f"sub{participant_id:03d}_{exp_type}_v3"

    # --- Save the numpy array (the primary v2 artefact) ---------------------
    npy_path = os.path.join(out_dir, f"WP1_{tag}_seq.npy")
    np.save(npy_path, seq_array)

    # --- Save metadata ------------------------------------------------------
    meta = sanitize_for_json({
        'version': 3,
        'reinf_time_bias': REINF_TIME_BIAS,
        'tail_standards': TAIL_STANDARDS,
        'target_baseline_tones': TARGET_BASELINE_TONES,
        'target_conditioning_tones': TARGET_CONDITIONING_TONES,
        'target_total_tones': TARGET_TOTAL_TONES,
        'participant_id': participant_id,
        'experiment_type': exp_type,
        'modalities': modalities,
        'cs_plus_modality': cs_plus_modality,
        'cs_plus_value': int(cs_plus_value),
        'random_seed': seed,
        'generated_at': datetime.now().isoformat(),
        # array description
        'array_shape': list(seq_array.shape),
        'columns': {
            'freq_step': COL_FREQ,
            'azi_step': COL_AZI,
            'shock': COL_SHOCK,
            'pattern': COL_PATTERN,
            'base_freq_index': COL_BASEFREQ,
        },
        'column_semantics': {
            'freq_step': 'roving step, accumulate into frequency_list index',
            'azi_step': 'roving step, accumulate into position_list index',
            'shock': 'absolute shock pulse count for this trial',
            'pattern': 'absolute pattern identity, look up in pattern_structures',
            'base_freq_index': ('direct index into frequency_list for the pattern '
                                f'base tone; {NO_BASEFREQ} = unused'),
        },
        'shock_codes': {'none': 0, 'cs_minus_tap': 1, 'cs_plus_cascade': 5},
        'no_basefreq_sentinel': NO_BASEFREQ,
        # stimulus resolution tables (roving walks these with index clamping)
        'frequency_list': FREQUENCY_LIST,
        'position_list': POSITION_LIST,
        'freq_center_index': len(FREQUENCY_LIST) // 2,
        'azi_center_index': len(POSITION_LIST) // 2,
        # abstract-pattern resolution
        'pattern_structures': PATTERN_STRUCTURES,
        'pattern_step': PATTERN_STEP,
        'pattern_tone_duration': PATTERN_TONE_DURATION,
        'pattern_span': PATTERN_SPAN,
        # timing — SOA and stimulus span are TYPE-SPECIFIC
        'ITI': ITI,
        'tone_duration': TONE_DURATION,
        'iti_within_pattern': ITI_WITHIN_PATTERN,
        'SOA': soa,                       # A_SOA for type 'a', ITI+tone otherwise
        'standard_SOA': SOA,
        'A_SOA': A_SOA,
        'stim_span': stim_span,           # on-air duration of one trial's stimulus
        'max_cumsum': MAX_CUMSUM,
        # In 'both' sessions the dimensions run as independent concurrent
        # streams, each with the full deviant budget, kept this many trials apart.
        'concurrent_streams': len(modalities) > 1,
        'min_modality_separation': MIN_MODALITY_SEPARATION if len(modalities) > 1 else None,
        'shock_onset_in_iti': SHOCK_ONSET_IN_ITI,
        # Shock fires at:  onset + stim_span + shock_delay_after_offset.
        # One rule for single tones and patterns alike.
        'shock_delay_after_offset': SHOCK_DELAY_AFTER_OFFSET,
        'shock_time_from_onset': stim_span + SHOCK_DELAY_AFTER_OFFSET,
        # per-modality CS+ assignment + this participant's session order
        'cs_plus_assignment': {k: int(v) for k, v in assignment.items()},
        'order_code': order_code,
        'session_number': (order_code.index(exp_type) + 1
                           if exp_type in order_code else None),
        'cs_plus_label': CS_PLUS_LABELS[cs_plus_modality][int(cs_plus_value)],
        'study_dir': STUDY_DIR,
        # block layout
        'blocks': block_table,
    })
    json_path = os.path.join(out_dir, f"WP1_{tag}_seq.json")
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # --- Save a readable long-form CSV (inspection only) --------------------
    csv_path = os.path.join(out_dir, f"WP1_{tag}_seq.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row', 'block_label', 'freq_step', 'azi_step', 'shock',
                    'pattern', 'base_freq_index', 'base_freq_hz'])
        for blk in block_table:
            for r in range(blk['row_start'], blk['row_end']):
                bf_idx = int(seq_array[r, COL_BASEFREQ])
                bf_hz = '' if bf_idx == NO_BASEFREQ else f"{FREQUENCY_LIST[bf_idx]:.2f}"
                w.writerow([r, blk['label'],
                            seq_array[r, COL_FREQ],
                            seq_array[r, COL_AZI],
                            seq_array[r, COL_SHOCK],
                            seq_array[r, COL_PATTERN],
                            bf_idx, bf_hz])

    total_trials = seq_array.shape[0]
    shocked_trials = int(np.count_nonzero(seq_array[:, COL_SHOCK]))   # trials, not pulses
    total_pulses = int(seq_array[:, COL_SHOCK].sum())                 # cascade counts 5x
    print(f"\n  -> {npy_path}")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")
    print(f"     {total_trials} trials, {len(block_table)} blocks, "
          f"{shocked_trials} shocked trials ({total_pulses} pulses)")

    return {
        'participant_id': participant_id,
        'experiment_type': exp_type,
        'total_trials': total_trials,
        'n_blocks': len(block_table),
        'shocked_trials': shocked_trials,
        'total_pulses': total_pulses,
        'npy': npy_path,
        'json': json_path,
        'csv': csv_path,
        'seed': seed,
    }


def prompt_for_type():
    """Interactive session-type prompt with validation."""
    print(f"Session types:  {TYPE_HELP}")
    choices = ' / '.join(TYPES)
    t = input(f"Enter session type ({choices}): ").strip().lower()
    while t not in TYPES:
        t = input(f"  Invalid — enter one of {choices}: ").strip().lower()
    return t


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate WP1 v3 unified-array sequences.')
    parser.add_argument('n_participants', type=int, nargs='?', default=None,
                        help='Number of participants (batch mode)')
    parser.add_argument('--participant', type=int, default=None,
                        help='Single participant ID (single mode)')
    parser.add_argument('--type', choices=TYPES, default=None, dest='experiment_type',
                        help=f'Session type: {TYPE_HELP}')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random)')
    parser.add_argument('--allocate', type=int, default=None, metavar='N',
                        help='Only write the allocation sheet for N participants '
                             '(order + CS+ per person), no sequences')
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else random.randint(0, 999999)

    print(f"Study folder: {os.path.abspath(STUDY_DIR)}")

    # ---- Allocation sheet only ----
    if args.allocate is not None:
        write_allocation(args.allocate)
        raise SystemExit

    # ---- Single mode ----
    if args.participant is not None:
        exp_type = args.experiment_type or prompt_for_type()
        out_dir = SEQ_DIR
        info = generate_one(args.participant, exp_type, base_seed, out_dir=out_dir)
        print(f"\nDone: {info['npy']}")

    # ---- Batch mode ----
    elif args.n_participants is not None:
        n = args.n_participants
        exp_types = [args.experiment_type] if args.experiment_type else TYPES
        out_dir = SEQ_DIR

        print(f"\n{'#' * 70}")
        print(f"  BATCH GENERATION (v3): {n} participants x types {exp_types}")
        print(f"  Base seed: {base_seed}")
        print(f"  Output dir: {out_dir}/")
        print(f"{'#' * 70}")

        # The schedule sheet for these n participants: order + CS+ per person.
        write_allocation(n)

        summaries = []
        for pid in range(1, n + 1):
            for t in exp_types:
                derived_seed = base_seed + pid * 10 + TYPES.index(t)
                summaries.append(generate_one(pid, t, derived_seed, out_dir))

        print(f"\n{'=' * 70}")
        print(f"WP1 v3 SUMMARY  ({n} participants x {len(exp_types)} types)")
        print(f"{'=' * 70}")
        print(f"{'Sub':>4s} | {'type':>4s} | {'trials':>7s} | {'blocks':>6s} | {'shockd':>6s}")
        print(f"{'':->4s}-+-{'':->4s}-+-{'':->7s}-+-{'':->6s}-+-{'':->6s}")
        for s in summaries:
            print(f"{s['participant_id']:04d} | {s['experiment_type']:>4s} | {s['total_trials']:>7d} | "
                  f"{s['n_blocks']:>6d} | {s['shocked_trials']:>6d}")
        print(f"{'=' * 70}")
        print(f"Files saved to: {out_dir}/\n")

    else:
        print("=== WP1 Sequence Generator (v3) ===")
        participant_id = int(input("Enter participant number: "))
        exp_type = prompt_for_type()
        out_dir = SEQ_DIR
        info = generate_one(participant_id, exp_type, base_seed, out_dir=out_dir)
        print(f"\nDone! Sequence saved to: {info['npy']}")
