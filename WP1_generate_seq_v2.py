"""
Generate WP1 experiment sequences — VERSION 2 (unified array format).

SELF-CONTAINED: this file has no dependency on WP1_generate_seq.py. All the
building blocks (roving, balancing, reinforcement, counterbalancing) are
inlined below so v2 can be run and version-controlled on its own.

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
Batch (N participants):
    python WP1_generate_seq_v2.py 20 --type bf
    python WP1_generate_seq_v2.py 20 --type a --seed 42

Single:
    python WP1_generate_seq_v2.py --participant 1 --type bp

Output per participant:
    WP1_sub001_bf_v2_seq.npy    -> the (n_trials, 5) int array
    WP1_sub001_bf_v2_seq.json   -> metadata (block table, roving lists, timing, CS+)
    WP1_sub001_bf_v2_seq.csv    -> human-readable long form (inspection only)
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
# NOTE: the same line exists at the top of WP1_run_Exp_v2.py and
# WP1_balance.py. Change all three when you switch studies. Each script prints
# the folder it is using when it starts, so a mismatch is visible immediately.

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
REINF_PROB_START = 0.7
REINF_PROB_END = 0.3
SHOCK_ONSET_IN_ITI = 0.1
SOA = ITI + TONE_DURATION

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


def soa_for_type(exp_type):
    """Pattern sessions run at A_SOA; everything else at the standard SOA."""
    return A_SOA if exp_type == 'a' else SOA


def stim_span_for_type(exp_type):
    """On-air duration of one trial's stimulus, for the given session type."""
    return PATTERN_SPAN if exp_type == 'a' else TONE_DURATION


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
# ROVING SEQUENCE GENERATION
# ============================================================================
def create_balanced_deviant_vector(n_trains, max_cumsum=4, max_attempts=1000):
    """Equal numbers of +1 / -1 deviants whose cumsum never leaves +/-max_cumsum."""
    if n_trains % 2 != 0:
        raise ValueError("n_trains must be even to have equal 1s and -1s")

    deviants = [1] * (n_trains // 2) + [-1] * (n_trains // 2)

    for attempt in range(max_attempts):
        random.shuffle(deviants)
        cumsum = np.cumsum(deviants)
        if np.all(cumsum >= -max_cumsum) and np.all(cumsum <= max_cumsum):
            print(f"  Valid sequence found after {attempt + 1} attempt(s)")
            print(f"  Cumsum range: [{np.min(cumsum)}, {np.max(cumsum)}]")
            return deviants

    raise RuntimeError(
        f"Could not find valid sequence after {max_attempts} attempts. "
        f"Try increasing max_attempts or max_cumsum."
    )


def create_roving_sequence(train_lengths=[5, 6, 7, 8, 9, 10],
                           n_deviants=200, max_cumsum=4, soa=0.3,
                           block_size=50):
    """
    Build a 1-D roving sequence: runs of standards (0) each terminated by a
    +1/-1 deviant. Deviants are balanced within sub-blocks of `block_size` so
    the running cumsum stays inside +/-max_cumsum (i.e. the roving index never
    walks off the end of the stimulus ladder).
    """
    original_n_deviants = n_deviants
    if n_deviants % 2 != 0:
        n_deviants += 1
        print(f"  Adjusted to {n_deviants} deviants (must be even for balance)")

    n_blocks = int(np.ceil(n_deviants / block_size))
    base_deviants_per_block = n_deviants // n_blocks
    remainder = n_deviants % n_blocks

    deviants_per_block_list = [base_deviants_per_block] * n_blocks
    for i in range(remainder):
        deviants_per_block_list[i] += 1

    for i in range(len(deviants_per_block_list)):
        if deviants_per_block_list[i] % 2 != 0:
            deviants_per_block_list[i] += 1

    actual_n_deviants = sum(deviants_per_block_list)

    print(f"\n{'=' * 70}")
    print(f"CREATING BLOCK-BASED SEQUENCE")
    print(f"{'=' * 70}")
    print(f"Requested deviants:     {original_n_deviants}")
    print(f"Actual deviants:        {actual_n_deviants}")
    print(f"Number of sub-blocks:   {n_blocks}")
    print(f"Deviants per sub-block: {deviants_per_block_list}")
    print(f"Train lengths:          {train_lengths} (mean: {np.mean(train_lengths):.1f})")
    print(f"Max cumsum:             +/-{max_cumsum}")
    print(f"{'=' * 70}\n")

    all_sequences = []
    all_deviants = []

    for block_num in range(n_blocks):
        deviants_this_block = deviants_per_block_list[block_num]
        print(f"  Sub-block {block_num + 1}/{n_blocks} ({deviants_this_block} deviants)...")

        n_different_lengths = len(train_lengths)
        base_trains_per_length = deviants_this_block // n_different_lengths
        remainder_trains = deviants_this_block % n_different_lengths

        train_distribution = [base_trains_per_length] * n_different_lengths
        if remainder_trains > 0:
            remainder_indices = random.sample(range(n_different_lengths), remainder_trains)
            for idx in remainder_indices:
                train_distribution[idx] += 1

        train_templates = []
        for length_idx, length in enumerate(train_lengths):
            train_templates.extend([[0] * length] * train_distribution[length_idx])

        random.shuffle(train_templates)
        block_deviants = create_balanced_deviant_vector(len(train_templates),
                                                        max_cumsum=max_cumsum)

        block_sequence = []
        for deviant, train_template in zip(block_deviants, train_templates):
            block_sequence.extend(train_template)
            block_sequence.append(deviant)

        all_sequences.append(block_sequence)
        all_deviants.extend(block_deviants)

        block_deviant_values = [x for x in block_sequence if x != 0]
        block_cumsum = np.cumsum(block_deviant_values)
        print(f"    {len(block_deviant_values)} deviants, "
              f"{len(block_sequence)} total tones, "
              f"cumsum range [{np.min(block_cumsum)}, {np.max(block_cumsum)}]")

    sequence = [tone for block in all_sequences for tone in block]

    deviant_positions = [i for i, val in enumerate(sequence) if val != 0]
    actual_train_lengths = []
    for i in range(len(deviant_positions)):
        if i == 0:
            train_len = deviant_positions[0]
        else:
            train_len = deviant_positions[i] - deviant_positions[i - 1] - 1
        actual_train_lengths.append(train_len)

    mean_train_length = np.mean(actual_train_lengths)
    total_seconds = len(sequence) * soa
    total_minutes = total_seconds / 60

    print(f"\n{'=' * 70}")
    print(f"FINAL SEQUENCE")
    print(f"{'=' * 70}")
    print(f"Total deviants:      {len(all_deviants)}")
    print(f"  - Up (+1):         {sum(1 for x in all_deviants if x == 1)}")
    print(f"  - Down (-1):       {sum(1 for x in all_deviants if x == -1)}")
    print(f"Total standards:     {sum(1 for x in sequence if x == 0)}")
    print(f"Total tones:         {len(sequence)}")
    print(f"Mean train length:   {mean_train_length:.2f} standards")
    print(f"Total run time:      {total_minutes:.2f} min ({total_seconds:.0f} sec)")
    print(f"{'=' * 70}\n")

    return sequence, all_deviants


# ============================================================================
# REINFORCEMENT SCHEDULE
# ============================================================================
def generate_reinforcement_schedule(sequence, cs_plus_value, n_shock=100,
                                    prob_start=0.9, prob_end=0.1,
                                    cs_minus_shocks=1, cs_plus_shocks=5):
    """
    Returns num_shock per trial:
    - CS+ shocked trials: cs_plus_shocks (5) — cascade
    - CS+ clean trials: 0
    - CS- shocked trials: cs_minus_shocks (1) — tap control
    - CS- clean trials: 0

    Uses deterministic weighted selection (early bias) for both CS+ and CS-.
    Same number of shocked trials for CS+ and CS- (balanced).
    """
    cs_plus_indices = [i for i, s in enumerate(sequence) if s == cs_plus_value]
    cs_minus_indices = [i for i, s in enumerate(sequence) if s == -cs_plus_value]
    n_cs_plus = len(cs_plus_indices)
    n_cs_minus = len(cs_minus_indices)

    num_shock = [0] * len(sequence)

    if n_cs_plus == 0:
        return num_shock, {'n_cs_plus': 0, 'n_reinforced': 0, 'n_clean_cs_plus': 0}

    n_shock = min(n_shock, n_cs_plus, n_cs_minus)

    def select_shocked_indices(indices, n_select):
        """Linear-decay weighted selection: earlier deviants more likely shocked."""
        n_total = len(indices)
        if n_total == 0 or n_select == 0:
            return set()
        weights = np.array([
            prob_start - (prob_start - prob_end) * (r / max(n_total - 1, 1))
            for r in range(n_total)
        ])
        weights /= weights.sum()
        return set(np.random.choice(n_total, size=n_select, replace=False, p=weights))

    cs_plus_shocked_ranks = select_shocked_indices(cs_plus_indices, n_shock)
    for rank, trial_idx in enumerate(cs_plus_indices):
        if rank in cs_plus_shocked_ranks:
            num_shock[trial_idx] = cs_plus_shocks  # cascade

    cs_minus_shocked_ranks = select_shocked_indices(cs_minus_indices, n_shock)
    for rank, trial_idx in enumerate(cs_minus_indices):
        if rank in cs_minus_shocked_ranks:
            num_shock[trial_idx] = cs_minus_shocks  # tap

    n_clean_cs_plus = n_cs_plus - n_shock
    n_clean_cs_minus = n_cs_minus - n_shock

    print(f"\n{'=' * 70}")
    print(f"REINFORCEMENT SCHEDULE (balanced tap control)")
    print(f"{'=' * 70}")
    print(f"CS+ value:           {cs_plus_value:+d}")
    print(f"Total CS+ trials:    {n_cs_plus}")
    print(f"CS+ cascade (x{cs_plus_shocks}):   {n_shock}")
    print(f"CS+ clean (0):       {n_clean_cs_plus}")
    print(f"CS- total:           {n_cs_minus}")
    print(f"CS- tap (x{cs_minus_shocks}):       {n_shock}")
    print(f"CS- clean (0):       {n_clean_cs_minus}")
    print(f"{'=' * 70}\n")

    schedule_info = {
        'n_cs_plus': n_cs_plus,
        'n_cs_minus': n_cs_minus,
        'n_reinforced': n_shock,
        'n_clean_cs_plus': n_clean_cs_plus,
        'n_clean_cs_minus': n_clean_cs_minus,
        'cs_minus_shocks': cs_minus_shocks,
        'cs_plus_shocks': cs_plus_shocks,
    }
    return num_shock, schedule_info


# ============================================================================
# COUNTERBALANCING
# ============================================================================
def get_cs_plus_assignment(participant_id):
    """
    CS+ direction (+1 / -1) per stimulus dimension, read from allocation.csv
    in the study folder. That file is the single source of truth for both
    session order and CS+ assignment — build it with:

        python WP1_balance.py --n 24

    Mappings:
    f: +1 (High) / -1 (Low)
    p: +1 (Right) / -1 (Left)
    a: +1 (Up)    / -1 (Down)
    """
    return read_allocation()[int(participant_id)]['cs_plus']


def read_allocation():
    """{participant_id: {'order_code', 'session_types', 'cs_plus'}} from the study folder."""
    if not os.path.exists(ALLOCATION_FILE):
        raise SystemExit(
            f"\nERROR: no allocation table at '{ALLOCATION_FILE}'.\n"
            f"Run:  python WP1_balance.py --n 24\n")
    out = {}
    with open(ALLOCATION_FILE, newline='') as f:
        for r in csv.DictReader(f):
            out[int(r['participant_id'])] = {
                'order_code': r['order_code'],
                'session_types': [r['session_1_type'], r['session_2_type'],
                                  r['session_3_type']],
                'cs_plus': {'f': int(r['cs_plus_f']),
                            'p': int(r['cs_plus_p']),
                            'a': int(r['cs_plus_a'])},
            }
    return out


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

    For modality 'a' the sequence is still generated by create_roving_sequence
    (so up/down deviants stay balanced and evenly spaced), but the values are
    written as pattern IDENTITIES rather than roving steps, and every row gets
    a pre-drawn base-frequency index in COL_BASEFREQ.
    """
    if len(modalities) == 1:
        m = modalities[0]
        seq, _ = create_roving_sequence(n_deviants=n_deviants,
                                        max_cumsum=MAX_CUMSUM, soa=soa)
        seq = np.asarray(seq, dtype=int)
        block = np.zeros((len(seq), N_COLS), dtype=int)
        block[:, COL_BASEFREQ] = NO_BASEFREQ
        block[:, MODALITY_COL[m]] = seq
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
        streams = {}
        for m in modalities:
            seq, _ = create_roving_sequence(n_deviants=n_deviants,
                                            max_cumsum=MAX_CUMSUM, soa=soa)
            streams[m] = np.asarray(seq, dtype=int)

        # The CS+ modality anchors the timeline; the other one yields, so
        # conditioning timing is never perturbed.
        other = [m for m in modalities if m != cs_plus_modality]
        if len(streams) != 2 or not other:
            raise ValueError(f"expected exactly 2 modalities incl. the CS+, got {modalities}")
        mover_mod = other[0]

        anchor, mover, n_moved, shifts = enforce_min_separation(
            streams[cs_plus_modality], streams[mover_mod])

        block = np.zeros((len(anchor), N_COLS), dtype=int)
        block[:, COL_BASEFREQ] = NO_BASEFREQ
        block[:, MODALITY_COL[cs_plus_modality]] = anchor
        block[:, MODALITY_COL[mover_mod]] = mover

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

    # Reinforcement: only the CS+ modality's deviants can be shocked.
    if use_reinforcement:
        cs_col = MODALITY_COL[cs_plus_modality]
        num_shock, _ = generate_reinforcement_schedule(
            block[:, cs_col].tolist(), cs_plus_value,
            n_shock=N_SHOCK, prob_start=REINF_PROB_START, prob_end=REINF_PROB_END,
        )
        block[:, COL_SHOCK] = np.asarray(num_shock, dtype=int)

    return block


# ============================================================================
# BUILD ONE PARTICIPANT
# ============================================================================
def generate_one_v2(participant_id, exp_type, seed, out_dir=SEQ_DIR):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    tcfg = TYPE_CONFIG[exp_type]
    modalities = tcfg['modalities']
    cs_plus_modality = tcfg['cs_plus_modality']

    # CS+ direction per modality, from the study folder's allocation.csv.
    alloc = read_allocation()
    if int(participant_id) not in alloc:
        raise SystemExit(f"\nERROR: participant {participant_id} is not in "
                         f"'{ALLOCATION_FILE}'. Extend it with WP1_balance.py.\n")
    alloc_row = alloc[int(participant_id)]
    assignment = alloc_row['cs_plus']                    # {'f':±1,'p':±1,'a':±1}
    cs_plus_value = assignment[cs_plus_modality]

    # Timing depends on the session type: pattern trials are longer on air and
    # therefore need a longer SOA.
    soa = soa_for_type(exp_type)
    stim_span = stim_span_for_type(exp_type)

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
        n = block.shape[0]
        block_rows.append(block)

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
            'n_freq_deviants': int(np.count_nonzero(block[:, COL_FREQ])),
            'n_azi_deviants': int(np.count_nonzero(block[:, COL_AZI])),
            'n_pattern_deviants': int(np.count_nonzero(block[:, COL_PATTERN])),
            'n_shock_trials': int(np.count_nonzero(block[:, COL_SHOCK])),
        })
        cursor += n

    seq_array = np.vstack(block_rows).astype(int)  # (total_trials, N_COLS)

    tag = f"sub{participant_id:03d}_{exp_type}_v2"

    # --- Save the numpy array (the primary v2 artefact) ---------------------
    npy_path = os.path.join(out_dir, f"WP1_{tag}_seq.npy")
    np.save(npy_path, seq_array)

    # --- Save metadata ------------------------------------------------------
    meta = sanitize_for_json({
        'version': 2,
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
        # per-modality CS+ assignment, and where it came from
        'cs_plus_assignment': {k: int(v) for k, v in assignment.items()},
        'order_code': alloc_row['order_code'],
        'session_number': (alloc_row['session_types'].index(exp_type) + 1
                           if exp_type in alloc_row['session_types'] else None),
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
    total_shocks = int(seq_array[:, COL_SHOCK].sum())
    print(f"\n  -> {npy_path}")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")
    print(f"     {total_trials} trials, {len(block_table)} blocks, "
          f"{total_shocks} total shock pulses")

    return {
        'participant_id': participant_id,
        'experiment_type': exp_type,
        'total_trials': total_trials,
        'n_blocks': len(block_table),
        'total_shocks': total_shocks,
        'npy': npy_path,
        'json': json_path,
        'csv': csv_path,
        'seed': seed,
    }


# ============================================================================
# CONVENIENCE: resolve an array row back to actual stimulus values
# ============================================================================
def resolve_pattern_row(pattern_value, base_freq_index,
                        frequency_list=None, step=PATTERN_STEP):
    """
    Turn a COL_PATTERN / COL_BASEFREQ pair into the actual tone frequencies.
    Mirrors v1's resolve_values_pattern, but reads the pre-drawn base freq
    from the array instead of drawing it live.
    """
    frequency_list = frequency_list if frequency_list is not None else FREQUENCY_LIST
    base_freq = frequency_list[int(base_freq_index)]
    step_mult = 1 + step
    freq_map = {0: base_freq, 1: base_freq * step_mult, -1: base_freq / step_mult}
    structure = PATTERN_STRUCTURES[int(pattern_value)]
    names = {0: 'standard', 1: 'up', -1: 'down'}
    return {
        'pattern_name': names[int(pattern_value)],
        'base_freq': base_freq,
        'structure': structure,
        'frequencies': [freq_map[t] for t in structure],
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
    parser = argparse.ArgumentParser(description='Generate WP1 v2 unified-array sequences.')
    parser.add_argument('n_participants', type=int, nargs='?', default=None,
                        help='Number of participants (batch mode)')
    parser.add_argument('--participant', type=int, default=None,
                        help='Single participant ID (single mode)')
    parser.add_argument('--type', choices=TYPES, default=None, dest='experiment_type',
                        help=f'Session type: {TYPE_HELP}')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random)')
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else random.randint(0, 999999)

    # ---- Single mode ----
    if args.participant is not None:
        exp_type = args.experiment_type or prompt_for_type()
        out_dir = SEQ_DIR
        info = generate_one_v2(args.participant, exp_type, base_seed, out_dir=out_dir)
        print(f"\nDone: {info['npy']}")

    # ---- Batch mode ----
    elif args.n_participants is not None:
        n = args.n_participants
        exp_types = [args.experiment_type] if args.experiment_type else TYPES
        out_dir = SEQ_DIR

        print(f"\n{'#' * 70}")
        print(f"  BATCH GENERATION (v2): {n} participants x types {exp_types}")
        print(f"  Base seed: {base_seed}")
        print(f"  Output dir: {out_dir}/")
        print(f"{'#' * 70}")

        summaries = []
        for pid in range(1, n + 1):
            for t in exp_types:
                derived_seed = base_seed + pid * 10 + TYPES.index(t)
                summaries.append(generate_one_v2(pid, t, derived_seed, out_dir))

        print(f"\n{'=' * 70}")
        print(f"WP1 v2 SUMMARY  ({n} participants x {len(exp_types)} types)")
        print(f"{'=' * 70}")
        print(f"{'Sub':>4s} | {'type':>4s} | {'trials':>7s} | {'blocks':>6s} | {'shocks':>6s}")
        print(f"{'':->4s}-+-{'':->4s}-+-{'':->7s}-+-{'':->6s}-+-{'':->6s}")
        for s in summaries:
            print(f"{s['participant_id']:04d} | {s['experiment_type']:>4s} | {s['total_trials']:>7d} | "
                  f"{s['n_blocks']:>6d} | {s['total_shocks']:>6d}")
        print(f"{'=' * 70}")
        print(f"Files saved to: {out_dir}/\n")

    else:
        print("=== WP1 Sequence Generator (v2) ===")
        participant_id = int(input("Enter participant number: "))
        exp_type = prompt_for_type()
        out_dir = SEQ_DIR
        info = generate_one_v2(participant_id, exp_type, base_seed, out_dir=out_dir)
        print(f"\nDone! Sequence saved to: {info['npy']}")
