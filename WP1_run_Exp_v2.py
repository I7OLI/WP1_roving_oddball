"""
Run WP1 experiment — VERSION 2 (unified array format).

Reads the v2 artefacts produced by WP1_generate_seq_v2.py:
    WP1_sub001_<type>_v2_seq.npy    -> (n_trials, 5) int array
                                       [freq_step, azi_step, shock, pattern, base_freq_idx]
    WP1_sub001_<type>_v2_seq.json   -> metadata (block table, roving lists, timing, CS+)

The runner iterates the array ROW BY ROW, builds the stimulus, plays it, and
fires the shock in column 2 (0/1/5) if non-zero.

Two stimulus families, one loop:
  * f / p / bf / bp  -- columns 0 and 1 are ROVING STEPS. Advance the roving
                        index for whichever column carries the step, clamp it,
                        and resolve to Hz + azimuth. One tone per trial.
  * a                -- column 3 is an ABSOLUTE PATTERN IDENTITY (not a step),
                        column 4 is a direct index into frequency_list giving
                        the pattern's base frequency. Two half-length tones
                        separated by iti_within_pattern, concatenated into ONE
                        buffer so within-pattern timing is hardware-precise.

Timing is type-specific and comes from the JSON:
    SOA        0.3 s for f/p, A_SOA (0.65 s) for 'a'
    stim_span  0.1 s for f/p, 0.2 s for 'a' (tone + gap + tone)
    shock      fires at  onset + stim_span + shock_delay_after_offset,
               i.e. a fixed delay after STIMULUS OFFSET in both families.

Usage:
    python WP1_run_Exp_v2.py sequences/WP1_sub001_bf_v2_seq.npy            # with Pupil
    python WP1_run_Exp_v2.py sequences/WP1_sub001_a_v2_seq.npy --no-pupil  # without
"""
import os
import sys
import csv
import json
import time
import argparse

import numpy as np
import slab
import freefield as ff

# zmq / msgpack are only needed for Pupil recording; import lazily so the script
# still runs without them when Pupil is off.
try:
    import zmq
    import msgpack as serializer
except ImportError:
    zmq = serializer = None

# ============================================================================
# PUPIL TOGGLE
# ============================================================================
# Set this to False to run WITHOUT the Pupil Labs eye-tracker (no recording,
# no annotations, no waiting for Pupil Capture). Can also be overridden on the
# command line with  --pupil  /  --no-pupil .
USE_PUPIL = True

# Shock INTENSITY (number of hardware pulses), set here at run time. The
# sequence file only decides WHICH trials are shocked; these decide how hard.
SHOCK_HIGH = 4                      # CS+ cascade  (change this per session)
SHOCK_LOW = 1                       # CS- tap control

# ============================================================================
# OUTPUT LEVEL
# ============================================================================
# slab's default tone level puts the peak at ~0.09 of full scale, i.e. ~16 dB
# of unused headroom, which forces the amplifier gain up and makes its noise
# floor audible as a buzz. Worst-case peak across all freq x azimuth combos is
# 0.160, so x5.0 lands at 0.80 peak with room to spare.
#
# CRITICAL: this is ONE GLOBAL constant applied equally to both channels.
# Do NOT normalise per trial or per channel — the interaural level difference
# IS the azimuth cue, and per-trial normalisation would destroy it.
#
# CALIBRATION: raising this WILL make the output louder. Turn the amplifier
# down by the same amount (x5.0 = 14 dB) and re-verify dB SPL at the ear with
# a meter before collecting data. Loudness drives pupil dilation, so a level
# change silently breaks comparability with anything recorded at the old
# setting. Set to 1.0 to reproduce the previous (buzzy) behaviour exactly.
OUTPUT_SCALE = 5.0
MAX_SAFE_PEAK = 0.95                # warn if scaling pushes us near clipping

# Array columns (must match WP1_generate_seq_v2.py)
COL_FREQ, COL_AZI, COL_SHOCK = 0, 1, 2
COL_PATTERN, COL_BASEFREQ = 3, 4
NO_BASEFREQ = -1                    # sentinel: this trial has no pattern base

fs = 48828.125
slab.set_default_samplerate(fs)
rcx_file = 'shock.rcx'
procsser = 'RM1'

ff.initialize(
    setup='headphones',
    device=[['RM1', procsser, rcx_file]],
    zbus=False,
    connection='usb'
)

# ============================================================================
# PUPIL LABS SETUP  (unchanged from v1)
# ============================================================================

def connect_to_pupil():
    ctx = zmq.Context()
    pupil_remote = zmq.Socket(ctx, zmq.REQ)
    pupil_remote.connect('tcp://127.0.0.1:50020')
    pupil_remote.send_string("PUB_PORT")
    pub_port = pupil_remote.recv_string()
    pub_socket = zmq.Socket(ctx, zmq.PUB)
    pub_socket.connect(f"tcp://127.0.0.1:{pub_port}")
    time.sleep(0.5)
    print(f"Connected to Pupil Capture (PUB port: {pub_port})")
    return pupil_remote, pub_socket


def measure_clock_offset(pupil_remote, n_samples=20):
    """Offset such that pupil_time = local_time + offset."""
    offsets = []
    for _ in range(n_samples):
        t0 = time.time()
        pupil_remote.send_string("t")
        pupil_time = float(pupil_remote.recv_string())
        offsets.append(pupil_time - (t0 + time.time()) / 2)
    offset = float(np.median(offsets))
    print(f"Clock offset: {offset:.6f}s  (jitter ±{np.std(offsets) * 1000:.3f}ms)")
    return offset


def pupil_notify(pupil_remote, notification):
    topic = "notify." + notification["subject"]
    payload = serializer.dumps(notification, use_bin_type=True)
    pupil_remote.send_string(topic, flags=zmq.SNDMORE)
    pupil_remote.send(payload)
    return pupil_remote.recv_string()


def send_annotation(pub_socket, label, clock_offset,
                    local_timestamp=None, duration=0.0, extra=None):
    if pub_socket is None:      # Pupil disabled -> annotations are no-ops
        return
    if local_timestamp is None:
        local_timestamp = time.time()
    pupil_timestamp = local_timestamp + clock_offset
    annotation = {
        "topic": "annotation",
        "label": label,
        "timestamp": pupil_timestamp,
        "duration": duration,
    }
    if extra:
        annotation.update(extra)
    payload = serializer.dumps(annotation, use_bin_type=True)
    pub_socket.send_string(annotation["topic"], flags=zmq.SNDMORE)
    pub_socket.send(payload)


def start_pupil_recording(pupil_remote, pub_socket, clock_offset, recording_dir):
    os.makedirs(recording_dir, exist_ok=True)
    pupil_notify(pupil_remote, {"subject": "start_plugin", "name": "Annotation_Capture", "args": {}})
    pupil_notify(pupil_remote, {"subject": "start_plugin", "name": "Recorder",
                                "args": {"rec_root_dir": recording_dir}})
    time.sleep(0.5)
    pupil_remote.send_string('R')
    print(f"Pupil recording started: {pupil_remote.recv_string()}")
    print(f"Saving eye data to:      {recording_dir}")
    send_annotation(pub_socket, "experiment_start", clock_offset)


def stop_pupil_recording(pupil_remote, pub_socket, clock_offset):
    send_annotation(pub_socket, "experiment_end", clock_offset)
    time.sleep(0.5)
    pupil_remote.send_string('r')
    print(f"Pupil recording stopped: {pupil_remote.recv_string()}")


# ============================================================================
# HELPERS
# ============================================================================

def precise_sleep_until(target_time, busy_wait_threshold=0.002):
    if target_time - time.time() > busy_wait_threshold:
        time.sleep(target_time - time.time() - busy_wait_threshold)
    while time.time() < target_time:
        pass


_clip_warned = False


def load_tone(tone):
    """
    Write one binaural stimulus into the RM1 play buffer.

    Applies OUTPUT_SCALE equally to both channels — a single global gain, so
    the interaural level difference (the azimuth cue) is preserved exactly.
    Buffers are flattened to 1-D, since tone.left.data is a (n, 1) column.
    """
    global _clip_warned
    left = np.asarray(tone.left.data, dtype=float).flatten() * OUTPUT_SCALE
    right = np.asarray(tone.right.data, dtype=float).flatten() * OUTPUT_SCALE

    peak = max(np.abs(left).max(), np.abs(right).max())
    if peak > MAX_SAFE_PEAK and not _clip_warned:
        print(f"\n*** WARNING: peak {peak:.3f} exceeds {MAX_SAFE_PEAK} with "
              f"OUTPUT_SCALE={OUTPUT_SCALE}. Reduce OUTPUT_SCALE or you will "
              f"clip. (This warning prints once.) ***\n")
        _clip_warned = True

    ff.write('playbuflen', len(tone), procsser)
    ff.write('data_l', left, procsser)
    ff.write('chan_l', 1, procsser)
    ff.write('data_r', right, procsser)
    ff.write('chan_r', 2, procsser)


def build_pattern_sound(base_freq, structure, pattern_step,
                        pattern_tone_duration, iti_within_pattern):
    """
    Concatenate a pattern's tones + inter-tone silence into ONE buffer.

    A single write + single trigger means within-pattern timing is
    hardware-precise rather than at the mercy of Python sleeps.
    Returns (binaural_sound, [frequencies]).
    """
    step_mult = 1 + pattern_step
    freq_map = {0: base_freq, 1: base_freq * step_mult, -1: base_freq / step_mult}
    freqs = [freq_map[t] for t in structure]

    silence_samples = int(iti_within_pattern * slab.get_default_samplerate())
    tones = [
        slab.Sound.tone(frequency=f, duration=pattern_tone_duration, n_channels=2)
                  .ramp('offset', 0.05)
        for f in freqs
    ]
    n_ch = tones[0].data.shape[1]
    parts = []
    for idx, tone in enumerate(tones):
        parts.append(tone.data)
        if idx < len(tones) - 1:
            parts.append(np.zeros((silence_samples, n_ch)))

    combined = slab.Binaural(np.vstack(parts),
                             samplerate=slab.get_default_samplerate())
    return combined, freqs


def build_trials(block_rows, cs_plus_modality, cs_plus_value, meta):
    """
    One pass over a block's (n, 5) rows: resolve each row to a stimulus, build
    its sound buffer, and precompute all labels. Returns a list of trial dicts.
    Roving indices reset to centre at the start of every block.

    Handles both stimulus families. Note the semantic difference:
      cols 0/1 are roving STEPS  -> accumulate into an index, then clamp
      col  3   is an absolute PATTERN IDENTITY -> look up directly, no accumulation
    """
    freq_list, pos_list = meta['frequency_list'], meta['position_list']
    freq_idx, azi_idx = meta['freq_center_index'], meta['azi_center_index']
    tone_duration = meta['tone_duration']

    # Pattern params (only present / needed for type 'a')
    pattern_structures = {int(k): v for k, v in meta.get('pattern_structures', {}).items()}
    pattern_step = meta.get('pattern_step', 0.15)
    pattern_tone_duration = meta.get('pattern_tone_duration', tone_duration / 2)
    iti_within_pattern = meta.get('iti_within_pattern', 0.1)

    pattern_names = {0: 'standard', 1: 'up', -1: 'down'}

    trials = []
    for row in block_rows:
        f_step = int(row[COL_FREQ])
        a_step = int(row[COL_AZI])
        shock = int(row[COL_SHOCK])
        pat = int(row[COL_PATTERN])
        bf_idx = int(row[COL_BASEFREQ])

        is_pattern_trial = bf_idx != NO_BASEFREQ

        if is_pattern_trial:
            # --- abstract-pattern trial: identity lookup, NO roving ----------
            base_freq = freq_list[bf_idx]
            sound, freqs = build_pattern_sound(
                base_freq, pattern_structures[pat], pattern_step,
                pattern_tone_duration, iti_within_pattern)

            dev_modality = 'a' if pat != 0 else None
            step = pat
            freq_hz, azi_deg = base_freq, 0.0
            pattern_name = pattern_names[pat]
        else:
            # --- roving tone trial (f / p / bf / bp) -------------------------
            freq_idx = max(0, min(freq_idx + f_step, len(freq_list) - 1))
            azi_idx = max(0, min(azi_idx + a_step, len(pos_list) - 1))
            freq_hz, azi_deg = freq_list[freq_idx], pos_list[azi_idx]

            if f_step != 0:
                dev_modality, step = 'f', f_step
            elif a_step != 0:
                dev_modality, step = 'p', a_step
            else:
                dev_modality, step = None, 0

            sound = slab.Binaural(
                slab.Sound.tone(frequency=freq_hz, duration=tone_duration, n_channels=2)
            ).at_azimuth(azi_deg).ramp('offset', 0.05)
            freqs = [freq_hz]
            pattern_name = ''

        # CS labelling is identical for both families
        if dev_modality is None:
            cs_label = "STD"
        elif dev_modality == cs_plus_modality:
            cs_label = "CS+" if step == cs_plus_value else "CS-"
        else:
            cs_label = "ODD"                        # deviant in the non-CS+ dimension

        trials.append({
            'freq_hz': freq_hz, 'azi_deg': azi_deg,
            'freq_step': f_step, 'azi_step': a_step, 'step': step,
            'pattern': pat, 'pattern_name': pattern_name,
            'base_freq': freq_hz if is_pattern_trial else '',
            'frequencies': freqs, 'is_pattern': is_pattern_trial,
            'shock': shock, 'dev_modality': dev_modality,
            'marker': 'DEV' if step != 0 else 'STD',
            'cs_label': cs_label,
            'is_cs_plus': dev_modality == cs_plus_modality and step == cs_plus_value,
            'tone': sound,
        })
    return trials


# ============================================================================
# BLOCK RUNNER
# ============================================================================

def run_block(block_meta, trials, participant_id, SOA, trial_log,
              pub_socket, clock_offset, shock_time):
    """
    Play one block row-by-row, annotate Pupil, log every trial.

    `shock_time` is measured from trial onset and already accounts for the
    stimulus span, so it is 0.1 s for f/p and 0.2 s for 'a' — a constant delay
    after stimulus OFFSET in both cases.
    """
    label, block_num = block_meta['label'], block_meta['block_num']
    n = len(trials)

    print(f"\n{'=' * 70}")
    print(f"PLAYING {label}  (CS+={block_meta['cs_plus_modality']}{block_meta['cs_plus_value']:+d})")
    print(f"{'=' * 70}\n")

    send_annotation(pub_socket, f"block_{block_num}_start", clock_offset,
                    extra={"block_label": label, "cs_plus_modality": block_meta['cs_plus_modality']})

    load_tone(trials[0]['tone'])                    # prime buffer with first tone

    for i, t in enumerate(trials):
        shocked = t['shock'] > 0
        # Sequence says shock/no-shock; SHOCK_HIGH/LOW set the intensity here.
        n_pulses = (SHOCK_HIGH if t['is_cs_plus'] else SHOCK_LOW) if shocked else 0

        t_onset = time.time()                       # timestamp just before trigger
        ff.play(1, [procsser])
        send_annotation(pub_socket, f"trial_{t['marker']}_{t['cs_label']}", clock_offset,
                        local_timestamp=t_onset,
                        extra={"block": block_num, "trial": i + 1,
                               "modality": t['dev_modality'] or "std",
                               "cs_label": t['cs_label'], "marker": t['marker'],
                               "freq": t['freq_hz'], "azimuth": t['azi_deg'],
                               "pattern": t['pattern_name'],
                               "shock": n_pulses})
        ff.wait_to_finish_playing()

        shock_note = f" | SHOCK x{n_pulses}" if shocked else ""
        if t['is_pattern']:
            freqs_str = '-'.join(f"{f:.0f}" for f in t['frequencies'])
            print(f"Trial {i + 1:3d}/{n}: {t['marker']} {t['cs_label']:>3s} "
                  f"[a] | {t['pattern_name']:>8s} | base={t['freq_hz']:7.1f} Hz | "
                  f"tones={freqs_str}{shock_note}")
        else:
            print(f"Trial {i + 1:3d}/{n}: {t['marker']} {t['cs_label']:>3s} "
                  f"[{t['dev_modality'] or '-'}] | freq={t['freq_hz']:7.1f} Hz | "
                  f"azi={t['azi_deg']:+4.0f} deg{shock_note}")

        if i + 1 < n:                               # preload next tone during SOA
            load_tone(trials[i + 1]['tone'])

        if shocked:                                 # fire the runtime intensity
            precise_sleep_until(t_onset + shock_time)
            ff.write('num_shock', n_pulses, procsser)
            ff.play(2, [procsser])
            send_annotation(pub_socket, "shock", clock_offset, local_timestamp=time.time(),
                            extra={"block": block_num, "trial": i + 1, "num_shock": n_pulses})

        precise_sleep_until(t_onset + SOA)

        trial_log.append({
            'participant_id': participant_id, 'block': block_num, 'block_label': label,
            'trial_num': i + 1, 'dev_modality': t['dev_modality'] or '',
            'freq_step': t['freq_step'], 'azi_step': t['azi_step'],
            'pattern': t['pattern'], 'pattern_name': t['pattern_name'],
            'base_freq': t['base_freq'],
            'pattern_freqs': '-'.join(f"{f:.1f}" for f in t['frequencies']) if t['is_pattern'] else '',
            'sequence_value': t['step'],
            'trial_type': t['cs_label'], 'freq_hz': t['freq_hz'], 'azi_deg': t['azi_deg'],
            'is_cs_plus': t['is_cs_plus'], 'shock_amount': n_pulses,
            'shock_delivered': shocked, 'timestamp': t_onset,
        })

    send_annotation(pub_socket, f"block_{block_num}_end", clock_offset, extra={"block_label": label})


# ============================================================================
# MAIN
# ============================================================================

def resolve_seq_file(seq_file):
    """Return the .npy path, prompting interactively if none was given."""
    if seq_file is not None:
        return seq_file
    print("=== WP1 Experiment Runner (v2) ===")
    pid = int(input("Enter participant number: "))
    print("Session types:  f = freq,  p = azimuth,  a = abstract pattern,  "
          "bf = both (freq CS+),  bp = both (azimuth CS+)")
    t = input("Enter session type (f / p / a / bf / bp): ").strip().lower()
    while t not in ('f', 'p', 'a', 'bf', 'bp'):
        t = input("  Invalid — enter f, p, a, bf, or bp: ").strip().lower()
    path = os.path.join('sequences', f"WP1_sub{pid:03d}_{t}_v2_seq.npy")
    if not os.path.exists(path):
        sys.exit(f"\nERROR: '{path}' not found. Run WP1_generate_seq_v2.py first.")
    print(f"Found sequence file: {path}\n")
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run WP1 v2 experiment.')
    parser.add_argument('seq_file', nargs='?', default=None,
                        help='Path to the WP1_..._v2_seq.npy file')
    parser.add_argument('--pupil', dest='pupil', action='store_true', help='Force Pupil ON')
    parser.add_argument('--no-pupil', dest='pupil', action='store_false', help='Run without Pupil')
    parser.set_defaults(pupil=USE_PUPIL)
    args = parser.parse_args()

    use_pupil = args.pupil
    if use_pupil and zmq is None:
        print("WARNING: pyzmq / msgpack not installed -> running WITHOUT Pupil.")
        use_pupil = False

    npy_file = resolve_seq_file(args.seq_file)
    json_file = os.path.splitext(npy_file)[0] + '.json'
    if not os.path.exists(json_file):
        sys.exit(f"\nERROR: metadata file '{json_file}' not found next to the .npy")

    seq_array = np.load(npy_file)
    with open(json_file) as f:
        meta = json.load(f)

    participant_id = meta['participant_id']
    exp_type = meta.get('experiment_type', 'v2')
    tone_duration = meta['tone_duration']
    SOA = meta.get('SOA', meta['ITI'] + tone_duration)
    blocks = meta['blocks']

    # Shock fires a fixed delay after STIMULUS OFFSET, so the interval from the
    # deviance-defining event to the shock is the same for tones and patterns.
    # Falls back to the old absolute-from-onset behaviour for pre-'a' JSONs.
    stim_span = meta.get('stim_span', tone_duration)
    shock_delay = meta.get('shock_delay_after_offset',
                           meta.get('shock_onset_in_iti', 0.1) - tone_duration)
    shock_time = meta.get('shock_time_from_onset', stim_span + shock_delay)

    if stim_span + shock_delay > SOA:
        sys.exit(f"\nERROR: shock would fire at {stim_span + shock_delay:.3f}s but "
                 f"SOA is only {SOA:.3f}s. Check the timing in {json_file}.")

    print(f"\n{'=' * 70}\nWP1 EXPERIMENT (v2)\n{'=' * 70}")
    print(f"Sequence file:   {npy_file}")
    print(f"Participant ID:  {participant_id}")
    print(f"Session type:    {exp_type}  (CS+ modality = {meta.get('cs_plus_modality')})")
    print(f"Array shape:     {seq_array.shape}")
    print(f"Blocks:          {len(blocks)}")
    print(f"SOA:             {SOA * 1000:.0f} ms  (stimulus span {stim_span * 1000:.0f} ms)")
    print(f"Shock at:        {shock_time * 1000:.0f} ms from onset "
          f"({shock_delay * 1000:+.0f} ms rel. stimulus offset)")
    print(f"Output scale:    x{OUTPUT_SCALE:g}"
          f"{'  <-- CHECK AMP GAIN / SPL CALIBRATION' if OUTPUT_SCALE != 1.0 else ''}")
    print(f"Seed:            {meta.get('random_seed', 'unknown')}")
    print(f"Pupil recording: {'ON' if use_pupil else 'OFF'}\n{'=' * 70}\n")

    recording_dir = os.path.join(
        r"C:\Users\neurobio\Projects\WP1_roving_oddball\recordings",
        f"sub{participant_id:03d}_{exp_type}_v2")

    if use_pupil:
        pupil_remote, pub_socket = connect_to_pupil()
        clock_offset = measure_clock_offset(pupil_remote)
        start_pupil_recording(pupil_remote, pub_socket, clock_offset, recording_dir)
    else:
        pupil_remote = pub_socket = None
        clock_offset = 0.0
        print(">>> Running WITHOUT Pupil (no eye recording / annotations) <<<\n")

    trial_log = []
    for b_i, block_meta in enumerate(blocks, start=1):
        block_meta = dict(block_meta, block_num=b_i)
        rows = seq_array[block_meta['row_start']:block_meta['row_end']]
        trials = build_trials(rows, block_meta['cs_plus_modality'],
                              block_meta['cs_plus_value'], meta)

        n_devs = sum(t['step'] != 0 for t in trials)
        print(f"\n>>> PREPARING {block_meta['label']} <<<")
        print(f"    {len(trials)} trials, {n_devs} deviants")
        input(f"\nPress Enter to start {block_meta['label']}...")

        run_block(block_meta, trials, participant_id, SOA, trial_log,
                  pub_socket, clock_offset, shock_time)

        if b_i < len(blocks):
            input(f"\n{block_meta['label']} COMPLETE. Press Enter to continue...")

    if use_pupil:
        stop_pupil_recording(pupil_remote, pub_socket, clock_offset)

    print(f"\n{'=' * 70}\nEXPERIMENT COMPLETE\n{'=' * 70}")

    # --- Save behavioural CSV + metadata sidecar ---------------------------
    csv_filename = f"WP1_sub{participant_id:03d}_{exp_type}_v2_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trial_log[0].keys())
        writer.writeheader()
        writer.writerows(trial_log)
    print(f"Behavioural data saved to: {csv_filename}")

    meta_filename = csv_filename.replace('.csv', '_meta.json')
    with open(meta_filename, 'w') as f:
        json.dump({"participant_id": participant_id, "version": 2,
                   "clock_offset": clock_offset, "pupil_recording_dir": recording_dir,
                   "csv_file": csv_filename, "seq_file": npy_file,
                   "recorded_at": time.strftime('%Y-%m-%dT%H:%M:%S'),
                   "note": "pupil_time = csv_timestamp + clock_offset"}, f, indent=2)
    print(f"Metadata saved to:         {meta_filename}")
    print(f"Total trials logged:       {len(trial_log)}")
