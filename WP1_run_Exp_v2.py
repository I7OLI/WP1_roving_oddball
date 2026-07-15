"""
Run WP1 experiment — VERSION 2 (unified array format).

Reads the v2 artefacts produced by WP1_generate_seq_v2.py:
    WP1_sub001_v2_seq.npy    -> (n_trials, 3) int array [freq_step, azi_step, shock]
    WP1_sub001_v2_seq.json   -> metadata (block table, roving lists, timing, CS+)

Instead of one script-branch per stimulus type, this runner iterates the array
ROW BY ROW. For each row it:
    * reads the freq step (col 0) and azimuth step (col 1),
    * advances the matching roving index (clamped to the list bounds),
    * resolves the actual tone frequency (Hz) and azimuth (deg),
    * plays the tone, and
    * if the shock column (col 2) is non-zero, fires that many shock pulses.

A "frequency block" and an "azimuth block" are identical code paths — they only
differ in which column carries the +/- steps (recorded per block in the JSON).

Usage:
    1. Open Pupil Capture manually, adjust cameras, calibrate.
    2. python WP1_run_Exp_v2.py WP1_sub001_v2_seq.npy
       (the matching .json next to it is loaded automatically)
"""
import sys
import os
import json
import time
import csv

import numpy as np
import slab

import freefield as ff
import zmq
import msgpack as serializer

# ============================================================================
# COLUMN LAYOUT (must match WP1_generate_seq_v2.py)
# ============================================================================
COL_FREQ = 0
COL_AZI = 1
COL_SHOCK = 2

# ============================================================================
# FREEFIELD CONFIGURATION
# ============================================================================
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
    pub_socket.connect("tcp://127.0.0.1:{}".format(pub_port))
    time.sleep(0.5)

    print(f"Connected to Pupil Capture (PUB port: {pub_port})")
    return pupil_remote, pub_socket


def measure_clock_offset(pupil_remote, n_samples=20):
    offsets = []
    for _ in range(n_samples):
        local_before = time.time()
        pupil_remote.send_string("t")
        pupil_time = float(pupil_remote.recv_string())
        local_after = time.time()
        local_mid = (local_before + local_after) / 2
        offsets.append(pupil_time - local_mid)
    offset = float(np.median(offsets))
    jitter_ms = float(np.std(offsets) * 1000)
    print(f"Clock offset: {offset:.6f}s  (measurement jitter: ±{jitter_ms:.3f}ms)")
    return offset


def pupil_notify(pupil_remote, notification):
    topic = "notify." + notification["subject"]
    payload = serializer.dumps(notification, use_bin_type=True)
    pupil_remote.send_string(topic, flags=zmq.SNDMORE)
    pupil_remote.send(payload)
    return pupil_remote.recv_string()


def send_annotation(pub_socket, label, clock_offset,
                    local_timestamp=None, duration=0.0, extra=None):
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
    response = pupil_remote.recv_string()
    print(f"Pupil recording started: {response}")
    print(f"Saving eye data to:      {recording_dir}")
    send_annotation(pub_socket, "experiment_start", clock_offset)


def stop_pupil_recording(pupil_remote, pub_socket, clock_offset):
    send_annotation(pub_socket, "experiment_end", clock_offset)
    time.sleep(0.5)
    pupil_remote.send_string('r')
    response = pupil_remote.recv_string()
    print(f"Pupil recording stopped: {response}")


# ============================================================================
# PRECISE TIMING
# ============================================================================

def precise_sleep_until(target_time, busy_wait_threshold=0.002):
    remaining = target_time - time.time()
    if remaining > busy_wait_threshold:
        time.sleep(remaining - busy_wait_threshold)
    while time.time() < target_time:
        pass


# ============================================================================
# ROVING RESOLUTION  (array steps -> concrete Hz / deg per trial)
# ============================================================================

def resolve_block(block_rows, freq_list, pos_list, freq_center, azi_center):
    """
    Walk one block's (n, 3) slab and turn the -1/0/+1 steps into concrete
    stimulus values, with index clamping (roving).

    Returns a list of per-trial dicts:
        freq_hz, azi_deg, freq_step, azi_step, shock, marker ('STD'/'DEV')
    Both freq and azimuth indices reset to their centre at the start of every
    block; the inactive modality simply never moves off its centre because its
    step column is all zeros for this block.
    """
    freq_idx = freq_center
    azi_idx = azi_center
    trials = []
    for row in block_rows:
        f_step = int(row[COL_FREQ])
        a_step = int(row[COL_AZI])
        shock = int(row[COL_SHOCK])

        freq_idx = max(0, min(freq_idx + f_step, len(freq_list) - 1))
        azi_idx = max(0, min(azi_idx + a_step, len(pos_list) - 1))

        step = f_step if f_step != 0 else a_step  # only one is ever non-zero
        trials.append({
            'freq_hz': freq_list[freq_idx],
            'azi_deg': pos_list[azi_idx],
            'freq_step': f_step,
            'azi_step': a_step,
            'step': step,
            'shock': shock,
            'marker': 'DEV' if step != 0 else 'STD',
        })
    return trials


def create_sounds_v2(trials, tone_duration):
    """One binaural tone per trial: pitch = freq_hz, location = azi_deg."""
    tones = []
    for t in trials:
        tone = slab.Sound.tone(frequency=t['freq_hz'], duration=tone_duration, n_channels=2)
        tone = slab.Binaural(tone).at_azimuth(t['azi_deg']).ramp('offset', 0.05)
        tones.append(tone)
    return tones


# ============================================================================
# BLOCK RUNNER
# ============================================================================

def run_block(block_meta, trials, tones, participant_id,
              SOA, tone_duration, trial_log, pub_socket, clock_offset,
              shock_onset=0.25):
    """Play one block row-by-row, annotate Pupil, log every trial."""
    label = block_meta['label']
    cs_plus_modality = block_meta['cs_plus_modality']   # 'f' or 'p'
    cs_plus_value = block_meta['cs_plus_value']
    cs_minus_value = block_meta['cs_minus_value']
    cs_plus_col = COL_FREQ if cs_plus_modality == 'f' else COL_AZI
    block_num = block_meta.get('block_num', 0)

    print(f"\n{'=' * 70}")
    print(f"PLAYING {label}  (CS+={cs_plus_modality}{cs_plus_value:+d})")
    print(f"{'=' * 70}\n")

    send_annotation(pub_socket, f"block_{block_num}_start", clock_offset,
                    extra={"block_label": label, "cs_plus_modality": cs_plus_modality})

    # Prime the buffer with the first tone
    ff.write('playbuflen', len(tones[0]), procsser)
    ff.write('data_l', tones[0].left.data, procsser)
    ff.write('chan_l', 1, procsser)
    ff.write('data_r', tones[0].right.data, procsser)
    ff.write('chan_r', 2, procsser)

    n = len(trials)
    for i in range(n):
        t = trials[i]
        step = t['step']

        # Which dimension carries this trial's deviant? ('f' / 'p' / None)
        if t['freq_step'] != 0:
            dev_modality = 'f'
        elif t['azi_step'] != 0:
            dev_modality = 'p'
        else:
            dev_modality = None

        # CS label. Only the CS+ modality's deviants count as CS+/CS-; a deviant
        # in the OTHER dimension (bf/bp control) is tagged "ODD" (unreinforced).
        if dev_modality is None:
            cs_label = "STD"
        elif dev_modality == cs_plus_modality:
            cs_label = "CS+" if step == cs_plus_value else "CS-"
        else:
            cs_label = "ODD"
        marker = t['marker']
        shock_amount = t['shock']
        shock_delivered = shock_amount > 0

        # --- Play tone, timestamp immediately before the hardware trigger ---
        t_onset = time.time()
        ff.play(1, [procsser])

        send_annotation(pub_socket, f"trial_{marker}_{cs_label}", clock_offset,
                        local_timestamp=t_onset,
                        extra={
                            "block":    block_num,
                            "trial":    i + 1,
                            "modality": dev_modality or "std",
                            "cs_label": cs_label,
                            "marker":   marker,
                            "freq":     t['freq_hz'],
                            "azimuth":  t['azi_deg'],
                            "shock":    int(shock_amount),
                        })

        ff.wait_to_finish_playing()

        print(f"Trial {i + 1:3d}/{n}: {marker} {cs_label:>3s} [{dev_modality or '-'}] | "
              f"freq={t['freq_hz']:7.1f} Hz | azi={t['azi_deg']:+4.0f} deg"
              f"{f' | SHOCK x{shock_amount}' if shock_delivered else ''}")

        # Preload the next tone while we wait out the SOA
        if i + 1 < n:
            nxt = tones[i + 1]
            ff.write('playbuflen', len(nxt), procsser)
            ff.write('data_l', nxt.left.data, procsser)
            ff.write('chan_l', 1, procsser)
            ff.write('data_r', nxt.right.data, procsser)
            ff.write('chan_r', 2, procsser)

        # --- Shock: fire the exact amount encoded in the array (0/1/5) ------
        if shock_delivered:
            precise_sleep_until(t_onset + shock_onset)
            ff.write('num_shock', shock_amount, procsser)
            ff.play(2, [procsser])
            send_annotation(pub_socket, "shock", clock_offset,
                            local_timestamp=time.time(),
                            extra={"block": block_num, "trial": i + 1,
                                   "num_shock": int(shock_amount)})

        precise_sleep_until(t_onset + SOA)

        trial_log.append({
            'participant_id':  participant_id,
            'block':           block_num,
            'block_label':     label,
            'trial_num':       i + 1,
            'dev_modality':    dev_modality or '',
            'freq_step':       t['freq_step'],
            'azi_step':        t['azi_step'],
            'sequence_value':  step,
            'trial_type':      cs_label,
            'freq_hz':         t['freq_hz'],
            'azi_deg':         t['azi_deg'],
            'is_cs_plus':      (dev_modality == cs_plus_modality and step == cs_plus_value),
            'shock_amount':    shock_amount,
            'shock_delivered': shock_delivered,
            'timestamp':       t_onset,
        })

    send_annotation(pub_socket, f"block_{block_num}_end", clock_offset,
                    extra={"block_label": label})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # --- Resolve the .npy + .json artefacts ---------------------------------
    if len(sys.argv) == 2:
        npy_file = sys.argv[1]
    else:
        print("=== WP1 Experiment Runner (v2) ===")
        participant_id_input = int(input("Enter participant number: "))
        print("Session types:  f = freq,  p = azimuth,  bf = both (freq CS+),  bp = both (azimuth CS+)")
        exp_type_input = input("Enter session type (f / p / bf / bp): ").strip().lower()
        while exp_type_input not in ('f', 'p', 'bf', 'bp'):
            exp_type_input = input("  Invalid — enter f, p, bf, or bp: ").strip().lower()
        npy_file = os.path.join('sequences',
                                f"WP1_sub{participant_id_input:03d}_{exp_type_input}_v2_seq.npy")
        if not os.path.exists(npy_file):
            print(f"\nERROR: Could not find '{npy_file}'")
            print("Run WP1_generate_seq_v2.py first to create the sequence.")
            sys.exit(1)
        print(f"Found sequence file: {npy_file}\n")

    json_file = os.path.splitext(npy_file)[0] + '.json'
    if not os.path.exists(json_file):
        print(f"\nERROR: metadata file '{json_file}' not found next to the .npy")
        sys.exit(1)

    seq_array = np.load(npy_file)
    with open(json_file) as f:
        meta = json.load(f)

    participant_id     = meta['participant_id']
    exp_type           = meta.get('experiment_type', 'v2')
    freq_list          = meta['frequency_list']
    pos_list           = meta['position_list']
    freq_center        = meta['freq_center_index']
    azi_center         = meta['azi_center_index']
    ITI                = meta['ITI']
    tone_duration      = meta['tone_duration']
    SOA                = meta.get('SOA', ITI + tone_duration)
    shock_onset        = meta.get('shock_onset_in_iti', 0.25)
    blocks             = meta['blocks']

    print(f"\n{'=' * 70}")
    print(f"WP1 EXPERIMENT (v2)")
    print(f"{'=' * 70}")
    print(f"Sequence file:   {npy_file}")
    print(f"Participant ID:  {participant_id}")
    print(f"Session type:    {exp_type}  (CS+ modality = {meta.get('cs_plus_modality')})")
    print(f"Array shape:     {seq_array.shape}")
    print(f"Blocks:          {len(blocks)}")
    print(f"CS+ assignment:  {meta.get('cs_plus_assignment')}")
    print(f"Seed:            {meta.get('random_seed', 'unknown')}")
    print(f"{'=' * 70}\n")

    # --- Recording folder ---------------------------------------------------
    recording_dir = os.path.join(
        r"C:\Users\neurobio\Projects\WP1_roving_oddball\recordings",
        f"sub{participant_id:03d}_{exp_type}_v2"
    )

    # --- Connect to Pupil Capture (must already be open) --------------------
    pupil_remote, pub_socket = connect_to_pupil()
    clock_offset = measure_clock_offset(pupil_remote)
    start_pupil_recording(pupil_remote, pub_socket, clock_offset, recording_dir)

    # --- Run each block -----------------------------------------------------
    trial_log = []
    for b_i, block_meta in enumerate(blocks, start=1):
        block_meta = dict(block_meta)
        block_meta['block_num'] = b_i

        block_rows = seq_array[block_meta['row_start']:block_meta['row_end']]

        # Resolve roving steps -> concrete Hz/deg, then build the tones.
        trials = resolve_block(block_rows, freq_list, pos_list, freq_center, azi_center)
        tones = create_sounds_v2(trials, tone_duration)

        n_devs = sum(1 for t in trials if t['step'] != 0)
        print(f"\n>>> PREPARING {block_meta['label']} <<<")
        print(f"    {len(trials)} trials, {n_devs} deviants")

        print(f"\nPress Enter to start {block_meta['label']}...")
        input()

        run_block(
            block_meta=block_meta,
            trials=trials,
            tones=tones,
            participant_id=participant_id,
            SOA=SOA,
            tone_duration=tone_duration,
            trial_log=trial_log,
            pub_socket=pub_socket,
            clock_offset=clock_offset,
            shock_onset=shock_onset,
        )

        if b_i < len(blocks):
            print(f"\n{'=' * 70}")
            print(f"{block_meta['label']} COMPLETE. Press Enter to continue...")
            print(f"{'=' * 70}")
            input()

    # --- Stop recording -----------------------------------------------------
    stop_pupil_recording(pupil_remote, pub_socket, clock_offset)

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")

    # --- Save behavioural CSV ----------------------------------------------
    csv_filename = f"WP1_sub{participant_id:03d}_{exp_type}_v2_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trial_log[0].keys())
        writer.writeheader()
        writer.writerows(trial_log)
    print(f"Behavioural data saved to: {csv_filename}")

    # --- Save metadata sidecar ---------------------------------------------
    meta_filename = csv_filename.replace('.csv', '_meta.json')
    meta_out = {
        "participant_id":      participant_id,
        "version":             2,
        "clock_offset":        clock_offset,
        "pupil_recording_dir": recording_dir,
        "csv_file":            csv_filename,
        "seq_file":            npy_file,
        "recorded_at":         time.strftime('%Y-%m-%dT%H:%M:%S'),
        "note": "pupil_time = csv_timestamp + clock_offset",
    }
    with open(meta_filename, 'w') as f:
        json.dump(meta_out, f, indent=2)
    print(f"Metadata saved to:         {meta_filename}")
    print(f"Total trials logged:       {len(trial_log)}")
