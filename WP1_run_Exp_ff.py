"""
Run WP1 experiment from a pre-generated JSON sequence file,
with Pupil Labs eye-tracker recording and precise trial annotations.

Usage:
    1. Open Pupil Capture manually, adjust cameras, calibrate.
    2. python WP1_run_Exp_ff_pupil.py WP1_sub001_f_seq.json
"""
import sys
import os
import json
import time
import csv
import numpy as np
import slab
from slab.experiments.room_voice_interference import condition

import freefield as ff
#  import zmq
import msgpack as serializer

# ============================================================================
# FREEFIELD CONFIGURATION
# ============================================================================
fs = 48828.125
slab.set_default_samplerate(fs)
rcx_file = 'shock.rcx'
procsser = 'RM1'
condition_shock =1

ff.initialize(
    setup='headphones',
    device=[['RM1', procsser, rcx_file]],
    zbus=False,
    connection='usb'
)

# ============================================================================
# PUPIL LABS SETUP
# ============================================================================

def connect_to_pupil():
    """Connect to Pupil Remote and return (pupil_remote, pub_socket)."""
    ctx = zmq.Context()
    pupil_remote = zmq.Socket(ctx, zmq.REQ)
    pupil_remote.connect('tcp://127.0.0.1:50020')

    pupil_remote.send_string("PUB_PORT")
    pub_port = pupil_remote.recv_string()
    pub_socket = zmq.Socket(ctx, zmq.PUB)
    pub_socket.connect("tcp://127.0.0.1:{}".format(pub_port))
    time.sleep(0.5)  # allow PUB socket to connect

    print(f"Connected to Pupil Capture (PUB port: {pub_port})")
    return pupil_remote, pub_socket


def measure_clock_offset(pupil_remote, n_samples=20):
    """
    Measure offset between local time.time() and Pupil's clock once at startup.
    Uses the midpoint of the round-trip to estimate when Pupil sampled its clock.
    Returns offset such that:  pupil_time = local_time + offset
    """
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
    """Send a notification to Pupil Remote."""
    topic = "notify." + notification["subject"]
    payload = serializer.dumps(notification, use_bin_type=True)
    pupil_remote.send_string(topic, flags=zmq.SNDMORE)
    pupil_remote.send(payload)
    return pupil_remote.recv_string()


def send_annotation(pub_socket, label, clock_offset,
                    local_timestamp=None, duration=0.0, extra=None):
    """
    Send an annotation to Pupil Capture.
    local_timestamp is converted to Pupil time using the pre-measured clock
    offset — no network round-trip, sub-millisecond precision.
    """
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
    """Enable plugins, set save path, and start recording."""
    os.makedirs(recording_dir, exist_ok=True)

    # Enable annotation plugin
    pupil_notify(pupil_remote, {
        "subject": "start_plugin",
        "name": "Annotation_Capture",
        "args": {}
    })
    # Enable recorder and point it at our folder
    pupil_notify(pupil_remote, {
        "subject": "start_plugin",
        "name": "Recorder",
        "args": {"rec_root_dir": recording_dir}
    })
    time.sleep(0.5)  # give plugins time to load

    pupil_remote.send_string('R')
    response = pupil_remote.recv_string()
    print(f"Pupil recording started: {response}")
    print(f"Saving eye data to:      {recording_dir}")
    send_annotation(pub_socket, "experiment_start", clock_offset)


def stop_pupil_recording(pupil_remote, pub_socket, clock_offset):
    """Send end annotation and stop recording."""
    send_annotation(pub_socket, "experiment_end", clock_offset)
    time.sleep(0.5)  # ensure last annotation is written before stopping
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
# SOUND CREATION
# ============================================================================

def create_sounds(trials, experiment_type, tone_duration, iti_within_pattern=0.05):
    if experiment_type in ['f', 'p']:
        tones, values, indices = [], [], []
        for t in trials:
            if experiment_type == 'f':
                tone = slab.Sound.tone(frequency=t['value'], duration=tone_duration)
                tone = slab.Binaural(tone).ramp('offset', 0.05)
            else:
                tone = slab.Sound.tone(frequency=700, duration=tone_duration, n_channels=2)
                tone = slab.Binaural(tone).at_azimuth(t['value']).ramp('offset', 0.05)
            tones.append(tone)
            values.append(t['value'])
            indices.append(t['index'])
        return tones, values, indices

    elif experiment_type == 'a':
        patterns, pattern_info = [], []
        silence_samples = int(iti_within_pattern * slab.get_default_samplerate())
        for t in trials:
            pattern_tones = [
                slab.Sound.tone(frequency=f, duration=tone_duration / 2).ramp('offset', 0.05)
                for f in t['frequencies']
            ]
            n_ch = pattern_tones[0].data.shape[1]
            parts = []
            for idx, tone in enumerate(pattern_tones):
                parts.append(tone.data)
                if idx < len(pattern_tones) - 1:
                    parts.append(np.zeros((silence_samples, n_ch)))
            combined = slab.Sound(np.vstack(parts),
                                  samplerate=slab.get_default_samplerate())
            patterns.append(combined)
            pattern_info.append(t)
        return patterns, pattern_info


# ============================================================================
# BLOCK RUNNER
# ============================================================================

def run_block(sequence, stimuli, experiment_type, block_num, block_label,
              participant_id, cs_plus_value, ITI, SOA, tone_duration, trial_log,
              pub_socket, clock_offset,
              reinforcement=None, shock_onset=0.25, max_cumsum=4,
              A_SOA=0.65, iti_within_pattern=0.1):
    """Play one block, send precisely-timed Pupil annotations, log all trials."""
    print(f"\n{'=' * 70}")
    print(f"PLAYING BLOCK {block_num}: {block_label}")
    print(f"{'=' * 70}\n")

    send_annotation(pub_socket, f"block_{block_num}_start", clock_offset,
                    extra={"block_label": block_label})

    cs_minus_value = -cs_plus_value

    if experiment_type in ['f', 'p']:
        tones, values, indices = stimuli
        value_label = "freq" if experiment_type == 'f' else "azimuth"
        value_unit  = "Hz"  if experiment_type == 'f' else "deg"
        ff.write('playbuflen', len(tones[0]), procsser)
        ff.write('data_l', tones[0].left.data, procsser)
        ff.write('chan_l', 1, procsser)
        ff.write('data_r', tones[0].right.data, procsser)
        ff.write('chan_r', 2, procsser)
    elif experiment_type == 'a':
        patterns, pattern_info = stimuli
        ff.write('playbuflen', len(patterns[0]), procsser)
        ff.write('data_l', patterns[0].data, procsser)
        ff.write('chan_l', 1, procsser)
        ff.write('data_r', patterns[0].data, procsser)
        ff.write('chan_r', 2, procsser)

    for i in range(len(sequence)):

        if sequence[i] == cs_plus_value:
            cs_label = "CS+"
        elif sequence[i] == cs_minus_value:
            cs_label = "CS-"
        else:
            cs_label = "STD"
        marker = "DEV" if sequence[i] != 0 else "STD"
        shock_delivered = reinforcement is not None and reinforcement[i]

        if experiment_type in ['f', 'p']:

            # Capture timestamp immediately before hardware trigger
            t_onset = time.time()
            ff.play(1, [procsser])

            # Annotation uses t_onset directly — zero network delay
            send_annotation(pub_socket, f"trial_{marker}_{cs_label}", clock_offset,
                            local_timestamp=t_onset,
                            extra={
                                "block":     block_num,
                                "trial":     i + 1,
                                "cs_label":  cs_label,
                                "marker":    marker,
                                value_label: values[i],
                                "shock":     int(shock_delivered)
                            })

            ff.wait_to_finish_playing()
            stimulus_value = values[i]

            print(f"Tone {i + 1:3d}/{len(sequence)}: {marker} {cs_label:>3s} | "
                  f"{value_label}={values[i]:7.1f} {value_unit} | "
                  f"index={indices[i] - max_cumsum}"
                  f"{' | SHOCK' if shock_delivered else ''}")

            if i + 1 < len(tones):
                ff.write('playbuflen', len(tones[i + 1]), procsser)
                ff.write('data_l', tones[i + 1].left.data, procsser)
                ff.write('chan_l', 1, procsser)
                ff.write('data_r', tones[i + 1].right.data, procsser)
                ff.write('chan_r', 2, procsser)

            if shock_delivered:
                precise_sleep_until(t_onset + shock_onset)
                ff.write('num_shock',condition_shock)
                ff.play(2, [procsser])
                send_annotation(pub_socket, "shock", clock_offset,
                                local_timestamp=time.time(),
                                extra={"block": block_num, "trial": i + 1})

            precise_sleep_until(t_onset + SOA)

        elif experiment_type == 'a':
            info = pattern_info[i]
            freqs_str = '-'.join([f"{f:.0f}" for f in info['frequencies']])
            stimulus_value = info['base_freq']

            t_onset = time.time()
            ff.play(1, [procsser])

            send_annotation(pub_socket, f"trial_{marker}_{cs_label}", clock_offset,
                            local_timestamp=t_onset,
                            extra={
                                "block":     block_num,
                                "trial":     i + 1,
                                "cs_label":  cs_label,
                                "marker":    marker,
                                "base_freq": info['base_freq'],
                                "pattern":   info['pattern_name'],
                                "shock":     int(shock_delivered)
                            })

            print(f"Trial {i + 1:3d}/{len(sequence)}: {marker} {cs_label:>3s} | "
                  f"{info['pattern_name']:>8s} | base={info['base_freq']:.0f}Hz "
                  f"tones={freqs_str}{' | SHOCK' if shock_delivered else ''}")

            ff.wait_to_finish_playing()

            if shock_delivered:
                precise_sleep_until(t_onset + shock_onset + iti_within_pattern)
                ff.write('num_shock',condition_shock)
                ff.play(2, [procsser])
                send_annotation(pub_socket, "shock", clock_offset,
                                local_timestamp=time.time(),
                                extra={"block": block_num, "trial": i + 1})

            if i + 1 < len(patterns):
                ff.write('playbuflen', len(patterns[i + 1]), procsser)
                ff.write('data_l', patterns[i + 1].data, procsser)
                ff.write('chan_l', 1, procsser)
                ff.write('data_r', patterns[i + 1].data, procsser)
                ff.write('chan_r', 2, procsser)

            precise_sleep_until(t_onset + A_SOA)

        trial_log.append({
            'participant_id':  participant_id,
            'block':           block_num,
            'block_label':     block_label,
            'trial_num':       i + 1,
            'experiment_type': experiment_type,
            'sequence_value':  sequence[i],
            'trial_type':      cs_label,
            'stimulus_value':  stimulus_value,
            'is_cs_plus':      sequence[i] == cs_plus_value,
            'shock_delivered': shock_delivered,
            'timestamp':       t_onset
        })

    send_annotation(pub_socket, f"block_{block_num}_end", clock_offset,
                    extra={"block_label": block_label})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) == 2:
        seq_file = sys.argv[1]
    else:
        print("=== WP1 Experiment Runner ===")
        participant_id_input = int(input("Enter participant number: "))
        print("Experiment types:  f = frequency,  p = position,  a = abstract")
        exp_type_input = input("Enter experiment type (f / p / a): ").strip().lower()
        while exp_type_input not in ['f', 'p', 'a']:
            exp_type_input = input("  Invalid — please enter f, p, or a: ").strip().lower()

        seq_file = os.path.join('sequences', f"WP1_sub{participant_id_input:03d}_{exp_type_input}_seq.json")
        if not os.path.exists(seq_file):
            print(f"\nERROR: Could not find '{seq_file}'")
            print("Please run WP1_generate_seq.py first to create the sequence.")
            sys.exit(1)
        print(f"Found sequence file: {seq_file}\n")

    with open(seq_file) as f:
        data = json.load(f)

    meta               = data['metadata']
    participant_id     = meta['participant_id']
    experiment_type    = meta['experiment_type']
    cs_plus_value      = meta['cs_plus_value']
    ITI                = meta['ITI']
    tone_duration      = meta['tone_duration']
    iti_within_pattern = meta.get('iti_within_pattern', 0.05)
    shock_onset        = meta.get('shock_onset_in_iti', 0.25)
    max_cumsum         = meta.get('max_cumsum', 4)

    print(f"\n{'=' * 70}")
    print(f"WP1 EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"Sequence file:       {seq_file}")
    print(f"Participant ID:      {participant_id}")
    print(f"Experiment type:     {experiment_type}")
    print(f"CS+ deviant:         {cs_plus_value:+d}")
    print(f"CS- deviant:         {meta['cs_minus_value']:+d}")
    print(f"Blocks:              {len(data['blocks'])}")
    print(f"Seed:                {meta.get('random_seed', 'unknown')}")
    print(f"Generated at:        {meta.get('generated_at', 'unknown')}")
    print(f"{'=' * 70}\n")

    # ── Recording folder: one subfolder per participant + type ───────────────
    recording_dir = os.path.join(
        r"C:\Users\neurobio\Projects\WP1_roving_oddball\recordings",
        f"sub{participant_id:03d}_{experiment_type}"
    )

    # ── Connect to Pupil Capture (must already be open) ──────────────────────

    pupil_remote, pub_socket = connect_to_pupil()
    clock_offset = measure_clock_offset(pupil_remote)
    start_pupil_recording(pupil_remote, pub_socket, clock_offset, recording_dir)

    # ── Run experiment blocks ────────────────────────────────────────────────
    trial_log = []

    for block_data in data['blocks']:
        block_num     = block_data['block_num']
        label         = block_data['label']
        seq           = block_data['sequence']
        reinforcement = block_data['reinforcement']

        n_devs = sum(1 for s in seq if s != 0)
        print(f"\n>>> PREPARING BLOCK {block_num}: {label} <<<")
        print(f"    {len(seq)} trials, {n_devs} deviants")

        stimuli = create_sounds(block_data['trials'], experiment_type,
                                tone_duration, iti_within_pattern)

        print(f"\nPress Enter to start BLOCK {block_num} ({label})...")
        input()

        run_block(
            sequence=seq,
            stimuli=stimuli,
            experiment_type=experiment_type,
            block_num=block_num,
            block_label=label,
            participant_id=participant_id,
            cs_plus_value=cs_plus_value,
            ITI=ITI,
            SOA=ITI + tone_duration,
            tone_duration=tone_duration,
            trial_log=trial_log,
            pub_socket=pub_socket,
            clock_offset=clock_offset,
            reinforcement=reinforcement,
            shock_onset=shock_onset,
            max_cumsum=max_cumsum,
            iti_within_pattern=iti_within_pattern
        )

        if block_num < len(data['blocks']):
            print(f"\n{'=' * 70}")
            print(f"BLOCK {block_num} COMPLETE. Press Enter to continue...")
            print(f"{'=' * 70}")
            input()

    # ── Stop Pupil recording ─────────────────────────────────────────────────
    stop_pupil_recording(pupil_remote, pub_socket, clock_offset)

    # ── Save behavioural CSV ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")

    csv_filename = f"WP1_sub{participant_id:03d}_{experiment_type}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trial_log[0].keys())
        writer.writeheader()
        writer.writerows(trial_log)
    print(f"Behavioural data saved to: {csv_filename}")

    # ── Save metadata (clock offset + paths) for later analysis ─────────────
    meta_filename = csv_filename.replace('.csv', '_meta.json')
    meta_out = {
        "participant_id":    participant_id,
        "experiment_type":   experiment_type,
        "clock_offset":      clock_offset,
        "pupil_recording_dir": recording_dir,
        "csv_file":          csv_filename,
        "recorded_at":       time.strftime('%Y-%m-%dT%H:%M:%S'),
        "note": (
            "To align CSV timestamps with Pupil data: "
            "pupil_time = csv_timestamp + clock_offset"
        )
    }
    with open(meta_filename, 'w') as f:
        json.dump(meta_out, f, indent=2)
    print(f"Metadata saved to:         {meta_filename}")
    print(f"Total trials logged:       {len(trial_log)}")