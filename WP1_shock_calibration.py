"""
Shock calibration — interactive workup
======================================

Workflow: the INTENSITY is set by hand on the stimulator dial. This script
only fires a chosen NUMBER of pulses so you can feel each step.

    press 1  -> one pulse       (adjust the dial, press 1 again, repeat)
    press 3  -> three pulses    (the 'cascade' used in the experiment)
    press 5  -> five pulses     (ceiling)
    press r  -> record a rating / note for the last delivery
    escape   -> stop and print the log

Nothing fires without an explicit keypress, and the script blocks between
deliveries. `num_shock` is a PULSE COUNT, not an intensity — confirm the
dial setting against whatever ceiling your ethics approval specifies.
"""

import time
import numpy as np
import slab
import freefield as ff
from psychopy.hardware import keyboard

# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════

fs        = 48828.125
rcx_file  = 'shock.rcx'
procsser  = 'RM1'

MAX_PULSES   = 5      # hard ceiling; presses above this are REFUSED, not clamped
MIN_INTERVAL = 1.0    # s, lockout so a double-press can't fire back-to-back

VALID_KEYS = [str(n) for n in range(1, MAX_PULSES + 1)] + ['r', 'escape', 't']

def load_tone(tone):
    """
    Write one binaural stimulus into the RM1 play buffer.

    Applies OUTPUT_SCALE equally to both channels — a single global gain, so
    the interaural level difference (the azimuth cue) is preserved exactly.
    Buffers are flattened to 1-D, since tone.left.data is a (n, 1) column.
    """
    left = np.asarray(tone.left.data, dtype=float).flatten()
    right = np.asarray(tone.right.data, dtype=float).flatten()

    ff.write('playbuflen', len(tone), procsser)
    ff.write('data_l', left, procsser)
    ff.write('chan_l', 1, procsser)
    ff.write('data_r', right, procsser)
    ff.write('chan_r', 2, procsser)

TestTone = slab.Sound.tone(frequency=700, duration=100, n_channels=2)


# ════════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════════

slab.set_default_samplerate(fs)

ff.initialize(setup='headphones',
              device=[['RM1', procsser, rcx_file]],
              zbus=False, connection='usb')

kb = keyboard.Keyboard()

print(__doc__)
if input("Participant connected and ready? Type 'yes' to arm: ").strip().lower() != 'yes':
    print("Not armed. Exiting without firing.")
    raise SystemExit

print(f"\nARMED. 1-{MAX_PULSES} = fire that many pulses | r = note | escape = stop\n")

log = []
last_fire = 0.0


# ════════════════════════════════════════════════════════════════════
# LOOP
# ════════════════════════════════════════════════════════════════════

try:
    while True:
        # clear first: buffered presses must never queue up deliveries
        kb.clearEvents()
        keys = kb.waitKeys(keyList=VALID_KEYS, waitRelease=False)

        if not keys:
            continue

        name = keys[0].name

        if name == 'escape':
            print("\nStopped by user.")
            break

        if name == 'r':
            if not log:
                print("  (nothing delivered yet)")
                continue
            note = input("  note/rating for last delivery: ").strip()
            log[-1]['note'] = note
            continue
        if name == 't':
            load_tone(TestTone)
            ff.play(1, [procsser])
            ff.wait_to_finish_playing()
        n_pulses = int(name)

        # refuse rather than silently substitute: if you press 8 and it fires 1,
        # you would write down 8 in your notes and be wrong
        if not 1 <= n_pulses <= MAX_PULSES:
            print(f"  REFUSED: {n_pulses} is outside 1-{MAX_PULSES}. Nothing fired.")
            continue

        since = time.time() - last_fire
        if since < MIN_INTERVAL:
            print(f"  lockout: {MIN_INTERVAL - since:.1f}s to go. Nothing fired.")
            continue

        ff.write('num_shock', n_pulses, procsser)
        ff.play(2, [procsser])
        last_fire = time.time()

        log.append({'time': last_fire, 'n_pulses': n_pulses, 'note': ''})
        print(f"  fired {n_pulses} pulse{'s' if n_pulses > 1 else ''}"
              f"   [#{len(log)}]  (press r to annotate)")

except KeyboardInterrupt:
    print("\nInterrupted.")

finally:
    ff.write('num_shock', 0, procsser)   # leave the buffer in a non-firing state
    print(f"\n=== CALIBRATION LOG ({len(log)} deliveries) ===")
    if log:
        t0 = log[0]['time']
        for i, entry in enumerate(log, 1):
            print(f"  {i:3d}.  t+{entry['time'] - t0:6.1f}s  "
                  f"{entry['n_pulses']} pulse(s)  {entry['note']}")
    print("\nRecord the final dial setting in your notes — it is NOT captured here.")
