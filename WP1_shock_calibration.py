import slab
import freefield as ff

fs = 48828.125
slab.set_default_samplerate(fs)
rcx_file = 'shock.rcx'
procsser = 'RM1'

ff.initialize(setup='headphones',
              device=[['RM1', procsser, rcx_file]],
              zbus=False, connection='usb')



shock_high = 4
ff.write('num_shock', shock_high, procsser)
ff.play(2, [procsser])