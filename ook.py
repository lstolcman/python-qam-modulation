import sys
from qam import Qam
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
	print "Usage: %s <data-bits>" % sys.argv[0]
	exit(1)

modulation = { 
    '0' : (0,0),
    '1' : (1,0),
    }
q = Qam(baud_rate = 10,
        bits_per_baud = 1,
        carrier_freq = 50,
        modulation = modulation)

s = q.generate_signal(sys.argv[1])

plt.figure(1)
q.plot_constellation()
plt.figure(2)
s.plot(dB=False, phase=False, stem=False, frange=(0,500))
