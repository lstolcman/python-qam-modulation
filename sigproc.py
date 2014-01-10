# set encoding=utf8

############################################################################
# Signal Processing Module
#
# FEATURES
# - Load/save signal in wav format
# - Manipulate signals in both time and frequency domains
# - Visualize signal in both time and frequency domains
#
# AUTHOR
#
# Chaiporn (Art) Jaikaeo
# Intelligent Wireless Networking Group (IWING) -- http://iwing.cpe.ku.ac.th
# Department of Computer Engineering
# Kasetsart University
# chaiporn.j@ku.ac.th
############################################################################

import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.io import wavfile

plt.rc('font', family='Sawasdee', weight='bold')
#plt.rc('font', family='Garuda')
plt.rc('axes', unicode_minus=False)

###########################################
class Signal(object):

    #######################################
    def __init__(self, duration=1.0, sampling_rate=22050, func=None):
        '''
        Initialize a signal object with the specified duration (in seconds)
        and sampling rate (in Hz).  If func is provided, signal
        data will be initialized to values of this function for the entire
        duration.
        '''
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.freqs = np.arange(int(duration*sampling_rate), dtype=complex)
        self.freqs[:] = 0j
        if func is not None:
            self.sample_time_function(func)

    #######################################
    def read_wav(self, wav_file, channel='left'):
        '''
        Read data from the specified wave file into the signal object.  For a
        stereo stream, only one channel ('left' or 'right') can be extracted.
        '''
        rate,data = wavfile.read(wav_file)
        n = data.shape[0]
        self.sampling_rate = rate
        self.duration = float(n)/rate

        if data.dtype == np.dtype('int16'):
            normalizer = 32768.0
        elif data.dtype == np.dtype('int8'):
            normalizer = 256.0
        else:
            raise(Exception('Unsupport data type'))

        if len(data.shape) == 2: # stereo stream
            if channel == 'left':
                data = data[:,0]
            elif channel == 'right':
                data = data[:,1]
            else:
                raise(Exception('Invalid channel choice "%s"' % channel))

        self.freqs = fft(data/normalizer)

    #######################################
    def write_wav(self, wav_file):
        '''
        Write signal data into the specified wave file using int16 data type
        '''
        wavfile.write(
                wav_file, 
                self.sampling_rate, 
                (ifft(self.freqs).real*32768).astype(np.dtype('int16')))

    #######################################
    def get_sampling_rate(self):
        '''
        Return the sampling rate associated with the signal in Hz
        '''
        return self.sampling_rate

    #######################################
    def get_duration(self):
        '''
        Return the duration of the signal in seconds
        '''
        return self.duration

    #######################################
    def amplify(self, factor):
        '''
        Amplify the signal by the specified factor
        '''
        self.freqs *= factor

    #######################################
    def clear(self, cond=lambda f:True):
        '''
        Set amplitudes of all frequencies satisfying the condition, cond, to
        zero, where cond is a boolean function that takes a frequency in Hz.
        '''
        n = len(self.freqs)
        for i in range(n):
            # convert index to corresponding frequency value
            f = float(i)*self.sampling_rate/n
            if cond(f):
                self.freqs[i] = 0j

    #######################################
    def set_freq(self, freq, amplitude, phase=0):
        '''
        Set a particular frequency component with the specified amplitude and
        phase-shift (in degree) to the signal
        '''
        n = len(self.freqs)

        # compute the index at which the specified frequency is located in the
        # array
        index = np.round(float(freq)*n/self.sampling_rate)

        # distribute the signal amplitude over the real and imaginary axes
        re = float(n)*amplitude*np.cos(phase*np.pi/180.0)
        im = float(n)*amplitude*np.sin(phase*np.pi/180.0)

        # distribute AC component evenly over positive and negative
        # frequencies
        if freq != 0: 
            re = re/2.0
            im = im/2.0

            # to ensure real-valued time-domain signal, the two parts need to
            # be complex conjugate of each other
            self.freqs[ index] = re + 1j*im
            self.freqs[-index] = re - 1j*im

        else:
            # DC component has only one part
            self.freqs[index] = re + 1j*im

    #######################################
    def sample_time_function(self, func):
        '''
        Sample values from a time-domain, real-valued function, func(t), where
        t will be specified in second.  Samples are collected at the
        sampling rate associated with the Signal object.
        '''
        n = len(self.freqs)
        signal = np.arange(n, dtype=float)
        for i in range(n):
            signal[i] = func(float(i)/self.sampling_rate)
        self.freqs = fft(signal)

    ###########################################
    def square_wave(self, freq, flimit=8000):
        '''
        Generate a band-limited square wave on to the signal object
        '''
        self.clear()
        f = freq
        while f <= flimit:
            self.set_freq(f, 1.0/f, -90)
            f += 2*freq

    #######################################
    def get_time_domain(self):
        '''
        Return a tuple (X,Y) where X is an array storing the time axis,
        and Y is an array storing time-domain representation of the signal
        '''
        x_axis = np.linspace(0, self.duration, len(self.freqs))
        y_axis = ifft(self.freqs).real
        return x_axis, y_axis

    #######################################
    def get_freq_domain(self):
        '''
        Return a tuple (X,A,P) where X is an array storing the frequency axis
        up to the Nyquist frequency (excluding negative frequency), and A and
        P are arrays storing the amplitude and phase shift (in degree) of each
        frequency
        '''
        n = len(self.freqs)
        num_freqs = np.ceil((n+1)/2.0)
        x_axis = np.linspace(0, self.sampling_rate/2.0, num_freqs)

        # extract only positive frequencies and scale them so that the
        # magnitude does not depend on the length of the array
        a_axis = abs(self.freqs[:num_freqs])/float(n)
        p_axis = np.arctan2(
                    self.freqs[:num_freqs].imag,
                    self.freqs[:num_freqs].real) * 180.0/np.pi

        # double amplitudes of the AC components (since we have thrown away
        # the negative frequencies)
        a_axis[1:] = a_axis[1:]*2

        return x_axis, a_axis, p_axis

    #######################################
    def shift_freq(self, offset):
        '''
        Shift signal in the frequency domain by the amount specified by offset
        (in Hz).  If offset is positive, the signal is shifted to the right
        along the frequency axis.  If offset is negative, the signal is
        shifted to the left along the frequency axis.
        '''
        n = len(self.freqs)
        nyquist = n/2

        # compute the array-based index from the specified offset in Hz
        offset = int(np.round(float(offset)*n/self.sampling_rate))
        if abs(offset) > nyquist:
            raise Exception(
            'Shifting offset cannot be greater than the Nyquist frequency')

        if offset > 0:
            self.freqs[offset:nyquist] = np.copy(self.freqs[:nyquist-offset])
            self.freqs[:offset] = 0

            self.freqs[-nyquist+1:-offset] = np.copy(self.freqs[-(nyquist-offset)+1:])
            self.freqs[-offset+1:] = 0
        else:
            offset = -offset
            self.freqs[:nyquist-offset] = np.copy(self.freqs[offset:nyquist])
            self.freqs[nyquist-offset:nyquist] = 0

            self.freqs[-(nyquist-offset)+1:] = np.copy(self.freqs[-nyquist+1:-offset])
            self.freqs[-nyquist+1:-nyquist+offset] = 0


    #######################################
    def shift_time(self, offset):
        '''
        Shift signal in the time domain by the amount specified by offset
        (in seconds).  If offset is positive, the signal is shifted to the
        right along the time axis.  If offset is negative, the signal is
        shifted to the left along the time axis.
        '''
        noff = offset*self.sampling_rate
        x,y = self.get_time_domain()
        if noff > 0:
            y[noff:] = y[:len(x)-noff].copy()
            y[:noff] = 0.0
        elif noff < 0:
            noff = -noff
            y[:len(x)-noff] = y[noff:].copy()
            y[len(x)-noff:] = 0.0
        self.freqs = fft(y)

    #######################################
    def copy(self):
        '''
        Clone the signal object into another identical signal object.
        '''
        s = Signal()
        s.duration = self.duration
        s.sampling_rate = self.sampling_rate
        s.freqs = np.array(self.freqs)
        return s

    #######################################
    def mix(self, signal):
        '''
        Mix the signal with another given signal.  Sampling rate and duration
        of both signals must match.
        '''
        if self.sampling_rate != signal.sampling_rate \
           or len(self.freqs) != len(signal.freqs):
            raise Exception(
                'Signal to mix must have identical sampling rate and duration')

        self.freqs += signal.freqs

    #######################################
    def __add__(self, s):
        newSignal = self.copy()
        newSignal.mix(s)
        return newSignal

    #######################################
    def plot(self, dB=False, phase=False, stem=False, frange=(0,10000)):
        '''
        Generate three subplots showing frequency-domain (both amplitude and
        phase) and time-domain representations of the given signal.

        If stem is True, stem plots will be used for both amplitude and phase

        If dB is True, the amplitude in the frequency domain plot will be shown
        with the log scale.

        If phase is True, the phase-shift plot will also be created.
        '''
        plt.subplots_adjust(hspace=.4)

        if phase:
            num_plots = 3
        else:
            num_plots = 2

        # plot time-domain signal
        plt.subplot(num_plots, 1, 1)
        plt.cla()
        x,y = self.get_time_domain()
        plt.grid(True)
        plt.xlabel(u'Time (s)')
        plt.ylabel('Value')
        plt.plot(x,y,'g')

        # plot frequency vs. amplitude
        x,a,p = self.get_freq_domain()
        start_index = int(float(frange[0])/self.sampling_rate*len(self.freqs))
        stop_index  = int(float(frange[1])/self.sampling_rate*len(self.freqs))
        x = x[start_index:stop_index]
        a = a[start_index:stop_index]
        p = p[start_index:stop_index]
        plt.subplot(num_plots, 1, 2)
        plt.cla()
        plt.grid(True)
        plt.xlabel(u'Frequency (Hz)')

        if dB:
            a = 10.*np.log10(a + 1e-10) + 100
            plt.ylabel(u'Amplitude (dB)')
        else:
            plt.ylabel(u'Amplitude')

        if stem:
            plt.stem(x,a,'b')
        else:
            plt.plot(x,a,'b')

        # plot frequency vs. phase-shift
        if phase:
            plt.subplot(num_plots, 1, 3)
            plt.cla()
            plt.grid(True)
            plt.xlabel(u'Frequency (Hz)')
            plt.ylabel(u'Phase (degree)')
            plt.ylim(-180,180)
            if stem:
                plt.stem(x[start_index:stop_index],p[start_index:stop_index],'r')
            else:
                plt.plot(x[start_index:stop_index],p[start_index:stop_index],'r')

        plt.show()


###########################################
def test1():
    '''
    generate a 5Hz square wave with 50Hz cutoff frequency
    then display the time-domain signal
    '''
    s = Signal()
    s.square_wave(5,flimit=50)
    x,y = s.get_time_domain()
    plt.plot(x,y)
    plt.grid(True)
    plt.show()

###########################################
def test2():
    '''
    generate a 2Hz square wave with 50Hz cutoff frequency
    then display both time-domain and frequency-domain signal
    '''
    s = Signal()
    s.square_wave(2,flimit=50)
    s.plot(stem=True,phase=True,frange=(0,50))

###########################################
def test3():
    '''
    generate composite signal containing 3 Hz and 2 Hz sine waves
    '''

    def test_func(t):
        return 0.2*np.sin(2*np.pi*t*3) + 0.3*np.sin(2*np.pi*t*2)

    s = Signal(func=test_func)
    s.plot(frange=(0,10), stem=True)

###########################################
def test4():
    '''
    generate a DTMF (Dual-Tone Multi-Frequency) signal representing keypad '2'
    then write the wave output to a file
    '''
    s = Signal()
    s.set_freq(770, .3, 0)
    s.set_freq(1336, .3, 0)
    s.plot(frange=(0,1500), stem=False)
    s.write_wav('2.wav')

###########################################
def test5():
    '''
    Read a wave file containing keypad '6' DTMF wave form and display its
    signal and frequency spectrum
    '''
    s = Signal()
    s.read_wav('Dtmf6.wav')
    s.plot(frange=(0,2000), stem=False)

###########################################
def test6():
    '''
    Test frequency shifting and mixing of signals
    '''
    s1 = Signal()
    s1.set_freq(50,.3)
    s2 = s1.copy()
    s2.shift_freq(-30)
    s1.mix(s2)
    s2.shift_freq(70)
    s1.mix(s2)
    s1.plot(stem=True, phase=False, frange=(0,100))

###########################################
if __name__ == '__main__':
    test6()
