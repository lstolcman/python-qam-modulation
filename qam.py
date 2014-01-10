import numpy as np
from matplotlib import pyplot as plt
from sigproc import Signal

#################################
class Qam:

    #################################
    def __init__(self, 
            modulation = {'0':(0,0), '1':(1,0)},
            baud_rate = 10,
            bits_per_baud = 1,
            carrier_freq = 100):
        '''
        Create a modulator using OOK by default
        '''
        self.modulation    = modulation
        self.baud_rate     = baud_rate
        self.bits_per_baud = bits_per_baud
        self.carrier_freq  = carrier_freq

    #################################
    def generate_signal(self, data):
        '''
        Generate signal corresponding to the current modulation scheme to
        represent given binary string, data.
        '''

        def create_func(data):
            slot_data = []
            for i in range(0,len(data),self.bits_per_baud):
                slot_data.append(self.modulation[data[i:i+self.bits_per_baud]])

            def timefunc(t):
                slot = int(t*self.baud_rate)
                start = float(slot)/self.baud_rate
                offset = t - start
                amplitude,phase = slot_data[slot]
                return amplitude*np.sin(2*np.pi*self.carrier_freq*offset +
                        phase/180.0*np.pi)

            return timefunc

        func = create_func(data)
        duration = float(len(data))/(self.baud_rate*self.bits_per_baud)
        s = Signal(duration=duration, func=func)
        return s

    #################################
    def plot_constellation(self):
        '''
        Plot a constellation diagram representing the modulation scheme.
        '''
        data = [(a*np.cos(p/180.0*np.pi), a*np.sin(p/180.0*np.pi), t) 
                for t,(a,p) in self.modulation.items()]
        sx,sy,t = zip(*data)
        plt.clf()
        plt.scatter(sx,sy,s=30)
        plt.axes().set_aspect('equal')
        for x,y,t in data:
            plt.annotate(t,(x-.03,y-.03), ha='right', va='top')
        plt.axis([-1.5,1.5,-1.5,1.5])
        plt.axhline(0, color='red')
        plt.axvline(0, color='red')
        plt.grid(True)
    
