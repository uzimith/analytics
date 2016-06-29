import numpy as np
import scipy.signal
from itertools import chain

def separate(erps, channel_num=8):
    frame_length = len(erps[0]) / channel_num
    erps_of_channels = [np.squeeze(np.reshape(erp, (channel_num, frame_length))) for erp in erps]

    return erps_of_channels

def combine(erp):
    return erp.flatten()

def decimate(erps, factor, channel_num=8):
    frame_length = len(erps[0]) / channel_num
    erps_of_channels = separate(erps)
    erps_of_channels = [scipy.signal.decimate(erp, factor) for erp in erps_of_channels]
    erps = [list(chain.from_iterable(erp)) for erp in erps_of_channels]
    return erps
