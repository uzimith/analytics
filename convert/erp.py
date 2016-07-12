import numpy as np
import scipy.signal
from itertools import chain
import scipy.spatial.distance as dis
import random

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

def undersampling(erps, block_num, method="cosine", far=30):
    if method == "euclidean":
        target_erp = np.average(erps[1], axis=0)
        erps[0].sort(key=(lambda erp, target_erp=target_erp: np.linalg.norm(target_erp - erp)) )
        erps[0] = erps[0][far:far+block_num]
    if method == "cosine":
        non_target_erp = np.average(erps[0], axis=0).flatten()
        erps[0] = sorted(erps[0], key=(lambda erp, non_target_erp=non_target_erp: dis.cosine(non_target_erp, erp.flatten())) )
        erps[0] = erps[0][far:far+block_num]
    if method == "cosine_more":
        # unuseful
        target_erp = np.average(erps[1], axis=0).flatten()
        non_target_erp = np.average(erps[0], axis=0).flatten()
        erps[0].sort(key=(lambda erp, non_target_erp=non_target_erp: dis.cosine(non_target_erp, erp.flatten())) )
        erps[0] = erps[0][far:far+block_num-10]
        erps[1].sort(key=(lambda erp, target_erp=non_target_erp: dis.cosine(target_erp, erp.flatten())) )
        erps[1] = erps[1][far:far+block_num-10]
    if method == "random":
        erps[0] = random.sample(erps[0], block_num)
    return erps

# def highpass(numtaps=255, cutoff=30):
#     b = scipy.signal.firwin(numtaps, fe)
#     frame_length = len(erps[0]) / channel_num
#     erps_of_channels = separate(erps)
#     erps_of_channels = [scipy.signal.lfilter(b, 1, erp) for erp in erps_of_channels]
#     erps = [list(chain.from_iterable(erp)) for erp in erps_of_channels]
#     return erps
#
# def lowpass(numtaps=255, cutoff=0.1, frequency=256):
#     b = scipy.signal.firwin(numtaps, cutoff / (frequency), pass_zero=False)
#     frame_length = len(erps[0]) / channel_num
#     erps_of_channels = separate(erps)
#     erps_of_channels = [scipy.signal.lfilter(b, 1, erp) for erp in erps_of_channels]
#     erps = [list(chain.from_iterable(erp)) for erp in erps_of_channels]
#     return erps
