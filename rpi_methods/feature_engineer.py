# -*- coding: utf-8 -*-

# import pandas as pd
import numpy as np
#from librosa.feature import zero_crossing_rate, mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth, rmse
from pyAudioAnalysis import audioFeatureExtraction as af
import python_speech_features as psf


# chroma_cens, rmse

__all__ = [
    'FeatureEngineer'
]


class FeatureEngineer:
    """
    Derive features
    """

    RATE = 44100   # All recordings in ESC are 44.1 kHz
    FRAME = 512    # Frame size in samples
    nFFT = RATE/2
    # Features' names
    COL = ['zcr', 'rms_energy',
           'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11',
           'mfcc12',
           'sp_centroid', 'sp_rolloff', 'sp_bw'
           # 'chroma1', 'chroma2', 'chroma3', 'chroma4', 'chroma5', 'chroma6', 'chroma7',
           # 'chroma8', 'chroma9', 'chroma10', 'chroma11', 'chroma12'
           ]

    def __init__(self):
        pass

    def feature_engineer(self, audio_data):
        """
        Extract features using librosa.feature.

        Each signal is cut into frames, features are computed for each frame and averaged [median].
        The numpy array is transformed into a data frame with named columns.

        :param audio_data: the input signal samples with frequency 44.1 kHz
        :return: a numpy array (numOfFeatures x numOfShortTermWindows)
        """
        loop_length = len(audio_data)/self.FRAME

        concat_feat = []

        zcr_feat = []
        rmse_feat = []
        spectral_bandwidth_feat = []
        spectral_centroid_feat = []
        spectral_rolloff_feat = []
        mfcc_feat = np.empty(shape=[13, 0])

        for i in range(loop_length):

            audio_data_batch = (audio_data[i*loop_length:(i*loop_length)+loop_length])

            zcr_feat_1 = af.stZCR(audio_data_batch)
            zcr_feat.append(zcr_feat_1)

            rmse_feat_1 = af.stEnergy(audio_data_batch)
            rmse_feat.append(rmse_feat_1)

            if rmse_feat_1.shape == (1, 427):
                rmse_feat_1 = np.concatenate((rmse_feat, np.zeros((1, 4))), axis=1)

            [fbank, freqs] = af.mfccInitFilterBanks(self.RATE, self.nFFT)
            #mfcc_feat = af.stMFCC(audio_data, fbank, 13)

            mfcc_feat_1 = psf.mfcc(audio_data_batch, self.RATE, nfft=1103)
            # mfcc_feat_1 = np.squeeze(mfcc_feat_1).shape
            mfcc_feat_1 = np.transpose(mfcc_feat_1)
            mfcc_feat = np.append(mfcc_feat, mfcc_feat_1, axis=1)
            spectral_centroid_and_spread_1 = af.stSpectralCentroidAndSpread(audio_data_batch, self.RATE)
            spectral_centroid_feat_1 = spectral_centroid_and_spread_1[0]
            spectral_centroid_feat.append(spectral_centroid_feat_1)

            spectral_bandwidth_feat_1 = spectral_centroid_and_spread_1[1]
            spectral_bandwidth_feat.append(spectral_bandwidth_feat_1)

            spectral_rolloff_feat_1 = af.stSpectralRollOff(audio_data_batch, 0.90, self.RATE)
            spectral_rolloff_feat.append(spectral_rolloff_feat_1)

            # chroma_cens_feat = chroma_cens(y=audio_data, sr=self.RATE, hop_length=self.FRAME)

        # zcr_feat = np.asarray(zcr_feat)
        # rmse_feat = np.asarray(rmse_feat)
        # spectral_bandwidth_feat = np.asarray(spectral_bandwidth_feat)
        # spectral_centroid_feat = np.asarray(spectral_centroid_feat)
        # spectral_rolloff_feat = np.asarray(spectral_rolloff_feat)

        concat_feat.append(zcr_feat)
        concat_feat.append(rmse_feat)
        concat_feat.append(spectral_bandwidth_feat)
        concat_feat.append(spectral_centroid_feat)
        concat_feat.append(spectral_rolloff_feat)
        # concat_feat.append(mfcc_feat)
        # mfcc_feat = np.asarray(mfcc_feat, dtype=np.float32)
        concat_feat = np.array(concat_feat)
        concat_feat = np.concatenate((concat_feat, mfcc_feat), axis=0)
        # print concat_feat.shape
        return np.mean(concat_feat, axis=1, keepdims=True).transpose()
