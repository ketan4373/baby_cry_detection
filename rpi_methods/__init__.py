# -*- coding: utf-8 -*-

#import librosa
# import re
# import pydub
# import os
import soundfile

__all__ = [
    'Reader'
]


class Reader:
    """
    Read input audio file
    file_name: 'path/to/file/filename.mp3'
    """

    def __init__(self, file_name):
        self.file_name = file_name
        pass

    def read_audio_file(self):
        """
        Read audio file using librosa package. librosa allows resampling to desired sample rate and convertion to mono.

        :return:
        * play_list: a list of audio_data as numpy.ndarray. There are 5 overlapping signals, each one is 5-second long.
        """
        #
        # play_list = list()
        # info = soundfile._SoundFileInfo(self.file_name, 1)
        # frames =  info.frames
        # for offset in range(5):
        #     # audio_data, _ = librosa.load(self.file_name, sr=44100, mono=True, offset=offset, duration=5.0)
        #     # play_list.append(audio_data)
        #     start = offset * (frames/5)
        #     stop = (offset + 1) * (frames/5)
        #     data = soundfile.read(self.file_name, start = start, stop = stop)
        #     audio_data = data[0]
        #     for i in range(len(audio_data)):
        #         audio_data[i] = (audio_data[i] / 4.0) + 1
        #
        #     play_list.append(audio_data)


            # audio_data, _ = librosa.load(self.file_name, sr=44100, mono=True, offset=offset, duration=5.0)
            # play_list.append(audio_data)

        play_list = list()
        data = soundfile.read(self.file_name)
        audio_data = data[0]
        for i in range(len(audio_data)):
            audio_data[i] = (audio_data[i] / 4.0) + 1

        play_list.append(audio_data)

        return play_list

    #     # Librosa doesn't work for wav files on python 2: https://github.com/librosa/librosa/issues/390
    #     # return librosa.load(self.file_name, sr=44100, mono=True, duration=5.0)
    #
    #     play_list = list()
    #
    #     if self._is_mp3_or_ogg(self.file_name):
    #
    #         # Split 9 sec signal into 5 5-second-long overlapping signals
    #         for offset in range(0, 5):
    #             audio_data, _ = librosa.load(self.file_name, sr=44100, mono=True, offset=offset, duration=5.0)
    #             play_list.append(audio_data)
    #
    #         return play_list
    #
    #     else:
    #         wav_file_pydub = pydub.AudioSegment.from_file(self.file_name)
    #
    #         ogg_file_name = self._ogg_file_name(self.file_name)
    #
    #         with wav_file_pydub.export(ogg_file_name, format='ogg', codec='libvorbis', bitrate='192k') as wav_file:
    #             wav_file.close()
    #
    #         for offset in range(0, 5):
    #             audio_data, _ = librosa.load(ogg_file_name, sr=44100, mono=True, offset=offset, duration=5.0)
    #             play_list.append(audio_data)
    #
    #         return play_list
    #
    # @staticmethod
    # def _is_mp3_or_ogg(string):
    #     """
    #     File name search to detect if the input file is mp3 or wav
    #     :param string: file name as a string
    #     :return: 1 (it's mp3 or ogg); 0 (it's wav or something alse)
    #     """
    #
    #     return re.search('.+\.mp3|.+\.ogg', string)
    #
    # @staticmethod
    # def _ogg_file_name(string):
    #     """
    #     Split the path to file as 'path/to/file' and 'file_name.wav' and transforms 'file_name.wav' to file_name.ogg '
    #     :param string: file name as a string
    #     :return: path to new file_name
    #     """
    #
    #     head, tail = os.path.split(string)
    #
    #     new_tail = '.'.join([re.split('\.', tail)[0], 'ogg'])
    #
    #     return os.path.join(head, new_tail)
