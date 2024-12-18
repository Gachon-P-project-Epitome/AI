#================================================================================
# FMA_Create_Echo.py
# Inputs:
#   Wave files of FMA 8 genres in 8 directories, f = 16kHz, PCM 16
# Outputs
#   Echoed Wave files of 8 genres in 8 directories, fs = 16kHz, PCM 16
#   Echo Duration: 250 ms, 3 times repeated, decreased amplitude by a factor 0.25
# Authors: TVL & DTLT
#================================================================================
import librosa
import soundfile as sf
from audioop import add
from audioop import mul
import wave
from warnings import warn
import os
import sys

def input_wave(filename,frames=5000000): # 5000000 is par default, enough large number of frames
    with wave.open(filename,'rb') as wave_file:
		# Get parameters from file header
        params=wave_file.getparams()
		# audio contains an array of bytes
        audio=wave_file.readframes(frames)
    return params, audio


def output_wave(audio, params, nfile):
    with wave.open(nfile,'wb') as wave_file:
        wave_file.setparams(params)
        wave_file.writeframes(audio)

def delay(audio_bytes,params,offset_ms,factor=1.0,num=1):
    # calculate the number of bytes which corresponds to the offset in milliseconds
    offset=params.sampwidth*offset_ms*int(params.framerate/1000)
    # add extra space at the end for the delays
    audio_bytes=audio_bytes+b'\0'*offset*(num)
    # create a copy of the original to apply the delays
    delayed_bytes=audio_bytes
    for i in range(num):
        # create some silence
        beginning = b'\0'*offset*(i+1)
        # remove space from the end
        end = audio_bytes[:-offset*(i+1)]
        # multiply by the factor
        multiplied_end= mul(end,params.sampwidth,factor**(i+1))
        delayed_bytes= add(delayed_bytes, beginning+multiplied_end, params.sampwidth)
    return delayed_bytes

# Saving echoed wave file
def delay_to_file(audio_bytes, params, offset_ms, file_name, factor=1.0, num=1):
    echoed_bytes=delay(audio_bytes, params, offset_ms, factor,num)
    output_wave(echoed_bytes, params, file_name)

i = 0
genres = 'Electronic Experimental Folk Hip_Hop Instrumental International Pop Rock'.split()
for g in genres:
    for k, filename in enumerate(os.listdir(f'/home/hwang-gyuhan/Workspace/dataset/Spotify_wav_pcm/{g}')):
        songname = f'/home/hwang-gyuhan/Workspace/dataset/Spotify_wav_pcm/{g}/{filename}'
        file_name = f'{g}/{filename}'
        print(songname, ", i =  ", i, "/8000")
        newpath = f'/home/hwang-gyuhan/Workspace/dataset/Spotify_wav_pcm_echo/{g}'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        # Path to echoed files
        new_name = f'/home/hwang-gyuhan/Workspace/dataset/Spotify_wav_pcm_echo/{g}/{filename}'
        new_name = new_name.split(".")[0] + "_ECHO.wav"
        wav_params, wav_bytes = input_wave(songname )
        delay_to_file(wav_bytes, wav_params, offset_ms=250, file_name=new_name, factor=.25, num=3)
        print(new_name)
        i = i + 1
