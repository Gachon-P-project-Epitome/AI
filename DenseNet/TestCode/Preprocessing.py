import io
import subprocess
import librosa
import soundfile as sf
import numpy as np


class Preprocessing:
    def __init__(self, sr=16000):
        self.sr = sr

    def wav_convert(self, mp3_path):
        temp_wav = io.BytesIO()
        cmd = f"ffmpeg -hide_banner -loglevel panic -y -i \"{mp3_path}\" -f wav pipe:1"
        process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        temp_wav.write(process.stdout)
        temp_wav.seek(0)
        data, sr = librosa.load(temp_wav, sr=self.sr, mono=True)
        print(f"Successfully converted MP3 to WAV (in memory): {mp3_path}")
        return data, sr

    def pcm_convert(self, wav_data, samplerate):
        pcm_buffer = io.BytesIO()
        sf.write(pcm_buffer, wav_data, samplerate, format='WAV', subtype='PCM_16')
        pcm_buffer.seek(0)
        print("Successfully converted WAV to PCM16 (in memory)")
        return pcm_buffer

    def process_audio(self, mp3_path):
        wav_data, samplerate = self.wav_convert(mp3_path)  # MP3 -> WAV 변환
        if wav_data is not None:
            pcm_data = self.pcm_convert(wav_data, samplerate)  # WAV -> PCM 변환
            return pcm_data
        
        return None