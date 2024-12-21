import subprocess
import librosa
import soundfile as sf
import numpy as np


class Preprocessing:
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def wav_convert(self, mp3_path, output_wav_path):
        """Converts MP3 to WAV and saves it to disk."""
        cmd = f"ffmpeg -hide_banner -loglevel panic -y -i \"{mp3_path}\" -f wav \"{output_wav_path}\""
        subprocess.run(cmd, shell=True, check=True)
        print(f"Successfully converted MP3 to WAV: {mp3_path} -> {output_wav_path}")
        
    def pcm_convert(self, wav_path, output_pcm_path):
        """Converts WAV to PCM and saves it to disk."""
        wav_data, sr = librosa.load(wav_path, sr=self.sr, mono=True)
        sf.write(output_pcm_path, wav_data, sr, format='WAV', subtype='PCM_16')
        print(f"Successfully converted WAV to PCM16: {wav_path} -> {output_pcm_path}")
    
    def process_audio(self, mp3_path, wav_path, pcm_path):
        """Processes MP3 file by converting to WAV and then to PCM."""
        self.wav_convert(mp3_path, wav_path) 
        self.pcm_convert(wav_path, pcm_path)  