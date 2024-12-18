#
#  FMA conversion script,  genre classification
#  Author: Scott Hawley
#  License: Do as you like
#
# For FMA dataset https://github.com/mdeff/fma
# to be used with panotti https://github.com/drscotthawley/panotti
#
# This will create a directory called Samples/
#   inside which will be directories for each genre, e.g. Rock/ Instrumental/ etc.
#   within these directories will be the audio files converted to .wav format
#
# Requires: fma
#           librosa
#           ffmpeg
#           ...hour(s) to run.  it's slow, runs in serial
import os
import subprocess
import pandas as pd
import librosa
import soundfile as sf

class AudioProcessor:
    def __init__(self, tracks_csv_path, audio_root_dir, wav_root_dir):
        self.tracks_csv_path = tracks_csv_path
        self.audio_root_dir = audio_root_dir  # MP3 파일이 여러 폴더에 저장된 상위 폴더
        self.wav_root_dir = wav_root_dir  # Wav 파일을 저장할 상위 폴더

    def _load_and_filter_tracks(self):
        # CSV에서 'small' 세트와 track_id, genre 필터링
        tracks_df = pd.read_csv(self.tracks_csv_path, header=[0, 1], low_memory=False)
        set_column_name = tracks_df.columns[32]  # 'set' 컬럼
        genre_column_name = tracks_df.columns[40]  # 'genre' 컬럼

        # 'small' 데이터 필터링
        filtered_tracks = tracks_df[tracks_df[set_column_name] == 'small']
        track_ids = filtered_tracks.iloc[:, 0].tolist()
        genres = filtered_tracks[genre_column_name].tolist()
        return dict(zip(track_ids, genres))

    def _find_mp3_file(self, track_id):
        # 상위 폴더에서 하위 폴더 재귀적으로 탐색하여 track_id에 해당하는 MP3 파일 경로 반환
        for root, _, files in os.walk(self.audio_root_dir):
            for file in files:
                if file == f"{str(track_id).zfill(6)}.mp3":
                    return os.path.join(root, file)
        return None  # 파일이 없는 경우

    def convert_and_save(self):
        # 트랙 ID와 장르 매핑
        track_genre_mapping = self._load_and_filter_tracks()

        # 장르별 출력 폴더 생성
        for genre in set(track_genre_mapping.values()):
            genre_folder = os.path.join(self.wav_root_dir, genre)
            os.makedirs(genre_folder, exist_ok=True)

        # MP3 -> WAV 변환
        sr = 16000  # 샘플링 주파수
        for track_id, genre in track_genre_mapping.items():
            mp3_path = self._find_mp3_file(track_id)
            if not mp3_path:
                print(f"MP3 파일을 찾을 수 없습니다: track_id={track_id}")
                continue

            # 출력 경로 설정
            wav_path = os.path.join(self.wav_root_dir, genre, f"{str(track_id).zfill(6)}.wav")
            print(f"Converting {mp3_path} -> {wav_path}")

            # MP3 -> WAV 변환
            try:
                cmd = f"ffmpeg -hide_banner -loglevel panic -y -i {mp3_path} {wav_path}"
                subprocess.run(cmd, shell=True, check=True)

                # librosa로 WAV 파일 다시 저장 (품질 유지)
                data, _ = librosa.load(wav_path, sr=sr, mono=True)
                sf.write(wav_path, data, sr)  # soundfile을 사용하여 저장
            except Exception as e:
                print(f"오류 발생: {mp3_path}: {e}")

# 사용 예시
tracks_csv_path = '/home/hwang-gyuhan/Workspace/dataset/fma_metadata/tracks.csv'
audio_root_dir = '/home/hwang-gyuhan/Workspace/dataset/fma_small'  # MP3 상위 폴더
wav_root_dir = '/home/hwang-gyuhan/Workspace/dataset/fma_wav'  # Wav 출력 상위 폴더

processor = AudioProcessor(tracks_csv_path, audio_root_dir, wav_root_dir)
processor.convert_and_save()