import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# FeatureLabelProcessor 클래스 정의
class FeatureLabelProcessor:
    def __init__(self, tracks_csv_path):
        self.tracks_csv_path = tracks_csv_path
        self.track_genre_mapping = self._load_and_filter_tracks()
        self.label_binarizer = LabelBinarizer()
        self.genre_to_onehot = self._onehot_encode_genres()

    def _load_and_filter_tracks(self):
        # 데이터 로드 및 필터링
        tracks_df = pd.read_csv(self.tracks_csv_path, header=[0, 1])
        set_column_name = tracks_df.columns[32]
        genre_column_name = tracks_df.columns[40]
        valid_sets = ['small']
        filtered_tracks = tracks_df[tracks_df[set_column_name].isin(valid_sets)]
        track_ids = filtered_tracks.iloc[:, 0].tolist()
        genres = filtered_tracks[genre_column_name].tolist()
        track_genre_mapping = dict(zip(track_ids, genres))
        return track_genre_mapping

    def _onehot_encode_genres(self):
        # 장르 리스트 추출
        genres = list(self.track_genre_mapping.values())
        # LabelBinarizer로 원-핫 인코딩
        onehot_encoded_genres = self.label_binarizer.fit_transform(genres)
        
        # 장르 원-핫 인코딩 딕셔너리로 변환 (트랙 ID -> 원-핫 인코딩된 장르)
        track_ids = list(self.track_genre_mapping.keys())
        genre_to_onehot = dict(zip(track_ids, onehot_encoded_genres))
        return genre_to_onehot

    def print_unique_genre_onehot_mapping(self):
        # 고유한 장르와 그에 대응하는 원-핫 인코딩 결과 출력
        unique_genres = self.label_binarizer.classes_
        onehot_encoded_genres = self.label_binarizer.transform(unique_genres)
        
        print("Unique Genre to One-Hot Encoded Mapping:")
        for genre, onehot in zip(unique_genres, onehot_encoded_genres):
            print(f"Genre: {genre}, One-Hot Encoded: {onehot}")

# 클래스 사용 예시
tracks_csv_path = '/home/hwang-gyuhan/Workspace/Tabnet/FMA 복사본/fma_metadata/tracks.csv'

processor = FeatureLabelProcessor(tracks_csv_path)
processor.print_unique_genre_onehot_mapping()
