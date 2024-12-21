import numpy as np
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append('/home/hwang-gyuhan/Workspace/DenseNet/TestCode')
from Test import GenrePredictor


class CosineSimilarity:
    def __init__(self, img_path, weights_file_path, vector_dir_path):
        self.img_path = img_path
        self.weights_file_path = weights_file_path
        self.vector_dir_path = vector_dir_path
        self.genre_predictor = self.initialize_genre_predictor()

    def initialize_genre_predictor(self):
        return GenrePredictor(self.img_path, self.weights_file_path, self.vector_dir_path)

    def extract_features(self, intermediate_layer_names):
        all_features = self.genre_predictor.extract_features(intermediate_layer_names)
        print("Extracted Features Shape:")
        print(all_features.shape)
        return all_features

    def predict_genre_and_calculate_similarity(self, all_features):
        predicted_genre_data = self.genre_predictor.predict_genre()

        if predicted_genre_data is not None:
            print("Shape of the extracted vector from the NPZ file:")
            print(predicted_genre_data.shape)

            npz_file_path = os.path.join(self.vector_dir_path, "Pop.npz")  
            data = np.load(npz_file_path, allow_pickle=True)
            file_names = data['file_names']

            cosine_similarities = cosine_similarity(all_features, predicted_genre_data)

            print("Cosine Similarities between the image vector and the predicted genre vectors:")
            print(cosine_similarities.shape)
            print(cosine_similarities)

            top_5_indices = np.argsort(cosine_similarities[0])[::-1][:5]
            top_5_values = cosine_similarities[0][top_5_indices]

            for i, (idx, value) in enumerate(zip(top_5_indices, top_5_values), 1):
                song_name = file_names[idx]  
                print(f"Top {i}: {song_name}, similarity: {value}")
        else:
            print("No NPZ data extracted.")


def main():
    img_path = "/home/hwang-gyuhan/Workspace/dataset/FMA_IMAGES/Pop4SCAQF5uqm2SxxGM7LmxPj.png"
    weights_file_path = "/home/hwang-gyuhan/Workspace/DenseNet/ForTraining&Testing/weights/epoch_070_weights.h5"
    vector_dir_path = "/home/hwang-gyuhan/Workspace/dataset/vector"

    intermediate_layer_names = [
        "conv2_block6_concat",
        "conv3_block12_concat",
        "conv4_block24_concat",
        "conv5_block16_concat"
    ]

    pipeline = CosineSimilarity(img_path, weights_file_path, vector_dir_path)
    all_features = pipeline.extract_features(intermediate_layer_names)
    pipeline.predict_genre_and_calculate_similarity(all_features)


if __name__ == "__main__":
    main()