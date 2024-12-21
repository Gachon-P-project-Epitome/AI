import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class FeatureExtractor:
    def __init__(self, input_size=(224, 224, 3), num_classes=8):
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        """Builds the DenseNet121 model."""
        base_model_densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=self.input_size)
        headModel = base_model_densenet.output
        headModel = Dropout(0.5)(headModel)
        headModel = GlobalAveragePooling2D()(headModel)
        headModel = Dense(1024, activation='relu')(headModel)
        headModel = Flatten()(headModel)
        headModel = Dense(1024, activation='relu')(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(1024, activation='relu')(headModel)
        headModel = Dense(self.num_classes, activation='sigmoid')(headModel)
        model = Model(inputs=base_model_densenet.input, outputs=headModel)
        return model

    def preprocess_image(self, image_path):
        """
        Preprocesses the image to the required input size and format.
        Args:
            image_path (str): Path to the image.
        Returns:
            np.array: Preprocessed image.
        """
        image = load_img(image_path, target_size=self.input_size[:2])  # Resize image
        image_array = img_to_array(image)  # Convert to array
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = image_array / 255.0  # Normalize to [0, 1]
        return image_array

    def extract_features(self, image_path, feature_layers):
        """
        Extracts features from specified layers.
        Args:
            image_path (str): Path to the input image.
            feature_layers (dict): Dictionary with indices and layer names.
        Returns:
            np.array: Extracted features as a 2D numpy array.
        """
        input_tensor = self.preprocess_image(image_path)  # Preprocess the image
        
        # List to store extracted features as numpy arrays
        features_list = []
        
        # Create intermediate models for each feature layer
        intermediate_models = {
            name: Model(inputs=self.model.input, outputs=self.model.get_layer(name).output)
            for name in feature_layers.values()
        }

        # Extract features for each layer and convert to numpy array
        for layer_name, sub_model in intermediate_models.items():
            features = sub_model(input_tensor)  # Get the features as a Tensor
            features = features.numpy()  # Convert to numpy
            
            # Flatten the features to 2D
            features_flattened = features.flatten()  # Flatten into a 1D array
            features_list.append(features_flattened)

        # Stack all the 1D features into a single 2D numpy array
        return np.concatenate(features_list, axis=0)  # Concatenate all features into a 1D array

def process_images_and_extract_features(image_dir, feature_layers, output_npz, target_genre):
    extractor = FeatureExtractor()
    
    # Get all files in the directory that don't contain '_'
    valid_files = [
        f for f in os.listdir(image_dir)
        if f.endswith('.png') and '_' not in f
    ]
    
    # List to store all features (one array for each image)
    all_features = []
    all_file_names = []  # List to store the corresponding file names
    
    total_files = len(valid_files)
    print(f"Total files to process: {total_files}")

    # Define the genres to filter
    genres = ["Electronic", "Experimental", "Folk", "HipPop", "Pop", "Rock", "Instrumental", "International"]

    # Filter files by the target genre
    target_files = [
        f for f in valid_files
        if target_genre in f.split('.')[0]  # 'Pop' 포함
        and not f.split('.')[0].startswith("Hip")
        ]

    print(f"Files to process for genre '{target_genre}': {len(target_files)}")

    for idx, file_name in enumerate(target_files):
        print(f"Processing file {idx + 1}/{len(target_files)}: {file_name}")
        image_path = os.path.join(image_dir, file_name)
        extracted_features = extractor.extract_features(image_path, feature_layers)
        
        # Add the extracted features to the list
        all_features.append(extracted_features)
        all_file_names.append(file_name)  # Save the file name corresponding to the features

    # Convert the list of features to a single numpy array
    all_features = np.array(all_features)

    # Save the features and corresponding file names to a single NPZ file
    np.savez_compressed(output_npz, features=all_features, file_names=all_file_names)
    print(f"Features saved to {output_npz}")

# Example usage
if __name__ == "__main__":
    image_dir = "/home/hwang-gyuhan/Workspace/dataset/FMA_IMAGES"  # Replace with the path to your image directory
    feature_layers = {
        '48': 'conv2_block6_concat',
        '136': 'conv3_block12_concat',
        '308': 'conv4_block24_concat',
        '424': 'conv5_block16_concat',
    }
    output_npz = "/home/hwang-gyuhan/Workspace/dataset/vector/Electronic.npz"  # Output NPZ file path
    
    # Set the target genre to "Pop"
    process_images_and_extract_features(image_dir, feature_layers, output_npz, target_genre="Electronic")