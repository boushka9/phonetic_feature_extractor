import os
import numpy as np
from feature_extractor import FeatureExtractor


def process_directory(input_directory, output_directory, extractor):
    """
    Processes all .wav files in the input directory and saves the extracted features to the output directory.
    :param input_directory: Directory containing .wav files.
    :param output_directory: Directory where extracted features will be saved.
    :param extractor: An instance of the FeatureExtractor class.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over all .wav files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_features.npy")

            print(f"Processing: {input_path}")

            # Extract features using the FeatureExtractor
            features_predicted = extractor.get_transformation_layers(input_path)

            if features_predicted is not None:
                # Save features to the output directory
                with open(output_path, "wb") as f:
                    for feature in features_predicted:
                        features_numpy = feature.cpu().numpy()
                        np.save(f, features_numpy)
                print(f"Saved features to: {output_path}")
            else:
                print(f"Failed to process: {input_path}")


if __name__ == '__main__':
    # Directories
    input_directory = "input-wavs"
    output_directory = "output-features"

    # Initialize the feature extractor
    hubert_base = FeatureExtractor(bundle='hubert_b')

    # Process the directory
    process_directory(input_directory, output_directory, hubert_base)