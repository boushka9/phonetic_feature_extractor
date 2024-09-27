import os 
import torch
import numpy
from feature_extractor import FeatureExtractor

hubert_base = FeatureExtractor()
wavfiles = ["EN_006.wav","EN_007.wav","EN_013.wav","EN_033.wav","EN_043.wav",]
for wav in wavfiles:
    title,extension = wav.split(".")
    filepath = "features_npformat_"+title+".npy"
    filenp = open(filepath,"bw+")

    path = os.path.join(os.getcwd(),"en-dialogs/labelables/",wav)
    print("This is the current path",path)
    print("Extracting features from ",wav)

    features_predicted = hubert_base.get_transformation_layers(path)
    print("Features extracted from ",wav)
    print("Number of slices: ",len(features_predicted))

    for feature in features_predicted:
        print("this is the current tensor size ",feature.size())
        features_numpy = feature.numpy()
        numpy.save(filenp,features_numpy)

    print("Finished with ",wav)