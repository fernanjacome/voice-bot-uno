import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split

# all emotions on RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'Calmado',
    '03': 'Feliz',
    '04': 'Triste',
    '05': 'Enojado',
    '06': 'Asustado',
    '07': 'Disgustado',
    '08': 'surprised'
}

# DataFlair - Emotions to observe
AVAILABLE_EMOTIONS = ['Calmado', 'Feliz',
                      'Triste', 'Enojado', 'Asustado', 'Disgustado']


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("data1/Actor_*/*.wav"):
        basename = os.path.basename(file)
        emotion = emotions[basename.split("-")[2]]
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        X.append(features)
        y.append(emotion)
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
