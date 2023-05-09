import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from joblib import dump, load

# from pyin import pitch
# import parselmouth
from sklearn.preprocessing import StandardScaler
import os
import sys
import wave
import scipy
import scipy.io.wavfile as wav
import scipy.io.wavfile
from scipy.io.wavfile import read

# from joblib import Parallel, delayed
# import joblib

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

#import keras

from sklearn.preprocessing import StandardScaler, OneHotEncoder
# to play the audio files
#from IPython.display import Audio


model = load('/Users/jashshah/Documents/GitHub/BE_Project_Grp_52/rfmodel.joblib')
# constants
starttime = datetime.now()

CAT = ["unstressed", "neutral", "stressed"]

COLOR_DICT = {"neutral": "grey",
              "unstressed": "green",
              "stressed": "red",
              }

st.set_page_config(page_title="SER web-app",
                   page_icon=":speech_balloon:", layout="wide")


def log_file(txt=None):
    with open("log.txt", "a") as f:
        datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"{txt} - {datetoday};\n")


def save_audio(file):
    if file.size > 4000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {datetoday};\n")
    except:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb)


def create_waveplot(data, sr):
    plt.figure(figsize=(10, 3))
   # plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
   # plt.show()


def create_spectrogram(data, sr):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    #plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.show()
# @st.cache
# def get_title(predictions, categories=CAT6):
#     title = f"Detected emotion: {categories[predictions.argmax()]} \
#     - {predictions.max() * 100:.2f}%"
#     return title


# @st.cache
# def color_dict(coldict=COLOR_DICT):
#     return COLOR_DICT

# @st.cache
# def plot_polar(fig, predictions=TEST_PRED, categories=TEST_CAT,
#                title="TEST", colors=COLOR_DICT):
#     # color_sector = "grey"

#     N = len(predictions)
#     ind = predictions.argmax()

#     COLOR = color_sector = colors[categories[ind]]
#     theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
#     radii = np.zeros_like(predictions)
#     radii[predictions.argmax()] = predictions.max() * 10
#     width = np.pi / 1.8 * predictions
#     fig.set_facecolor("#d1d1e0")
#     ax = plt.subplot(111, polar="True")
#     ax.bar(theta, radii, width=width, bottom=0.0, color=color_sector, alpha=0.25)

#     angles = [i / float(N) * 2 * np.pi for i in range(N)]
#     angles += angles[:1]

#     data = list(predictions)
#     data += data[:1]
#     plt.polar(angles, data, color=COLOR, linewidth=2)
#     plt.fill(angles, data, facecolor=COLOR, alpha=0.25)

#     ax.spines['polar'].set_color('lightgrey')
#     ax.set_theta_offset(np.pi / 3)
#     ax.set_theta_direction(-1)
#     plt.xticks(angles[:-1], categories)
#     ax.set_rlabel_position(0)
#     plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
#     plt.suptitle(title, color="darkblue", size=12)
#     plt.title(f"BIG {N}\n", color=COLOR)
#     plt.ylim(0, 1)
#     plt.subplots_adjust(top=0.75)


def extract_features(data,sample_rate):
    # ZCR - The rate of sign-changes of the signal during the duration of a particular frame
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft -STFT represents information about the classification of pitch and signal structure
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC- Mel Frequency Cepstral Coefficients form a cepstral representation where the frequency bands are not linear but distributed according to the mel-scale
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(
        y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def get_feature(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data,sample_rate)
    result = np.array(res1)

    return result


def main():
    side_img = Image.open("Images/emotion3.jpg")
    with st.sidebar:
        st.image(side_img, width=300)
    st.sidebar.subheader("Menu")
    website_menu = st.sidebar.selectbox("Menu", ("Emotion Recognition", "Project description", "Our team",
                                                 "Leave feedback", "Relax"))
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.markdown("## Upload the file")

    with st.container():
        col1, col2 = st.columns(2)
        # audio_file = None
        # path = None
        with col1:
            audio_file = st.file_uploader(
                "Upload audio file", type=['wav', 'mp3', 'ogg'])
            if audio_file is not None:
                if not os.path.exists("audio"):
                    os.makedirs("audio")
                path = os.path.join("audio", audio_file.name)
                if_save_audio = save_audio(audio_file)
                if if_save_audio == 1:
                    st.warning("File size is too large. Try another file.")
                elif if_save_audio == 0:
                    # extract features
                    # display audio
                    st.audio(audio_file, format='audio/wav', start_time=0)
                    samplerate, data = read(path)

                    trimmed_file = data[np.absolute(data) > 50]

                    scipy.io.wavfile.write(
                        "trimmed_"+path, samplerate, trimmed_file)
                    p1 = path
                    path = "trimmed_"+path
                    # try:
                    #     wav, sr = librosa.load(path)
                    #     #create_waveplot(wav, sr)
                    #     create_spectrogram(wav, sr)
                    #     # # display audio
                    #     # st.audio(audio_file, format='audio/wav', start_time=0)
                    # except Exception as e:
                    #     audio_file = None
                    #     st.error(
                    #         f"Error {e} - wrong format of the file. Try another .wav file.")
                else:
                    st.error("Unknown error")
            else:
                if st.button("Try test file"):
                    p1 = "/Users/jashshah/Documents/GitHub/BE_Project_Grp_52/audio_dataset_final/Aditya1S1_angry.wav"
                    samplerate, data = read(p1)

                    trimmed_file = data[np.absolute(data) > 50]

                    scipy.io.wavfile.write(
                        "test.wav", samplerate, trimmed_file)
                    wav, sr = librosa.load("test.wav")
                    # display audio
                    st.audio(p1, format='audio/wav', start_time=0)
                    path = "test.wav"
                    audio_file = "test"

    with col2:
        if audio_file is not None:

            # samplerate, data = read(path)
            # trimmed_file = data[np.absolute(data) > 50]
            # #librosa.output.write_wav(path,data,sr=samplerate)
            # scipy.io.wavfile.write(path, samplerate, trimmed_file)
            # print(path)
            wav, sr = librosa.load(path)
            fig = plt.figure(figsize=(10, 2))
            fig.set_facecolor('#d1d1e0')
            plt.title("Wave-form")
            librosa.display.waveshow(wav, sr=44100)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.spines["right"].set_visible(False)
            plt.gca().axes.spines["left"].set_visible(False)
            plt.gca().axes.spines["top"].set_visible(False)
            plt.gca().axes.spines["bottom"].set_visible(False)
            plt.gca().axes.set_facecolor('#d1d1e0')
            st.write(fig)
        else:
            pass

    if audio_file is not None:
        st.markdown("## Analyzing...")
        if not audio_file == "test":
            st.sidebar.subheader("Audio file")
            file_details = {"Filename": audio_file.name,
                            "FileSize": audio_file.size}
            st.sidebar.write(file_details)

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                mfcc_features = librosa.feature.mfcc(y=wav, sr=sr)
                fig = plt.figure(figsize=(10, 2))
                fig.set_facecolor('#d1d1e0')
                plt.title("MFCCs")
                librosa.display.specshow(mfcc_features, sr=sr, x_axis='time')
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().axes.spines["right"].set_visible(False)
                plt.gca().axes.spines["left"].set_visible(False)
                plt.gca().axes.spines["top"].set_visible(False)
                st.write(fig)

            with col2:
                fig2 = plt.figure(figsize=(10, 2))
                fig2.set_facecolor('#d1d1e0')
                plt.title("Mel-Spectrogram")
                X = librosa.stft(wav)
                Xdb = librosa.amplitude_to_db(abs(X))
                librosa.display.specshow(
                    Xdb, sr=sr, x_axis='time', y_axis='hz')
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().axes.spines["right"].set_visible(False)
                plt.gca().axes.spines["left"].set_visible(False)
                plt.gca().axes.spines["top"].set_visible(False)
                st.write(fig2)

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                chromagram = librosa.feature.chroma_cqt(y=wav, sr=sr)
                fig = plt.figure(figsize=(10, 2))
                fig.set_facecolor('#d1d1e0')
                plt.title("Chromagram Plot")
                librosa.display.specshow(
                    chromagram, y_axis='chroma', x_axis='time')
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().axes.spines["right"].set_visible(False)
                plt.gca().axes.spines["left"].set_visible(False)
                plt.gca().axes.spines["top"].set_visible(False)
                st.write(fig)
                f = get_feature(path)
                scaler = StandardScaler()
                f=np.array(f).reshape(1, -1)
                f = scaler.fit_transform(f)
                prediction = model.predict(f)
                print(prediction)



            # with col2:
            #     tempo, beat_times = librosa.beat.beat_track(y=wav, sr=sr, start_bpm=60, units='time')

            #     librosa.display.waveshow(wav, alpha=0.6)
            #     plt.vlines(beat_times, -1, 1, color='r')
            #     plt.ylim(-1, 1)
            #     fig2 = plt.figure(figsize=(10, 2))
            #     fig2.set_facecolor('#d1d1e0')
            #     plt.title("Mel-Spectrogram")
            #     plt.gca().axes.get_yaxis().set_visible(False)
            #     plt.gca().axes.spines["right"].set_visible(False)
            #     plt.gca().axes.spines["left"].set_visible(False)
            #     plt.gca().axes.spines["top"].set_visible(False)
            #     st.write(fig2)


if __name__ == '__main__':
    main()
