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
from pydub import AudioSegment
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import os
import sys
import wave
import scipy
import scipy.io.wavfile as wav
import scipy.io.wavfile
from scipy.io.wavfile import read



# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler, OneHotEncoder


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
    website_menu = st.sidebar.selectbox("Menu", ("Stress Detection", "Project description", "Our team",
                                                 "Leave feedback", "Relax"))
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if website_menu == "Stress Detection":
        st.markdown("## Upload the file")

        with st.container():
            col1, col2 = st.columns(2)
            # audio_file = None
            # path = None
            with col1:
                audio_file = st.file_uploader(
                    "Upload audio file", type=['wav'])
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
                        #audio/
                        audioo = AudioSegment.from_file(path)
                        audioo.export("converted_"+path, format="wav")
                        path="converted_"+path
                        samplerate, data = read(path)

                        trimmed_file = data[np.absolute(data) > 50]

                        scipy.io.wavfile.write(
                            "trimmed_"+path, samplerate, trimmed_file)
                        p1 = path
                        path = "trimmed_"+path
                        
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
                    
            st.markdown("## Predictions...")
            with st.container():
                f = get_feature(path)
                print(f)
                scaler = StandardScaler()
                #f=np.array([f])
                f = scaler.fit_transform([f])
                #f=np.array(f).reshape(1, -1)
                print(f)
                prediction = model.predict(f)
                print(prediction)
                st.subheader("The Predicted output is:  "+prediction[0])



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

    elif website_menu == "Project description":
        import pandas as pd
        import plotly.express as px
        st.title("Project description")
        st.subheader("GitHub")
        link = '[GitHub repository of the web-application]' \
               '(https://github.com/jashshah-2103/BE_Project_Grp_52)'
        st.markdown(link, unsafe_allow_html=True)

        st.subheader("Theory")
        link = '[Theory behind - Medium article]' \
               '(https://talbaram3192.medium.com/classifying-emotions-using-audio-recordings-and-python-434e748a95eb)'
        st.markdown(link + ":clap::clap::clap: Tal!", unsafe_allow_html=True)
        with st.expander("See Wikipedia definition"):
            components.iframe("https://en.wikipedia.org/wiki/Voice_stress_analysis",
                              height=320, scrolling=True)  

        st.subheader("Dataset")
        txt = """
            This web-application is a part of the final year project. 
           
            """
        img = Image.open("Images/datasetdesc.png")
    
        st.image(img, width=300)

        st.markdown(txt, unsafe_allow_html=True)  

    elif website_menu == "Our team":
        st.subheader("Our team")
        st.balloons()
        col1, col2 = st.columns([3, 2])
        with col1:
            st.info("jashshah2103@gmail.com")
            st.info("adityashinde2722@gmail.com")
            st.info("kulkarnishreyas122@gmail.com")
            st.info("asherholder123@gmail.com")
        with col2:
            liimg = Image.open("Images/LI-Logo.png")
            st.image(liimg)
            st.markdown(f""":speech_balloon: [Jash Shah](https://www.linkedin.com/in/jashshah-2103/)""",
                        unsafe_allow_html=True)
            st.markdown(f""":speech_balloon: [Aditya Shinde](https://www.linkedin.com/in/aditya-shinde-23ba06202/)""",
                        unsafe_allow_html=True)
            st.markdown(f""":speech_balloon: [Shreyas Kulkarni](https://www.linkedin.com/in/shreyas-kulkarni-958434207/)""",
                        unsafe_allow_html=True)
            st.markdown(f""":speech_balloon: [Asher Holder](https://www.linkedin.com/in/asher-holder-526a05173)""",
                        unsafe_allow_html=True)

if __name__ == '__main__':
    main()
