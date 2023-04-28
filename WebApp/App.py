import streamlit as st
#import cv2
from tensorflow.keras.models import load_model
import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
import numpy as np



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
