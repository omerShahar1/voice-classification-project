import pickle as pk
import tkinter as tk
from tkinter import filedialog
import pyaudio
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack
from utils import create_model_age, create_model_gender

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30


def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def trim(snd_data):
    "Trim the blank spots at the start and end"

    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds * RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds * RATE))])
    return r


def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def extract_feature(file_name):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    X, sample_rate = librosa.load(file_name)

    stft = np.abs(librosa.stft(X))
    result = np.array([])

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))

    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, contrast))

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    result = np.hstack((result, tonnetz))
    return result


label2age = {
    0: "teens",
    1: "twenties",
    2: "thirties",
    3: "fourties",
    4: "fifties",
    5: "sixties",
    6: "seventies or more",
}


def calc_prob(file):
    model_age = create_model_age()
    model_gender = create_model_gender()
    model_age.load_weights("results/model_age.h5")
    model_gender.load_weights("results/model_gender.h5")

    features = np.zeros((1, 193))
    features[0] = extract_feature(file)  # extract features

    sc_age = pk.load(open("tools/MinMaxScaler_age.pkl", 'rb'))
    features_age = sc_age.transform(features)

    pca_age = pk.load(open("tools/PCA_age.pkl", 'rb'))
    features_age = pca_age.transform(features_age)

    sc_gender = pk.load(open("tools/MinMaxScaler_gender.pkl", 'rb'))
    features_gender = sc_gender.transform(features)

    pca_gender = pk.load(open("tools/PCA_gender.pkl", 'rb'))
    features_gender = pca_gender.transform(features_gender)
    # predict the gender!
    male_prob = model_gender.predict(features_gender)[0][0]
    #male_prob = model_gender.predict(features)[0][0]

    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"

    prob = model_age.predict(features_age)[0]
    max_prob = -1
    final_predict = -1
    for i in range(7):
        if max_prob < prob[i]:
            final_predict = label2age[i]
            max_prob = prob[i]
    # show the result!
    result_gender_label.configure(text=gender)
    result_age_label.configure(text=final_predict)

    print("Result:\n Gender: ", gender)
    print("predict Age: ", final_predict)



def on_enter_path_address():
    path = filedialog.askopenfilename()
    file_label.configure(text=path)
    calc_prob(path)


# Create the "Record Audio" button
def on_record_audio():
    print("Please talk")
    # put the file name here
    file = "test.wav"
    # record the file (start talking)
    record_to_file(file)
    file_label.configure(text="record")
    calc_prob(file)


def on_exit():
    window.destroy()


if __name__ == "__main__":
    # load the saved model (after training)
    # model = pickle.load(open("result/mlp_classifier.model", "rb"))
    window = tk.Tk()
    window.title("My GUI")
    window.geometry("800x500")
    enter_path_address_button = tk.Button(text="Enter Path Address", command=on_enter_path_address,font=("Helvetica", 20))
    enter_path_address_button.pack()

    record_audio_button = tk.Button(text="Record Audio", command=on_record_audio,font=("Helvetica", 20))
    record_audio_button.pack()

    exit_button = tk.Button(text="Exit", command=on_exit,font=("Helvetica", 20))
    exit_button.pack()

    file_label = tk.Label(text="file name: ",font=("Helvetica", 20))
    file_label.pack()

    result_gender_label = tk.Label(text="gender is: ",font=("Helvetica", 20))
    result_gender_label.pack()

    result_age_label = tk.Label(text="age is: 0",font=("Helvetica", 20))
    result_age_label.pack()

    window.configure(bg='lightblue')
    enter_path_address_button.configure(bg='lightgreen', activebackground='green')
    record_audio_button.configure(bg='yellow', activebackground='orange')
    exit_button.configure(bg='pink', activebackground='red')

    # Run the main loop
    window.mainloop()
