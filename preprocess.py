from pydub import AudioSegment
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt


def convert_mp3_to_mono_wav(src):
    mono_dst = src + ".wav"

    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    sound = sound.set_channels(1)  # stereo to mono
    sound.export(mono_dst, format="wav")
    # os.remove(sound)
    return mono_dst


def create_spectrogram_from_wav(file):
    mono_audio = convert_mp3_to_mono_wav(file)
    # https://stackoverflow.com/questions/47147146/save-spectrogram-only-content-without-axes-or-anything-else-to-a-file-using-m
    sample_rate, samples = wavfile.read(mono_audio)

    fig, ax = plt.subplots(1)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    # using scale linear to avoid divide by zero error
    pxx, freqs, bins, im = ax.specgram(x=samples, Fs=sample_rate)
    fig.savefig(mono_audio + ".png")  # 640 x 480

    # Memory issue
    plt.close(fig)
    os.remove(mono_audio)
    return


def walk_through_directory_creating_spectrograms_from_mp3s(path):
    # convert mp3 files
    for root, directories, filenames in os.walk(path):
        # for directory in directories:
        #     print(os.path.join(root, directory))
        for filename in filenames:
            filepath = os.path.join(root, filename)
            if filepath.endswith(".mp3") \
                    and not os.path.isfile(filepath + ".png"):
                try:
                    create_spectrogram_from_wav(filepath)
                except Exception as e:
                    print(filepath)
                    print(e)
                    # /home/dan/fma_large/002/002624.mp3
                    # /home/dan/fma_small/133/133297.mp3
                    # /home/dan/fma_small/099/099134.mp3
                    # /home/dan/fma_small/108/108925.mp3
    return


walk_through_directory_creating_spectrograms_from_mp3s("/home/dan/fma_large/")

# https://github.com/crowdAI/crowdai-musical-genre-recognition-starter-kit

# https://github.com/sdcubber/Keras-Sequence-boilerplate/blob/master/Keras-Sequence.ipynb

# https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/

# https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification

# https://www.tensorflow.org/tutorials/load_data/images