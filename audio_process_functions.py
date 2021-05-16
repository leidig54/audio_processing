from google.cloud import storage
from pathlib import Path
import glob
import numpy as np
import time
import os
from scipy import signal
from scipy.io.wavfile import read, write
from pylab import show, plot, subplot
import json
import codecs
from numpy import log10, mean
from pydub import AudioSegment
import timeit


def setup_environment():
    start = time.time()
    # Create and authenticate the google cloud python module
    storage_client = storage.Client.from_service_account_json(
        '../../keys/oto-ac5a8-bf12825a4018.json')

    # Fetch bucket
    bucket = storage_client.get_bucket('oto-ac5a8.appspot.com')

    # Get current blobs
    blob_list = []
    for file in storage_client.list_blobs(bucket):
        blob_list.append(file.name)

    end = time.time()
    print("Environment set up in " + str(end - start) + " seconds")
    return storage_client, bucket, blob_list


def process_normal_spectrogram(bucket, mask_name, blob_list, filepath):

    # Read the wav file
    sample_rate, data = read(filepath)

    # Get an audacity-style spectrogram
    s = 1024
    freqs, bins, pxx, = signal.spectrogram(
        data[:, 1], fs=sample_rate, window=signal.blackmanharris(s), nfft=s, noverlap=0, mode='magnitude')

    # Create a json file with the spectrogram values
    new_blob_name = "sound_therapy/spectrograms/normal/{}.json".format(
        mask_name)

    values = 20 * log10(mean(pxx, axis=1))

    json.dump(np.array(values).tolist(), codecs.open("temp.json", 'w', encoding='utf-8'),
              separators=(',', ':'), sort_keys=True, indent=4)

    # If a previous blob exists, delete it
    check_and_delete(new_blob_name=new_blob_name,
                     blob_list=blob_list, bucket=bucket)

    # Upload new blob
    blob = bucket.blob(new_blob_name)
    blob.upload_from_filename("temp.json")

    # Delete temporary json
    os.remove("temp.json")


def check_and_delete(new_blob_name, blob_list, bucket):
    if new_blob_name in blob_list:
        bucket.delete_blob(new_blob_name)


def process_normal_mp3(bucket, mask_name, blob_list, filepath):

    # Create normal mp3
    audio_segment = AudioSegment.from_wav(filepath)
    audio_segment.export("temp.mp3", format="mp3", bitrate="180")

    # Name normal mp3 blob
    new_blob_name = "sound_therapy/audio/normal/{}.mp3".format(mask_name)

    # Check if blob has previously been uploaded and delete it if found
    check_and_delete(new_blob_name=new_blob_name,
                     blob_list=blob_list, bucket=bucket)

    # Upload new blob
    blob = bucket.blob(new_blob_name)
    blob.upload_from_filename("temp.mp3")

    # Delete temporary mp3
    os.remove("temp.mp3")


def notch_filtering(wav, fs, w0, Q):
    """ Apply a notch (band-stop) filter to the audio signal.

    Args:
        wav: Waveform.
        fs: Sampling frequency of the waveform.
        w0: See scipy.signal.iirnotch.
        Q: See scipy.signal.iirnotch.

    Returns:
        wav: Filtered waveform.

    """
    b, a = signal.iirnotch(2 * w0/fs, Q)
    wav = signal.lfilter(b, a, wav)
    return wav


def peak_filtering(wav, fs, w0, Q):
    """ Apply a notch (band-stop) filter to the audio signal.

    Args:
        wav: Waveform.
        fs: Sampling frequency of the waveform.
        w0: See scipy.signal.iirnotch.
        Q: See scipy.signal.iirnotch.

    Returns:
        wav: Filtered waveform.
    """
    b, a = signal.iirpeak(2 * w0/fs, Q)
    wav = signal.lfilter(b, a, wav)
    return wav


def process_notched(bucket, mask_name, blob_list, filepath):

    # Generate notch frequency list
    frequency_list = []
    q_factor = 30

    # Low frequencies
    for i in range(20, 200, 20):
        frequency_list.append(i)

    # Mid frequencies
    for i in range(200, 1000, 100):
        frequency_list.append(i)

    # High frequencies
    for i in range(1000, 18000, 1000):
        frequency_list.append(i)

    sample_rate, data = read(filepath)

    for frequency in frequency_list:

        # Name the new blob
        new_blob_name = "sound_therapy/audio/notched/{}/{}.mp3".format(
            frequency, mask_name)

        # Notch the individual audio files
        result_left = notch_filtering(
            data[:, 0], sample_rate, frequency, q_factor)
        result_right = notch_filtering(
            data[:, 1], sample_rate, frequency, q_factor)

        print("Mask: " + str(mask_name) + ", Algorithm: Notched" + ", Frequency: " +
              str(frequency))

        # Generate wav and convert to mp3
        joined = np.transpose(
            np.array([result_left, result_right]))

        write("temp.wav", sample_rate, joined.astype(np.int32))
        sound = AudioSegment.from_wav("temp.wav")
        sound.export("temp.mp3", format="mp3", bitrate="180")

        # Check if blob has previously been uploaded and delete it if found
        check_and_delete(new_blob_name=new_blob_name,
                         blob_list=blob_list, bucket=bucket)

        # Upload new blob
        blob = bucket.blob(new_blob_name)
        blob.upload_from_filename("temp.mp3")

        # Delete temporary mp3
        os.remove("temp.mp3")

        # Get spectrogram for notched audio
        spectrogram_blob_name = "sound_therapy/spectrograms/notched/{}/{}.json".format(
            frequency, mask_name)

        # Read wav file
        sample_rate, data = read("temp.wav")

        # Get an audacity-style spectrogram
        s = 1024
        freqs, bins, pxx, = signal.spectrogram(
            data[:, 1], fs=sample_rate, window=signal.blackmanharris(s), nfft=s, noverlap=0, mode='magnitude')

        # Create a json file with the spectrogram values
        values = 20 * log10(mean(pxx, axis=1))

        json.dump(np.array(values).tolist(), codecs.open("temp.json", 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True, indent=4)

        # Check if blob has previously been uploaded and delete it if found
        check_and_delete(new_blob_name=spectrogram_blob_name,
                         blob_list=blob_list, bucket=bucket)

        # Upload new blob
        blob = bucket.blob(spectrogram_blob_name)
        blob.upload_from_filename("temp.json")

        # Delete temporary json
        os.remove("temp.json")
        os.remove("temp.wav")


def process_peaked(bucket, mask_name, blob_list, filepath):

    # Generate notch frequency list
    frequency_list = []
    q_factor = 1
    # Low frequencies
    for i in range(20, 200, 20):
        frequency_list.append(i)

    # Mid frequencies
    for i in range(200, 1000, 100):
        frequency_list.append(i)

    # High frequencies
    for i in range(1000, 18000, 1000):
        frequency_list.append(i)

    # Read wav
    sample_rate, data = read(filepath)

    # Get max amplitude of original audio for scaling later
    original_max_amplitude = max(
        [max(abs(data[:, 0])), max(abs(data[:, 1]))])

    for frequency in frequency_list:

        # Name new blob
        new_blob_name = "sound_therapy/audio/peaked/{}/{}.mp3".format(frequency,
                                                                      mask_name)

        # Get peak-filtered audio
        peaked_left = peak_filtering(
            data[:, 0], sample_rate, frequency, q_factor)
        peaked_right = peak_filtering(
            data[:, 1], sample_rate, frequency, q_factor)

        # Add the peak-filtered audio to the original audio
        combined_audio_left = (peaked_left) + data[:, 0]
        combined_audio_right = (peaked_right) + data[:, 1]

        # Get the max amplitude of the new audio file
        new_max_amplitude = max(
            [max(abs(combined_audio_left)), max(abs(combined_audio_right))])

        # Derive a scale-factor to apply to the new audio
        scale_factor = original_max_amplitude/new_max_amplitude

        joined = np.transpose(
            np.array([combined_audio_left, combined_audio_right]))

        # Apply the scale factor
        scaled_joined = joined * scale_factor

        # Generate an mp3 of the new audio
        write("temp.wav", sample_rate, scaled_joined.astype(np.int32))
        sound = AudioSegment.from_wav("temp.wav")
        sound.export("temp.mp3", format="mp3", bitrate="180")

        print("Mask: " + str(mask_name) + ", Algorithm: Peaked" + ", Frequency: " +
              str(frequency) + ", SF: " + str(scale_factor))

        # Check if blob has previously been uploaded and delete it if found
        check_and_delete(new_blob_name=new_blob_name,
                         blob_list=blob_list, bucket=bucket)

        # Upload new blob
        blob = bucket.blob(new_blob_name)
        blob.upload_from_filename("temp.mp3")

        os.remove("temp.mp3")

        spectrogram_blob_name = "sound_therapy/spectrograms/peaked/{}/{}.json".format(
            frequency, mask_name)

        sample_rate, data = read("temp.wav")

        # Get spectrogram of peaked audio
        s = 1024
        freqs, bins, pxx, = signal.spectrogram(
            data[:, 1], fs=sample_rate, window=signal.blackmanharris(s), nfft=s, noverlap=0, mode='magnitude')

        # Save spectrogram values as a json file
        values = 20 * log10(mean(pxx, axis=1))

        json.dump(np.array(values).tolist(), codecs.open("temp.json", 'w', encoding='utf-8'),
                  separators=(',', ':'), sort_keys=True, indent=4)

        # Check if blob has previously been uploaded and delete it if found
        check_and_delete(new_blob_name=spectrogram_blob_name,
                         blob_list=blob_list, bucket=bucket)

        # Upload new blob
        blob = bucket.blob(spectrogram_blob_name)
        blob.upload_from_filename("temp.json")

        # Delete temporary json
        os.remove("temp.json")
        os.remove("temp.wav")
