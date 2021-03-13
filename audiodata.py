#!/usr/bin/env python3
#https://www.tensorflow.org/io/tutorials/audio?hl=en
#tensorflow_io/docs/tutorials/audio.ipynb

"""
Overview
One of the biggest challanges in Automatic Speech Recognition is the
preparation and augmentation of audio data. Audio data analysis could be in
time or frequency domain, which adds additional complex compared with other
data sources such as images.

As a part of the TensorFlow ecosystem, tensorflow-io package provides quite a
few useful audio-related APIs that helps easing the preparation and
augmentation of audio data.

Setup::

    docker pull tfsigio/tfio:nightly
    docker run -it --rm --name tfio-nightly tfsigio/tfio:nightly
    # directly places you into Python 3.7 prompt: exit()
    docker run -it --rm --name tfio-nightly tfsigio/tfio:nightly /bin/bash

    # but instead for GUI
    docker run -it \
        --user=0 \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --name tfio-nightly tfsigio/tfio:nightly /bin/bash
    exit
    docker ps -a
    export containerID=$(docker ps -l -q)
    echo "$containerID"
    exit
    xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerID`
    docker start $containerID
    docker ps

    docker exec -it tfio-nightly /bin/bash
    apt-get update
    yes | apt-get install libx11-dev
    yes | apt-get install tk
    yes | apt-get install gcc
    yes | apt-get install alsalib
    yes | apt-get install libasound2-dev
    pip install matplotlib
    pip install seaborn
    pip install ipython
    pip install simpleaudio

    python
    # from tensorflow.keras.layers.experimental import preprocessing
    # ... Illegal instruction (core dumped)
    # python gone
    exit()

    exit
    docker stop $containerID
    docker rm $containerID
    docker ps -a



Usage
Read an Audio File
In TensorFlow IO, class tfio.audio.AudioIOTensor allows you to read an audio
file into a lazy-loaded IOTensor:
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import simpleaudio

# core-dump on any of next lines
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_io as tfio
from tensorflow.keras import layers
from tensorflow.keras import models

audio = tfio.audio.AudioIOTensor('gs://cloud-samples-tests/speech/brooklyn.flac')
print(audio)

# In the above example, the Flac file `brooklyn.flac` is from a publicly
# accessible audio clip in
# [google cloud](https://cloud.google.com/speech-to-text/docs/quickstart-gcloud).
# 
# The GCS address `gs://cloud-samples-tests/speech/brooklyn.flac` are used
# directly because GCS is a supported file system in TensorFlow. In addition to
# `Flac` format, `WAV`, `Ogg`, `MP3`, and `MP4A` are also supported by
# `AudioIOTensor` with automatic file format detection.
# 
# `AudioIOTensor` is lazy-loaded so only shape, dtype, and sample rate are shown
# initially. The shape of the `AudioIOTensor` is represented as `[samples, channels]`,
# which means the audio clip you loaded is mono channel with `28979` samples in `int16`.

# The content of the audio clip will only be read as needed, either by converting
# `AudioIOTensor` to `Tensor` through `to_tensor()`, or though slicing. Slicing
# is especially useful when only a small portion of a large audio clip is needed:


audio_slice = audio[100:]
# remove last dimension
audio_tensor = tf.squeeze(audio_slice, axis=[-1])
print(audio_tensor)

# The audio can be played through:

da = display.Audio(waveform,rate=16000)
simpleaudio.play_buffer(da.data,1,2,16000)

#It is more convinient to convert tensor into float numbers and show the audio clip in graph:

tensor = tf.cast(audio_tensor, tf.float32) / 32768.0
plt.figure()
plt.plot(tensor.numpy())
plt.show()

### Trim the noise
# Sometimes it makes sense to trim the noise from the audio, which could be done
# through API `tfio.experimental.audio.trim`. Returned from the API is a pair of
# `[start, stop]` position of the segement:

position = tfio.experimental.audio.trim(tensor, axis=0, epsilon=0.1)
print(position)

start = position[0]
stop = position[1]
print(start, stop)

processed = tensor[start:stop]

plt.figure()
plt.plot(processed.numpy())
plt.show()

### Fade In and Fade Out
# One useful audio engineering technique is fade, which gradually increases or
# decreases audio signals. This can be done through
# `tfio.experimental.audio.fade`. `tfio.experimental.audio.fade` supports
# different shapes of fades such as `linear`, `logarithmic`, or `exponential`:

fade = tfio.experimental.audio.fade(
    processed, fade_in=1000, fade_out=2000, mode="logarithmic")

plt.figure()
plt.plot(fade.numpy())
plt.show()

### Spectrogram
# Advanced audio processing often works on frequency changes over time. In
# `tensorflow-io` a waveform can be converted to spectrogram through
# `tfio.experimental.audio.spectrogram`:

# Convert to spectrogram
spectrogram = tfio.experimental.audio.spectrogram(
    fade, nfft=512, window=512, stride=256)

plt.figure()
plt.imshow(tf.math.log(spectrogram).numpy())

# Additional transformation to different scales are also possible:

# Convert to mel-spectrogram
mel_spectrogram = tfio.experimental.audio.melscale(
    spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)


plt.figure()
plt.imshow(tf.math.log(mel_spectrogram).numpy())

# Convert to db scale mel-spectrogram
dbscale_mel_spectrogram = tfio.experimental.audio.dbscale(
    mel_spectrogram, top_db=80)

plt.figure()
plt.imshow(dbscale_mel_spectrogram.numpy())

### SpecAugment
# In addition to the above mentioned data preparation and augmentation APIs,
# `tensorflow-io` package also provides advanced spectrogram augmentations, most
# notably Frequency and Time Masking discussed in
# [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition (Park et al., 2019)]
# (https://arxiv.org/pdf/1904.08779.pdf).

#### Frequency Masking
# In frequency masking, frequency channels 
# `[f0, f0 + f)` are masked where `f` is chosen from a uniform distribution
# from `0` to the frequency mask parameter `F`, and `f0` is chosen
# from `(0, ν − f)` where `ν` is the number of frequency channels.

# Freq masking
freq_mask = tfio.experimental.audio.freq_mask(dbscale_mel_spectrogram, param=10)

plt.figure()
plt.imshow(freq_mask.numpy())

#### Time Masking
# In time masking, `t` consecutive time steps
# `[t0, t0 + t)` are masked where `t` is chosen from a uniform
# distribution from `0` to the time mask parameter `T`, and `t0` is
# chosen from `[0, τ − t)` where `τ` is the time steps.

# Time masking
time_mask = tfio.experimental.audio.time_mask(dbscale_mel_spectrogram, param=10)

plt.figure()
plt.imshow(time_mask.numpy())

