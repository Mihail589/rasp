import sounddevice as sd
import numpy as np
import soundfile as sf

samplerate = 44100  # Частота дискретизации
duration = 5  # Длительность записи в секундах
channels = 1  # Количество каналов (моно)
def db():
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()  # Ждем завершения записи
    data = recording.flatten() # Преобразуем в одномерный массив
    rms = np.sqrt(np.mean(data**2))
    db = 20 * np.log10(rms / 32767)
    dbp = db -(db / 100 * 1)
    return dbp
def dbr():
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()  # Ждем завершения записи
    data = recording.flatten() # Преобразуем в одномерный массив
    rms = np.sqrt(np.mean(data**2))
    db = 20 * np.log10(rms / 32767)
    return db