import logging as log
log.basicConfig(
    level=log.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="log.log",  # Запись в файл
    filemode="w",  # "w" - перезапись, "a" - добавление в файл
    encoding="UTF-8"
)
import tensorflow as tf
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import socket
import audio
import os

CHUNK = 1024
SAMPLE_RATE = 22050  # Частота дискретизации аудио
DURATION = 2  # Длительность аудиофрагмента в секундах
N_MFCC = 13  # Количество MFCC-коэффициент

def record_audio(filename, duration, samplerate, channels):
    """Записывает аудио в WAV-файл.
    Args:
        filename: Имя файла для сохранения записи (по умолчанию "recording.wav").
        duration: Длительность записи в секундах (по умолчанию 5 секунд).
        samplerate: Частота дискретизации (по умолчанию 44100 Гц).
        channels: Количество каналов (1 - моно, 2 - стерео, по умолчанию 1).
    """
    print("Начинаю запись...")
    myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
    sd.wait()  # Wait until recording is finished
    sf.write(filename, myrecording, samplerate)

log.info("Загружаю неиросеть")

model = tf.keras.models.load_model("drone_detector.keras")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
log.info("Ниросеть загружена создаю сервер передачи")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
import psutil

# Получаем информацию о сетевых интерфейсах
addrs = psutil.net_if_addrs()

for interface, addresses in addrs.items():
    for addr in addresses:
        if addr.family == socket.AF_INET and interface == "Беспроводная сеть":  # Проверяем, что это IPv4
            ip = addr.address
print(f"Ip address = {ip}")
log.info(f"Ip address = {ip}")
server.bind((ip, 1111))

server.listen()

client, address = server.accept()
log.info("подключились")
def extract_mfcc(file_path):
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        return mfccs.T

def predict_drone(audio_file):
    mfccs = extract_mfcc(audio_file)
    prediction = model.predict(np.expand_dims(mfccs, axis=0))
    if prediction[0][0] > 0.5:
        return 'Дрон'
    else:
        return 'Не дрон'
    
p = round(audio.db(), 3)
print(p)
c = 0
status = False
try:
     while True:
        db = round(audio.dbr(), 3)
        print(f"{db} {c} {p} {status} {db - p}")
        if (db - p) > 0 and c >= 2:
            a = "rec.wav"
            record_audio(a, DURATION, SAMPLE_RATE, 1)
            result = predict_drone(a)
            os.remove(a)
            print(result)
            status = False
            if result == "Дрон":
               client.send("1".encode())
            else:
                client.send("0".encode())
        elif (db - p) > 0 and c >= 0 and c < 2:
            c += 1
        elif not status and (db - p) < 0 or c != 0:
            status = True
            client.send("0".encode())
            c = 0

except Exception as e:
     log.critical(e)
except KeyboardInterrupt as e:
    log.critical(e)
    server.close()
    client.close()