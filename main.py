import logging as log
log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="log.log",
    filemode="w",
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
import threading as th
import serial

CHUNK = 1024
SAMPLE_RATE = 44100
DURATION = 2
N_MFCC = 13

# Инициализация переменных до их использования
ser = None
client = None
client2 = None
server = None
scan = None
com = None
ip = None

try:
    ser = serial.Serial(
        port='/dev/ttyUSB0',
        baudrate=9600,
        timeout=1
    )
except Exception as e:
    log.error(f"Ошибка инициализации serial: {e}")

def record_audio(filename, duration, samplerate, channels):
    """Записывает аудио в WAV-файл."""
    print("Начинаю запись...")
    try:
        myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, device=1)
        sd.wait()
        sf.write(filename, myrecording, samplerate)
    except Exception as e:
        log.error(f"Ошибка записи аудио: {e}")

status2 = True

def camera():
    """Функция для работы с камерой"""
    global client2, status2
    while status2 and client2:
        try:
            msg = client2.recv(1024).decode()
            print("HEllo")
            if ser:
                ser.write(msg.encode())
        except Exception as e:
            log.error(f"Ошибка в функции camera: {e}")
            break

log.info("Загружаю нейросеть")

try:
    model = tf.keras.models.load_model("drone_detector.keras")
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    log.info("Нейросеть загружена создаю сервер передачи")
except Exception as e:
    log.error(f"Ошибка загрузки модели: {e}")
    exit(1)

try:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    scan = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    com = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    import psutil
    
    # Получаем информацию о сетевых интерфейсах
    addrs = psutil.net_if_addrs()
    ip = "localhost"  # значение по умолчанию
    
    for interface, addresses in addrs.items():
        for addr in addresses:
            if addr.family == socket.AF_INET and interface == "wlan0":
                ip = addr.address
                break
    
    print(f"Ip address = {ip}")
    log.info(f"Ip address = {ip}")
    
    scan.bind((ip, 1113))
    scan.listen()
    
    server.bind((ip, 1111))
    server.listen()
    
    com.bind((ip, 1112))
    com.listen()
    
    client, address = server.accept()
    log.info("Первый клиент подключился")
    
    client2, address2 = com.accept()
    log.info("Второй клиент подключился")
    
except Exception as e:
    log.error(f"Ошибка настройки сети: {e}")
    exit(1)

def extract_mfcc(file_path):
    """Извлечение MFCC-признаков из аудиофайла"""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        return mfccs.T
    except Exception as e:
        log.error(f"Ошибка извлечения MFCC: {e}")
        return None

def predict_drone(audio_file):
    """Предсказание наличия дрона"""
    try:
        mfccs = extract_mfcc(audio_file)
        if mfccs is None:
            return 'Ошибка'
        prediction = model.predict(np.expand_dims(mfccs, axis=0))
        if prediction[0][0] > 0.5:
            return 'Дрон'
        else:
            return 'Не дрон'
    except Exception as e:
        log.error(f"Ошибка предсказания: {e}")
        return 'Ошибка'

try:
    p = round(audio.db(), 3)
    print(p)
    c = 0
    status = False
    
    # Запускаем поток работы с камерой только если client2 инициализирован
    if client2:
        camt = th.Thread(target=camera)
        camt.daemon = True  # Делаем поток демоном для автоматического завершения
        camt.start()
    
    while True:
        try:
            db = round(audio.dbr(), 3)
            print(f"{db} {c} {p} {status} {db - p}")
            
            if (db - p) > 0 and c >= 2:
                a = "rec.wav"
                record_audio(a, DURATION, SAMPLE_RATE, 1)
                result = predict_drone(a)
                
                try:
                    os.remove(a)
                except:
                    pass
                
                print(result)
                status = False
                
                if client:
                    if result == "Дрон":
                        client.send("1".encode())
                    else:
                        client.send("0".encode())
                        
            elif (db - p) > 0 and c >= 0 and c < 2:
                c += 1
            elif not status and ((db - p) < 0 or c != 0):
                status = True
                if client:
                    client.send("0".encode())
                c = 0
                
        except Exception as e:
            log.error(f"Ошибка в основном цикле: {e}")
            break
            
except Exception as e:
    log.critical(f"Критическая ошибка: {e}")
except KeyboardInterrupt:
    log.info("Программа завершена пользователем")
finally:
    # Корректное закрытие ресурсов
    status2 = False
    
    if server:
        server.close()
    if com:
        com.close()
    if client:
        client.close()
    if client2:
        client2.close()
    if scan:
        scan.close()
    if ser:
        ser.close()
