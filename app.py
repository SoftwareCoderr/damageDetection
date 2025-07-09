from flask import Flask, render_template, request
import os
import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from collections import Counter

# Flask uygulamasını başlat
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded'

# Modeli yükle
model = load_model('../model/vgg16_car_damage_multilabel.h5')

# Etiketler ve Türkçe karşılıklar
label_translations = {
    'door_dent': 'Kapı Çökmesi',
    'door_scratch': 'Kapı Çizik',
    'bumper_dent': 'Tampon Çökmesi',
    'bumper_scratch': 'Tampon Çizik',
    'head_lamp': 'Ön Far Kırılması',
    'tail_lamp': 'Arka Far Kırılması',
    'glass_shatter': 'Cam Kırılması',
    'unknown': 'Hasar Yok'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            # Dosyayı kaydet
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            ext = os.path.splitext(filename)[1].lower()

            if ext in ['.jpg', '.jpeg', '.png']:
                # Fotoğraf işle
                try:
                    img = image.load_img(filepath, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    img_array = img_array / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    prediction = model.predict(img_array)[0]

                    # Tahmin edilen etiketler
                    threshold = 0.18
                    predicted_labels = [label for idx, label in enumerate(label_translations.keys()) if prediction[idx] > threshold]

                    # Eğer başka hasar varsa unknown'u kaldır
                    if 'unknown' in predicted_labels and len(predicted_labels) > 1:
                        predicted_labels.remove('unknown')

                    predicted_labels_tr = [label_translations.get(label, label) for label in predicted_labels]

                    if not predicted_labels_tr:
                        predicted_labels_tr = ["Hasar Yok"]

                    return render_template('index.html', prediction=predicted_labels_tr, image=filename, is_video=False)

                except Exception as e:
                    return f"Hata oluştu (Fotoğraf): {str(e)}"

            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                # Video işle
                try:
                    cap = cv2.VideoCapture(filepath)

                    if not cap.isOpened():
                        return "Video açma hatası!"

                    frame_predictions = []
                    frame_count = 0
                    success, frame = cap.read()

                    while success:
                        frame_count += 1
                        if frame_count % 2 == 0:  # Her 2. frame'de bir analiz yapıyoruz
                            frame_resized = cv2.resize(frame, (224, 224))
                            img_array = image.img_to_array(frame_resized)
                            img_array = img_array / 255.0
                            img_array = np.expand_dims(img_array, axis=0)

                            prediction = model.predict(img_array)[0]
                            predicted_labels = [label for idx, label in enumerate(label_translations.keys()) if prediction[idx] > 0.15]

                            frame_predictions.extend(predicted_labels)

                        success, frame = cap.read()

                    cap.release()

                    if frame_predictions:
                        final_labels = list(set(frame_predictions))

                        # unknown varsa ve başka hasar da varsa, unknown'u çıkar
                        if 'unknown' in final_labels and len(final_labels) > 1:
                            final_labels.remove('unknown')

                        final_labels_tr = [label_translations.get(label, label) for label in final_labels]
                        if not final_labels_tr:
                            final_labels_tr = ["Hasar Yok"]
                    else:
                        final_labels_tr = ["Hasar Yok"]

                    return render_template('index.html', prediction=final_labels_tr, video=filename, is_video=True)

                except Exception as e:
                    return f"Hata oluştu (Video): {str(e)}"

            else:
                return "Desteklenmeyen dosya türü!"

    return render_template('index.html')

if __name__ == '__main__':
       app.run(debug=True)