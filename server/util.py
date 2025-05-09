import pickle
import cv2
import pywt
import numpy as np
import base64
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

with open('assets/celeb_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)


def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255

    # Get coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # remove approximation (blur)

    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

def read_base64_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return img

def get_cropped_image_if_2_eyes(image_path):
    img = read_base64_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
        
def get_process_image(image_path):
    img = image_path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Encode the image to base64
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode()

    return jpg_as_text
        

def classify_image(base64_image):
    cropped_img = get_cropped_image_if_2_eyes(base64_image)

    x = []
    
    if cropped_img is not None:
        processed_img = get_process_image(cropped_img)
        scalled_raw_img = cv2.resize(cropped_img, (32, 32))
        img_har = w2d(cropped_img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))

        combined_img = np.vstack((
            scalled_raw_img.reshape(32*32*3, 1),
            scalled_img_har.reshape(32*32, 1)
        ))
        x.append(combined_img)

        x = np.array(x).reshape(len(x),4096).astype(float)

        predicted_class = model.predict(x)[0]
        probability = model.predict_proba(x)[0][predicted_class] * 100

        return predicted_class, probability, processed_img
    else:
        return ("No face detected or eyes not detected correctly.", 0, 0)

if __name__ == "__main__":
    # Sample image testing
    
    result = classify_image("data:image/jpeg;base64,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")
    print(result)