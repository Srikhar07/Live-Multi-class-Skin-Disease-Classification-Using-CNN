from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_skin_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)


classes = ['acne', 'eksim', 'herpes', 'panu', 'rosacea']

IMG_SIZE = 128 


# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    return img


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------------- IMAGE UPLOAD PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    name = request.form.get("user_name")
    age = request.form.get("age")
    phone = request.form.get("phone")
    email = request.form.get("email")

    file = request.files["image"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    img = preprocess_image(file_path)
    pred = model.predict(img)[0]

    class_index = np.argmax(pred)
    prediction = classes[class_index]
    confidence = float(pred[class_index]) * 100

    return render_template(
        "result.html",
        name=name,
        age=age,
        phone=phone,
        email=email,
        prediction=prediction,
        confidence=round(confidence, 2),
        image_path="static/uploads/" + file.filename
    )


# ---------------- LIVE WEBCAM PREDICTION ----------------
@app.route("/live_prediction", methods=["POST"])
def live_prediction():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

        pred = model.predict(img)[0]
        class_index = np.argmax(pred)
        label = classes[class_index]
        confidence = pred[class_index] * 100

        text = f"{label} ({confidence:.2f}%)"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Live Prediction - Press Q to Exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return redirect(url_for("home"))


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run()