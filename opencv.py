import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load trained model
cnn = load_model(r"D:\ML PROJECTS\Multi class skin disease classification\model\best_skin_model.h5")

# IMPORTANT: class order must match training_set.class_indices
classes = ['acne', 'eksim', 'herpes', 'panu', 'rosacea']

# open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # resize to model size (use SAME size used in training)
    img = cv2.resize(frame, (128,128))   # change to 128x128 if you trained with 128

    # preprocess
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # predict
    prediction = cnn.predict(img_array, verbose=0)

    # get class index
    class_index = np.argmax(prediction)

    # get class name
    label = classes[class_index]

    # get confidence
    confidence = np.max(prediction) * 100

    text = f"{label} ({confidence:.1f}%)"

    # show label on video
    cv2.putText(frame, text, (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Skin Disease Detection", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
