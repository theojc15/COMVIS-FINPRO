from flask import Flask, render_template, request, redirect
from flask_caching import Cache
from keras.models import load_model
import numpy as np
import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_bbox_coordinates(handLadmark, image_shape):

    all_x, all_y = [], [] # store all x and y points in list
    for hnd in mp_hands.HandLandmark:
        all_x.append(int(handLadmark.landmark[hnd].x * image_shape[1])) # multiply x by image width
        all_y.append(int(handLadmark.landmark[hnd].y * image_shape[0])) # multiply y by image height

    return min(all_x), min(all_y), max(all_x), max(all_y) # return as (xmin, ymin, xmax, ymax)

PEOPLE_FOLDER = os.path.join('static', 'hands')

app = Flask(__name__)
cache = Cache(app)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

model = load_model("model.h5")
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
    
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    file = request.files["image"]

    image = file.read()
    image = np.frombuffer(image, dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    annotated_image = []

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

        # image = cv2.flip(cv2.imread(file), 1)
        
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # if not results.multi_hand_landmarks:
        #     continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
            
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            xmin, ymin, xmax, ymax = get_bbox_coordinates(hand_landmarks, image.shape)
            
            print (xmin, ymin, xmax, ymax)
            annotated_image = annotated_image[ymin-30:ymax+30, xmin-30:xmax+30]

    image = cv2.resize(annotated_image, (224, 224), cv2.INTER_AREA)
    cv2.imwrite(f'static\hands\image.jpg', image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    prediction = class_names[np.argmax(prediction)]
    with app.app_context():
        cache.clear()
    return redirect("/result?prediction=" + str(prediction))

@app.route("/result")
def result():
   
    prediction = request.args.get("prediction")
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')

    return render_template("result.html", prediction=prediction, image=full_filename)

if __name__ == "__main__":
    app.run()
