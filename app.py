from flask import Flask, render_template, request
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import re

app = Flask(__name__)
app.config.from_object('config')

tesseract_path = r"static\models\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path
    

@app.route('/')
def index():
    return render_template('index.html',result='')

@app.route('/upload', methods=['POST'])
def about():

    inputs = request.files.get('input_img')
    image_stream = inputs.read()
    nparr = np.frombuffer(image_stream, np.uint8)
    inputs = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    model = YOLO('static\models\plate_detecor.pt')
    results = model(inputs)  
    plates = []
    for result in results:
        classes = result.boxes.cls
        rects = result.boxes.xyxy.numpy()
        license_rects = []
        for rect_id in range(len(rects)):
            if int(classes[rect_id]) == 0:
                license_rects.append([int(i) for i in rects[rect_id]])
            for rect in license_rects:
                plates.append(inputs[rect[1]:rect[3], rect[0]:rect[2]].copy())

    print("Plates discovered")
    plate_numbers= []
    for plate in plates:
        background = np.array([[[255,255,255] for _ in range(640)] for _ in range(640)])
        for i in range(plate.shape[0]):
            for j in range(plate.shape[1]):
                background[i,j] = plate[i,j,:]
        #background = cv2.imread('static\img\download.jpg')
        gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        

        kernel = np.ones((2, 1), np.uint8)
        img = cv2.erode(gray_plate, kernel, iterations=1)
        img = cv2.dilate(img, kernel, iterations=1)
        plate_number = pytesseract.image_to_string(img)
        plate_numbers.append(re.sub(r'[^A-Z0-9]', '', plate_number))
        
    plate_numbers = set(plate_numbers)
    for plate_number in plate_numbers:
        print("This image contains a plate with number: ",plate_number)

    return render_template('index.html',result = "This image contains a plate with number\n"+'\n'.join(plate_numbers) )

if __name__ == '__main__':
    
    app.run()

