import math
import time
from cv2 import cv2 as cv
# from google.colab.patches import cv2_imshow

def get_face_box(net, frame, conf_threshold = 0.7):
    frame_open_cv_dnn = frame.copy()
    frame_height = frame_open_cv_dnn.shape[0]
    frame_width = frame_open_cv_dnn.shape[1]
    blob = cv.dnn.blobFromImage(frame_open_cv_dnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frame_open_cv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)
    return frame_open_cv_dnn, bboxes

face_proto = 'opencv_face_detector.pbtxt'
face_model = 'opencv_face_detector_uint8.pb'

age_proto = 'age_deploy.prototxt'
age_model = 'age_net.caffemodel'

gender_proto = 'gender_deploy.prototxt'
gender_model = 'gender_net.caffemodel'

model_mean_values = [78.4263377603, 87.7689143744, 114.895847746]
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(43-48)', '(48-53)', '(53-60)', '(60-100)']
gender_list = ['Male', 'Female']

# Load network
age_net = cv.dnn.readNet(age_model, age_proto)
gender_net = cv.dnn.readNet(gender_model, gender_proto)
face_net = cv.dnn.readNet(face_model, face_proto)

padding = 20

def age_gender_detect(frame):
    t = time.time()
    frame_face, bboxes = get_face_box(face_net, frame)
    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1), max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        label = '{} {}'.format(gender, age)
        cv.putText(frame_face, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    return frame_face

# input = cv.imread('1.jpg')
# output = age_gender_detect(input)
# cv_imshow(output)

cap = cv.VideoCapture(0)
ret, frame = cap.read()
frame_height, frame_width, _ = frame.shape
out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
print('Processing video.....')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        out.release()
        break
    output = age_gender_detect(frame)
    out.write(output)
output.release()
print('Done')