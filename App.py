import cv2
import os
from datetime import datetime
import time
timer = time.time()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
fps_start_time, fps, prev_fps = 0, 0, 0
first_frame, out = None, None
face_dict, face_id_counter, face_num = {}, 1, 0
if not os.path.exists("detected_faces"): os.makedirs("detected_faces")
if not os.path.exists("detected_movement"): os.makedirs("detected_movement")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
while True:
    ret, frame = cap.read()
    fps_end_time = cv2.getTickCount()
    time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
    fps = 1.0 / time_diff
    fps_start_time = fps_end_time
    if time.time() - timer > 0.5: prev_fps, timer = fps, time.time()
    cv2.putText(frame, "FPS: {:.1f}".format(prev_fps), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
    cv2.putText(frame, "Unique Faces: {}".format(face_num), (10, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if first_frame is None:
        first_frame = gray
        continue
    frame_diff = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 1000: continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        unique_face = True
        for key, value in face_dict.items():
            if abs(x - value[0][0]) < 50 and abs(y - value[0][1]) < 50:
                unique_face = False
                break
        if unique_face:
            face_num += 1
            label = "Face {}".format(face_id_counter)
            face_dict[label] = [(x, y), (x+w, y+h)]
            face_id_counter += 1
            img_path = os.path.join("detected_faces", label + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg")
            try: cv2.imwrite(img_path, frame[y-50:y+h+50, x-50:x+w+50])
            except: cv2.imwrite(img_path, frame[y:y+h, x:x+w])
        else:
            for key, value in face_dict.items():
                if abs(x - value[0][0]) < 50 and abs(y - value[0][1]) < 50:
                    label = key
                    break
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    if out is None:
        filename = "detected_movement/{}.avi".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        out = cv2.VideoWriter(filename, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
    out.write(frame)
    cv2.imshow('Video Stream', frame)
    first_frame = gray
    if cv2.waitKey(1) == ord('q'): break
cap.release()
cv2.destroyAllWindows()
