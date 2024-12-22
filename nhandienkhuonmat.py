import cv2
import face_recognition
import threading
import time
import numpy as np
import os

# Webcam settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# Shared variables
frame = None
processed_frame = None
face_locations = []
face_names = []
lock = threading.Lock()  # Lock for thread-safe operations

# Load known face encodings and names
def load_known_faces():
    path = "captured_samples"
    images = []
    classNames = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                full_path = os.path.join(root, file)
                curImg = cv2.imread(full_path)
                curImg = cv2.resize(curImg, (0, 0), fx=0.5, fy=0.5)
                images.append(curImg)
                person_name = os.path.basename(root)
                classNames.append(person_name)

    encodeList = []
    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(rgb_img)
        if encodes:
            encodeList.append(encodes[0])

    return encodeList, classNames

known_face_encodings, known_face_names = load_known_faces()

# Face recognition thread
def face_recognition_thread():
    global frame, face_locations, face_names, processed_frame

    while True:
        if frame is not None:
            # Copy frame for processing
            with lock:
                local_frame = frame.copy()

            # Resize frame for faster processing
            small_frame = cv2.resize(local_frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Perform face recognition
            local_face_locations = face_recognition.face_locations(rgb_small_frame)
            local_face_encodings = face_recognition.face_encodings(rgb_small_frame, local_face_locations)
            local_face_names = []

            for face_encoding in local_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if matches:
                    best_match_index = int(np.argmin(face_distances))
                    if face_distances[best_match_index] <= 0.6:
                        local_face_names.append(known_face_names[best_match_index].upper())
                    else:
                        local_face_names.append("Unknown")
                else:
                    local_face_names.append("Unknown")

            # Update shared variables with lock
            with lock:
                face_locations = local_face_locations
                face_names = local_face_names
                processed_frame = local_frame

# Start face recognition thread
thread = threading.Thread(target=face_recognition_thread, daemon=True)
thread.start()

# Main loop for reading frames and displaying results
while True:
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Draw rectangles and names on the frame
    with lock:
        if processed_frame is not None:
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom), (right, bottom + 35), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    # Calculate and display FPS
    elapsed_time = time.time() - frame_start
    fps = 1 / elapsed_time if elapsed_time > 0 else 60
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    # Wait to maintain target FPS
    time.sleep(max(1 / 60 - elapsed_time, 0))

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
