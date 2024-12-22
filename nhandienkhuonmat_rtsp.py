import av
import tkinter as tk
from PIL import Image, ImageTk
import threading
import queue
import cv2
import numpy as np
import face_recognition
import os

RTSP_URL = "rtsp://admin:123admin@192.168.1.9:554/onvif1"

class VideoApp:
    def __init__(self, window, window_title, rtsp_url):
        self.window = window
        self.window.title(window_title)
        self.rtsp_url = rtsp_url

        # Tạo hàng đợi để lưu trữ các khung hình
        self.frame_queue = queue.Queue(maxsize=10)

        # Mở RTSP stream bằng PyAV
        try:
            self.container = av.open(self.rtsp_url)
        except av.AVError as e:
            print(f"Không thể mở RTSP stream: {e}")
            exit()

        self.stream = self.container.streams.video[0]

        # Tạo Label để hiển thị video
        self.label = tk.Label(window)
        self.label.pack()

        # Nút Thoát
        self.btn_quit = tk.Button(window, text="Thoát", command=self.quit)
        self.btn_quit.pack()

        # Khởi động luồng đọc video
        self.running = True
        self.thread = threading.Thread(target=self.read_frames, daemon=True)
        self.thread.start()

        # Tải dữ liệu khuôn mặt đã biết
        self.known_face_encodings, self.known_face_names = self.load_known_faces()

        # Bắt đầu cập nhật hình ảnh
        self.update_image()

        self.window.protocol("WM_DELETE_WINDOW", self.quit)
        self.window.mainloop()

    def load_known_faces(self):
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

    def read_frames(self):
        try:
            for frame in self.container.decode(video=0):
                if not self.running:
                    break
                # Chuyển đổi frame sang hình ảnh PIL
                img = frame.to_image()
                # Đặt vào hàng đợi nếu chưa đầy
                if not self.frame_queue.full():
                    self.frame_queue.put(img)
        except av.AVError as e:
            print(f"Lỗi khi đọc khung hình: {e}")
            self.quit()

    def update_image(self):
        try:
            if not self.frame_queue.empty():
                img = self.frame_queue.get()

                # Chuyển đổi PIL Image sang OpenCV Image
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                # Nhận diện khuôn mặt và xác định danh tính
                small_frame = cv2.resize(cv_img, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if matches:
                        best_match_index = int(np.argmin(face_distances))
                        if face_distances[best_match_index] <= 0.6:
                            face_names.append(self.known_face_names[best_match_index].upper())
                        else:
                            face_names.append("Unknown")
                    else:
                        face_names.append("Unknown")

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2
                    # Vẽ hình chữ nhật xung quanh khuôn mặt
                    cv2.rectangle(cv_img, (left, top), (right, bottom), (0, 255, 0), 2)
                    # Hiển thị tên bên dưới khuôn mặt
                    cv2.rectangle(cv_img, (left, bottom), (right, bottom + 35), (0, 255, 0), cv2.FILLED)
                    cv2.putText(cv_img, name, (left + 6, bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

                # Chuyển đổi trở lại hình ảnh PIL
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)))

                # Cập nhật hình ảnh vào Label
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)
        except Exception as e:
            print(f"Lỗi khi cập nhật hình ảnh: {e}")
            self.quit()
        finally:
            # Lên lịch cập nhật hình ảnh sau 30ms (~33 FPS)
            if self.running:
                self.window.after(30, self.update_image)

    def quit(self):
        self.running = False
        self.container.close()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root, "Hiển thị Video RTSP với Nhận Diện Khuôn Mặt", RTSP_URL)
