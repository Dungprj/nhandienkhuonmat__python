import cv2
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import time

# Function to capture samples
def capture_samples():
    person_name = name_entry.get()
    if not person_name:
        messagebox.showerror("Error", "Please enter a name.")
        return

    # Create a directory for the person if it doesn't exist
    output_dir = os.path.join("captured_samples", person_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sample_count = 0
    max_samples = 10
    last_sample_time = time.time()  # Initialize timestamp for sample interval control

    while sample_count < max_samples:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the captured frame continuously
        cv2.imshow('Capture Samples', frame)

        # Capture and save samples every 1 second
        current_time = time.time()
        if current_time - last_sample_time >= 1:
            sample_filename = os.path.join(output_dir, f"{person_name}_{sample_count:04d}.jpg")
            cv2.imwrite(sample_filename, frame)
            print(f"Đamg lấy mẫu lần {sample_count}")

            sample_count += 1

            # Update progress bar
            progress_bar['value'] = (sample_count / max_samples) * 100
            root.update_idletasks()

            # Update the timestamp
            last_sample_time = current_time

        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if sample_count == max_samples:
        messagebox.showinfo("Info", f"Captured {max_samples} samples for {person_name}.")

# Function to start capturing
def start_capture():
    capture_samples()

# Create GUI window
root = tk.Tk()
root.title("Sample Capture")

tk.Label(root, text="Enter Name:").pack(pady=5)
name_entry = tk.Entry(root)
name_entry.pack(pady=5)

start_button = tk.Button(root, text="Start Capture", command=start_capture)
start_button.pack(pady=10)

# Progress bar for sample capture
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=10)

root.mainloop()
