from tkinter import *
from PIL import Image, ImageTk
import cv2
import imutils
import numpy as np

win = Tk()

win.geometry("700x350")

label = Label(win)
label.grid(row=0, column=0)

recording = False
writer = None

button = Button(
    win,
    text="gravar",
    bg="blue",
    fg="yellow",
)
def on_btn_click(event):
    global recording
    global writer
    recording = not recording
    if recording:
        button.config(bg="red")
        writer = cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
    else:
        button.config(bg="blue")
        writer.release()

button.bind("<Button-1>", on_btn_click)
button.grid(row=0, column=1)

cap = cv2.VideoCapture(0)

facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        roi = imutils.resize(roi_gray, width=224)

    return frame

def show_frames():
    ret, frame = cap.read()

    if ret:
        frame = process_image(frame)
    
    if recording:
        #if not writer:
            #writer = cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))
        writer.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        win.destroy()
        return

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)

    imgtk = ImageTk.PhotoImage(image = img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    # Repeat after an interval to capture continiously
    label.after(20, show_frames)

show_frames()
win.mainloop()
