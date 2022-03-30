import os
import csv
import tkinter as tk
from datetime import datetime, timedelta
from tkVideoPlayer import TkinterVideo
from tkinter import filedialog

import cv2
import imutils
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input


recording = False
writer = None
external_window_opened = False
test_session_name = ''
count = 0

facecasc = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
classificator = load_model('model.hdf5')
labels = {0: 'Raiva', 1: 'Nojo', 2: 'Medo', 3: 'Felicidade', 4: 'Neutro', 5: 'Tristeza', 6: 'Surpresa'}


def most_frequent(List):
    return max(set(List), key = List.count)

def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    five_last_predictions = [] 
    detected_class_label = ''

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        roi = imutils.resize(roi_gray, width=224)

        try:
            roi = np.expand_dims(roi, axis=0)
            roi = preprocess_input(roi)
            features = classificator.predict(roi)
            label_index = np.argmax(features)
            detected_class_label = labels[label_index]

            five_last_predictions.append(detected_class_label)
            if len(five_last_predictions) > 5:
                five_last_predictions.pop(0)

            detected_class_label = most_frequent(five_last_predictions)
        except:
            print("deu erro")
            detected_class_label = five_last_predictions[-1] if five_last_predictions else ''

        cv2.putText(frame, detected_class_label, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # by now, only one face will be detected
        break

    return frame, detected_class_label


def show_frames(cap, label, *, enable_recording=False, writer=None, csv_writer=None):
    global count
    count = 0
    def inner_func():
        global count
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ret, frame = cap.read()
        count += 1

        if ret:
            frame, detected_class_label = process_image(frame)

            if enable_recording and recording:
                writer.write(frame)
            if detected_class_label and recording: csv_writer.writerow([detected_class_label])

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)

            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)

            # Repeat after an interval to capture continiously
            label.after(int(fps), inner_func)
    inner_func()


def writer_factory(session_name, width, height, fps):
    now = datetime.now()
    str_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    folder_name = os.path.join('sessions', f"{session_name}-{str_now}")
    os.mkdir(folder_name)

    file_path = os.path.join(folder_name, f"{session_name}.mp4")
    csv_path = os.path.join(folder_name, f"{session_name}.csv")
    video_writer = cv2.VideoWriter(
        file_path,
        cv2.VideoWriter_fourcc(*"DIVX"),
        fps,
        (width, height),
    )
    csv_file = open(csv_path, 'w')
    csv_writer = csv.writer(csv_file)
    return video_writer, csv_writer, csv_file


class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

    def show(self):
        self.lift()


class MainPage(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        label = tk.Label(self, text="Detector de emoções em teste de software")
        label.pack()


class ListTestsPage(Page):
    def __init__(self, *args, open_test_callback, **kwargs):
        Page.__init__(self, *args, **kwargs)
        # label = tk.Label(self, text="List Tests")
        # label.pack(side="top", fill="both", expand=True)

        # Colocar dentro de um frame e dar refresh nele
        refresh_btn = tk.Button(self, text="Atualizar", command=self.refresh)
        refresh_btn.pack()

        self.test_sessions = os.listdir('sessions')

        def callback_factory(v_path, c_path):
            def _callback():
                open_test_callback(v_path, c_path)
            return _callback

        for i, session in enumerate(self.test_sessions):
            files = [files for (_, _, files) in os.walk(os.path.join('sessions',session))][0]
            video_path = [path for path in files if path.find('.mp4') >= 0][0]
            video_path = os.path.join('sessions', session, video_path)
            csv_path = [path for path in files if path.find('.csv') >= 0][0]
            csv_path = os.path.join('sessions', session, csv_path)

            btn = tk.Button(self, text=session, command=callback_factory(video_path, csv_path))
            btn.pack()
    
    def show(self):
        self.lift()

    def refresh(self):
        self.destroy()
        self.__init__()


class CreateTestPage(Page):
    def __init__(self, *args, main_page, open_record_view, **kwargs):
        Page.__init__(self, *args, **kwargs)

        label = tk.Label(self, text="Criar nova sessão de teste")
        label.pack()

        self.entry = tk.Entry(self)
        self.entry.pack(fill=tk.X)

        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X)
        self.back_button = tk.Button(btn_frame, text="Voltar", command=main_page.show)
        self.back_button.pack(fill=tk.Y, side=tk.LEFT)

        def handle_open_record_view():
            global test_session_name
            new_test_session_name = self.entry.get()

            if new_test_session_name:
                test_session_name = new_test_session_name
                open_record_view()
                main_page.show()

        create_button = tk.Button(
            btn_frame, text="Iniciar", command=handle_open_record_view
        )
        create_button.pack(fill=tk.Y, side=tk.RIGHT)

    def set_back_button_go_page(self, page):
        self.back_button.bind("<Button-1>", lambda e: page.show)

    def show(self):
        self.entry.delete(0, tk.END)
        self.lift()


class CreateTestSessionView(tk.Frame):
    def __init__(self, *args, cap, handle_close, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        label = tk.Label(self)
        label.pack()
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        writer, csv_writer, csv_file = writer_factory(test_session_name, width, height, fps)


        button = tk.Button(
            self,
            text="gravar",
            bg="blue",
            fg="yellow",
        )

        def handle_back_btn():
            cap.release()
            handle_close()

        def on_btn_click(event):
            global recording
            recording = not recording
            if recording:
                button.config(bg="red", text="gravando")
            else:
                button.config(bg="blue", text="gravar")
                csv_file.close()
                writer.release()
                handle_back_btn()

        button.bind("<Button-1>", on_btn_click)
        button.pack()

        show_frames(cap, label, enable_recording=True, writer=writer, csv_writer=csv_writer)


class LiveTestView(tk.Frame):
    def __init__(self, *args, cap, handle_close, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        label = tk.Label(self)
        label.pack()

        def handle_back_btn():
            cap.release()
            handle_close()
            
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X)
        back_button = tk.Button(btn_frame, text="Voltar", command=handle_back_btn)
        back_button.pack()
        show_frames(cap, label)


class DetailTestSessionView(tk.Frame):
    def __init__(self, *args, video_path, csv_path, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        def update_duration(event):
            """ updates the duration after finding the duration """
            end_time["text"] = str(timedelta(seconds=vid_player.duration()))
        def seek(value):
            """ used to seek a specific timeframe """
            vid_player.seek(int(value))


        def skip(value: int):
            """ skip seconds """
            vid_player.skip_sec(value)

        def play_pause():
            """ pauses and plays """
            if vid_player.is_paused():
                vid_player.play()
                play_pause_btn["text"] = "Pause"

            else:
                vid_player.pause()
                play_pause_btn["text"] = "Play"


        def video_ended(event):
            """ handle video ended """
            play_pause_btn["text"] = "Play"

        df = pd.read_csv(csv_path, header=None, names=['emotion'])

        distributed_emotions = df['emotion'].value_counts().reset_index()
        distributed_emotions.columns = ['emotion', 'count']
        distributed_emotions = distributed_emotions.values

        emotions_count = len(df.index)

        most_frequent_emotion = df['emotion'].mode().values[0]


        vid_player = TkinterVideo(scaled=True, pre_load=True, master=self)
        vid_player.pack(expand=True, fill="both")

        play_pause_btn = tk.Button(self, text="Play", command=play_pause)
        play_pause_btn.pack()

        vid_player.bind("<<Duration>>", update_duration)
        vid_player.bind("<<Ended>>", video_ended )

        # load video
        folder_path = os.path.dirname(os.path.abspath(video_path))
        vid_player.load(video_path)
        play_pause_btn["text"] = "Play"

        from subprocess import call
        tk.Label(self).pack()
        btn_open_folder = tk.Button(self, text='Abrir pasta', command=lambda: call(["dolphin", folder_path]))
        btn_open_folder.pack()
        tk.Label(self, text=f"Emoção mais frequente: {most_frequent_emotion}", font=("Arial", 25)).pack()
        tk.Label(self, text="Distribuição percentual das emoções", font=("Arial", 22)).pack()

        for label, count in distributed_emotions:
            label_percentage = count * 100 /emotions_count
            tk.Label(self, text=f"\t{label}: {label_percentage:.2f}%", font=("Arial", 22)).pack()


class MainView(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        main_page = MainPage(self)

        def openNewTestingSessionWindow():
            global external_window_opened
            external_window_opened = True
            newWindow = tk.Toplevel(self)
            newWindow.title("Classificação em tempo real - com gravação")
            newWindow.geometry("720x600")
            cap = cv2.VideoCapture(0)
            
            def disable_event():
                external_window_opened = False
                cap.release()
                newWindow.destroy()

            def handle_close():
                external_window_opened = False
                newWindow.destroy()

            newWindow.protocol("WM_DELETE_WINDOW", disable_event)
            CreateTestSessionView(newWindow, cap=cap, handle_close=handle_close).pack(side="top", fill="both", expand=True)

        def openDetailTestSessionWindow(video_path, csv_path):
            global external_window_opened
            external_window_opened = True
            newWindow = tk.Toplevel(self)
            newWindow.title("Sessão de teste gravada")
            newWindow.geometry("720x600")
            print('OPEN VIEW: ', video_path)
            DetailTestSessionView(newWindow, video_path=video_path, csv_path=csv_path).pack(side="top", fill="both", expand=True)

        list_tests_page = ListTestsPage(self, open_test_callback=openDetailTestSessionWindow)
        create_test_page = CreateTestPage(
            self, main_page=main_page, open_record_view=openNewTestingSessionWindow
        )

        create_test_page.set_back_button_go_page(main_page)

        buttonframe = tk.Frame(self)
        container = tk.Frame(self)
        buttonframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        main_page.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        list_tests_page.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        create_test_page.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        def openLiveTestingWindow():
            global external_window_opened
            external_window_opened = True
            newWindow = tk.Toplevel(self)
            newWindow.title("Classificação em tempo real - Sem gravação")
            newWindow.geometry("720x540")
            cap = cv2.VideoCapture(0)
            
            def disable_event():
                external_window_opened = False
                cap.release()
                newWindow.destroy()

            def handle_close():
                external_window_opened = False
                newWindow.destroy()

            newWindow.protocol("WM_DELETE_WINDOW", disable_event)
            LiveTestView(newWindow, cap=cap, handle_close=handle_close).pack(side="top", fill="both", expand=True)


        create_test_btn = tk.Button(
            buttonframe,
            text="Criar nova sessão de teste",
            command=create_test_page.show,
        )
        create_test_btn.pack(side="left")

        list_tests_btn = tk.Button(
            buttonframe, text="Ver sessões de teste", command=list_tests_page.show
        )
        list_tests_btn.pack(side="left")

        live_test_btn = tk.Button(
            buttonframe, text="Executar classificação sem gravação", command=openLiveTestingWindow
        )
        live_test_btn.pack(side="left")

        main_page.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Detector de emoções faciais")
    main = MainView(root)
    main.pack(side="top", fill="both", expand=True)
    root.wm_geometry("720x500")
    root.mainloop()
