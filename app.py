import pushup
import pullup
import situp
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from tkinter import Button
from tkinter import OptionMenu
from tkinter import Tk
from tkinter import Label
from tkinter import PhotoImage
from tkinter import StringVar
from tkinter import CENTER
from PIL import Image
from PIL import ImageTk
from sklearn.preprocessing import LabelEncoder
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

class MainWindow(object):
    def __init__(self):
        # initialize window
        self.root = Tk()
        self.root.geometry('600x550')
        self.root.title('Sportstracker')
        self.panel = Label(self.root)
        self.panel.pack()

        # icon
        icon = PhotoImage(file='./image/icon_root.png')
        self.root.iconphoto(False, icon)

        # initialize dropdown
        self.selected_opt = StringVar()
        self.selected_opt.set('pushup')
        self.drop_menu = OptionMenu(self.root, self.selected_opt, "pushup", "situp", "pullup")
        self.drop_menu.place(relx=0.3, rely=0.95, anchor=CENTER)

        # initialize button run

        self.btn_run = Button(self.root, text="Run", command=self.button_run)
        self.btn_run.place(relx=0.5, rely=0.95, anchor=CENTER)

        # initialize button stop
        self.btn_stop = None

        # initialize state
        self.state = 'idle'     # value either 'idle' or 'running'

        # initialize camera
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.test()

        # initialize counter
        self.pushup_app = None
        self.pullup_app = None
        self.situp_app = None

    def button_run(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.state = 'running'
        self.btn_run.place_forget()
        self.drop_menu.place_forget()
        self.btn_stop = Button(self.root, text='Stop', command=self.button_stop)
        self.btn_stop.place(relx=0.6, rely=0.95, anchor=CENTER)
        self.live_feed()

    def button_stop(self):
        # handling cameras
        if self.selected_opt.get() == 'pushup':
            self.pushup_app.camera.release()
        elif self.selected_opt.get() == 'pullup':
            self.pullup_app.camera.release()
        elif self.selected_opt.get() == 'situp':
            self.situp_app.camera.release()
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # handling widgets and state
        self.state = 'idle'
        self.btn_run.place(relx=0.5, rely=0.95, anchor=CENTER)
        self.drop_menu.place(relx=0.3, rely=0.95, anchor=CENTER)
        self.btn_stop.place_forget()
        self.test()
        return

    def test(self):
        try:
            _, frame = self.camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.panel.configure(image=frame)
            self.panel.image = frame
            self.panel.after(10, self.test)
        except:
            return

    def live_feed(self):
        if self.selected_opt.get() == 'pushup' and self.state == 'running':
            self.pushup_app = PushupCounter(self)
            self.pushup_app()
        elif self.selected_opt.get() == 'pullup' and self.state == 'running':
            self.pullup_app = PullupCounter(self)
            self.pullup_app()
        elif self.selected_opt.get() == 'situp' and self.state == 'running':
            self.situp_app = SitupCounter(self)
            self.situp_app()  # callable class

    def main(self):
        self.root.mainloop()


class PushupCounter(MainWindow):

    def __init__(self, main_window):
        self.main_window = main_window

        # intialize variables
        self.pose_embedding = pushup.FullBodyPoseEmbedder()
        self.pose_tracker = mp_pose.Pose(min_detection_confidence=0.5)
        self.seq_list = []
        self.state = False
        self.counter = 0
        self.camera = self.main_window.camera
        self.camera.release()
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)


    def open_pushup_model(self):
        # open model pushup
        model_path = './models/pushups/model_6/pushups.csv'

        # import csv file and store in pandas
        df = pd.read_csv(model_path)
        df = df.drop(columns='Unnamed: 0')

        # Label Encoder
        le = LabelEncoder()
        y = le.fit_transform(df['push_up_motion'])

        # drop 'picture_name' and 'push_up_motion'
        df_copy = df.copy()
        df_copy = df_copy.drop(columns=['picture_name', 'push_up_motion'])
        X = df_copy.to_numpy()
        return X, y

    def seq_check(self, seq):
        ''' if the sequence is ['up', 'down', 'up'],
        it is considered as a valid sequence. Hence, the
        counter is added. Other than that the counter not
        added. the list that passed in this function
        is never empty'''

        if seq[0] == 'down' and len(seq) == 1:
            return seq.clear(), False
        elif seq[0] == 'up' and len(seq) == 1:
            return seq, False

        if len(seq) == 2:
            if seq[0] == seq[1]:
                seq.pop(0)
            return seq, False

        if len(seq) == 3:
            if seq[1] == seq[2]:
                seq.pop(1)
                return seq, False
            else:
                return seq.clear(), True

    def __call__(self):

        # video feed
        success, input_frame = self.camera.read()
        if not success:
            return

        # run pose tracker
        results = self.pose_tracker.process(image=input_frame)
        pose_landmarks = results.pose_landmarks

        # draw pose prediction
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)
            pose_landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark],
                                      dtype=np.float32)

            # process embedding and make classifications
            X, y = self.open_pushup_model()
            embedding = self.pose_embedding(pose_landmarks)
            my_knn = pushup.KNNClassifier(X, y, embedding, K=5)
            dict_result, distances_result = my_knn()

            if dict_result["up"] > dict_result["down"] and dict_result['conf_level'] > 50:
                cv2.putText(output_frame, 'up ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                self.seq_list.append('up')
                _, self.state = self.seq_check(self.seq_list)
                if self.state:
                    self.counter += 1
            elif dict_result["down"] > dict_result["up"] and dict_result['conf_level'] > 50:
                cv2.putText(output_frame, 'down ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                self.seq_list.append('down')
                _, self.state = self.seq_check(self.seq_list)
            else:
                cv2.putText(output_frame, 'not detected ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output_frame, 'Count: ' + str(self.counter), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output_frame, 'Push-up ', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # show pose detection and counter
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        output_frame = Image.fromarray(output_frame)
        output_frame = ImageTk.PhotoImage(output_frame)
        self.main_window.panel.configure(image=output_frame)
        self.main_window.panel.image = output_frame
        self.main_window.panel.after(1, self.__call__)

class SitupCounter(MainWindow):
    def __init__(self, main_window):
        self.main_window = main_window
        # intialize variables
        self.pose_embedding = situp.FullBodyPoseEmbedder()
        self.pose_tracker = mp_pose.Pose(min_detection_confidence=0.5)
        self.seq_list = []
        self.state = False
        self.counter = 0
        self.camera = self.main_window.camera
        self.camera.release()
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.situp_model = self.open_situp_model()

    def open_situp_model(self):
        # situp model path
        model_path = './models/situps/situp_model_v1.h5'

        loaded_model = load_model(model_path)
        return loaded_model

    def seq_check(self, seq):
        ''' if the sequence is ['up', 'down', 'up'],
        it is considered as a valid sequence. Hence, the
        counter is added. Other than that the counter not
        added. the list that passed in this function
        is never empty'''

        if seq[0] == 'up' and len(seq) == 1:
            return seq.clear(), False
        elif seq[0] == 'down' and len(seq) == 1:
            return self.seq_list, False

        if len(seq) == 2:
            if seq[0] == seq[1]:
                seq.pop(0)
            return seq, False

        if len(seq) == 3:
            if seq[1] == seq[2]:
                seq.pop(1)
                return seq, False
            else:
                return seq.clear(), True

    def __call__(self):

        # video feed
        success, input_frame = self.camera.read()
        if not success:
            return

        # run pose tracker
        results = self.pose_tracker.process(image=input_frame)
        pose_landmarks = results.pose_landmarks

        # draw pose prediction
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)
            pose_landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark],
                                      dtype=np.float32)

            # process embedding and make classifications
            embedding = self.pose_embedding(pose_landmarks)
            embedding = embedding.reshape(1,-1)
            embedding = embedding/180.0
            prediction = self.situp_model.predict(embedding)

            if prediction[0][0] > 0.8:
                cv2.putText(output_frame, 'up ' + str(100*prediction[0][0]) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                self.seq_list.append('up')
                _, self.state = self.seq_check(self.seq_list)

            elif prediction[0][0] < 0.2 :
                cv2.putText(output_frame, 'down ' + str((1 - prediction[0][0])*100) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                self.seq_list.append('down')
                _, self.state = self.seq_check(self.seq_list)
                if self.state:
                    self.counter += 1
                    self.seq_list = []
            else:
                cv2.putText(output_frame, 'not detected ' + str(100*prediction[0][0]) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output_frame, 'Count: ' + str(self.counter), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output_frame, 'Sit-up ', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # show pose detection and counter
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        output_frame = Image.fromarray(output_frame)
        output_frame = ImageTk.PhotoImage(output_frame)
        self.main_window.panel.configure(image=output_frame)
        self.main_window.panel.image = output_frame
        self.main_window.panel.after(1, self.__call__)

class PullupCounter(MainWindow):
    def __init__(self, main_window):
        self.main_window = main_window
        # intialize variables
        self.pose_embedding = pullup.FullBodyPoseEmbedder()
        self.pose_tracker = mp_pose.Pose(min_detection_confidence=0.5)
        self.seq_list = []
        self.state = False
        self.counter = 0
        self.camera = self.main_window.camera
        self.camera.release()
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def open_pullup_model(self):
        # open model pushup
        model_path = './models/pullups/model2_pullup.csv'

        # import csv file and store in pandas
        df = pd.read_csv(model_path)
        df = df.drop(columns='Unnamed: 0')

        # Label Encoder
        le = LabelEncoder()
        y = le.fit_transform(df['pull_up_motion'])

        # drop 'picture_name' and 'push_up_motion'
        df_copy = df.copy()
        df_copy = df_copy.drop(columns=['picture_name', 'pull_up_motion'])
        X = df_copy.to_numpy()
        return X, y

    def seq_check(self, seq):
        ''' if the sequence is ['up', 'down', 'up'],
        it is considered as a valid sequence. Hence, the
        counter is added. Other than that the counter not
        added. the list that passed in this function
        is never empty'''

        if seq[0] == 'up' and len(seq) == 1:
            return seq.clear(), False
        elif seq[0] == 'down' and len(seq) == 1:
            return self.seq_list, False

        if len(seq) == 2:
            if seq[0] == seq[1]:
                seq.pop(0)
            return seq, False

        if len(seq) == 3:
            if seq[1] == seq[2]:
                seq.pop(1)
                return seq, False
            else:
                return seq.clear(), True

    def __call__(self):

        # video feed
        success, input_frame = self.camera.read()
        if not success:
            return

        # run pose tracker
        results = self.pose_tracker.process(image=input_frame)
        pose_landmarks = results.pose_landmarks

        # draw pose prediction
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)
            pose_landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark],
                                      dtype=np.float32)

            # process embedding and make classifications
            X, y = self.open_pullup_model()
            embedding = self.pose_embedding(pose_landmarks)
            my_knn = pullup.KNNClassifier(X, y, embedding, K=5)
            dict_result, distances_result = my_knn()

            if dict_result["up"] > dict_result["down"] and dict_result['conf_level'] > 50:
                cv2.putText(output_frame, 'up ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                self.seq_list.append('up')
                _, self.state = self.seq_check(self.seq_list)

            elif dict_result["down"] > dict_result["up"] and dict_result['conf_level'] > 50:
                cv2.putText(output_frame, 'down ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                self.seq_list.append('down')
                _, self.state = self.seq_check(self.seq_list)
                if self.state:
                    self.counter += 1
                    self.seq_list = []
            else:
                cv2.putText(output_frame, 'not detected ' + str(dict_result['conf_level']) + '%', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output_frame, 'Count: ' + str(self.counter), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(output_frame, 'Pull-up ', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # show pose detection and counter
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        output_frame = Image.fromarray(output_frame)
        output_frame = ImageTk.PhotoImage(output_frame)
        self.main_window.panel.configure(image=output_frame)
        self.main_window.panel.image = output_frame
        self.main_window.panel.after(1, self.__call__)


