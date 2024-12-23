import smtplib
import string
import tkinter as tk
from doctest import master
import random
from email.message import EmailMessage
import pygame
from PIL import Image, ImageTk
import mysql.connector
from tkinter import Label, PhotoImage, messagebox, scrolledtext
from tkinter import ttk
from pydantic import ValidationError
from pygame import mixer, event
from tkinter import filedialog
from tkcalendar import Calendar
import os
import time
import cv2
import calendar
import datetime
import pymysql
import re
import email
import validate_email
import numpy as np
import tensorflow as tf
from collections import Counter
import pandas as pd

global suggested_path
title_styles = {"font": ("Arial", 30, "bold"),
                "background": "#116366",
                "foreground": "#116366"}

frame_styles = {"relief": "groove",
                "bd": 3, "bg": "#987D9A",
                "fg": "#073bb3", "font": ("Arial", 30, "bold")}


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="smart_mirror"
)
mycursor = mydb.cursor()


class CameraFeed:
    def __init__(self, frame, width=575, height=780):
        self.frame = frame
        self.width = width
        self.height = height
        self.video_label = tk.Label(self.frame)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.update_camera_feed()

    def update_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (self.width, self.height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)

            self.video_label.config(image=photo)
            self.video_label.image = photo

        self.video_label.after(10, self.update_camera_feed)

    def stop(self):
        self.cap.release()
        self.frame.destroy()
        

# Load the song data from CSV
csv_file_path = r'D:\University\1. Final Year Group Peoject\02. Implementation 02\CMS\CoderComrades_song_dataset_V1.2.csv'
song_data = pd.read_csv(csv_file_path)
pygame.mixer.init()

emotion_model = tf.keras.models.load_model(
    r'D:\University\1. Final Year Group Peoject\02. Implementation 02\CMS\Emotion_recognition_model.h5')

emotions = ['angry', 'fear', 'happy', 'neutral', 'sad']

face_cascade = cv2.CascadeClassifier('D:/University/1. Final Year Group Peoject/02. Implementation 02/CMS/haarcascade_frontalface_default.xml')

suggested_path = None

class EmotionMusicApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Emotion-Based Music Recommendation System")
        self.geometry("1200x800")
        self.configure(bg="#D5BCCF")

        # Left Frame for Camera Feed
        self.left_frame = tk.Frame(self, bg="#D5BCCF")
        self.left_frame.place(relx=0, rely=0, width=800, height=800)
        self.camera_label = tk.Label(self.left_frame)
        self.camera_label.pack()

        # Right Frame for Emotion and Music
        self.right_frame = tk.Frame(self, bg="#D5BCCF")
        self.right_frame.place(relx=0.67, rely=0.02, width=350, height=750)

        # Back Button
        self.back_button = tk.Button(self.right_frame, text="Back to Dashboard", command=self.go_to_dashboard)
        self.back_button.pack(pady=10)

        # Sub-Frame for Emotion Display
        self.emotion_label = tk.Label(self.right_frame, text="Detecting Emotion...", font=("Verdana", 15),
                                      bg="#E9DCE4", fg="#7A627C")
        self.emotion_label.pack(pady=20)

        # Sub-Frame for Music Player
        self.music_label = tk.Label(self.right_frame, text="Music Player", font=("Verdana", 12), bg="#E9DCE4", fg="#7A627C")
        self.music_label.pack(pady=10)
        self.music_info = tk.Label(self.right_frame, text="", font=("Verdana", 10), bg="#E9DCE4", fg="#7A627C")
        self.music_info.pack()

        # Initialize Video Capture
        self.cap = cv2.VideoCapture(0)
        self.detecting_emotion = True
        self.detected_emotion = None
        self.detected_emotions = []
        self.start_time = time.time()
        self.analysis_duration = 15
        self.show_camera_feed()

    def go_to_dashboard(self):
        """ Function to go back to the dashboard """
        self.destroy()
        dashboard()

    def show_camera_feed(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                desired_width = 575
                desired_height = 780
                frame = cv2.resize(frame, (desired_width, desired_height))

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

                if self.detecting_emotion:
                    self.detect_emotion(frame)

            self.after(10, self.show_camera_feed)

    def detect_emotion(self, frame):
        global suggested_path  # Declare global variable

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If faces are detected, process the first one
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_image = frame[y:y + h, x:x + w]
            preprocessed_face = self.preprocess_face(face_image)
            emotion_prediction = emotion_model.predict(preprocessed_face)
            emotion_index = np.argmax(emotion_prediction[0])
            self.detected_emotions.append(emotion_index)
            detected_emotion = emotions[emotion_index]

            if time.time() - self.start_time >= self.analysis_duration:
                self.detecting_emotion = False
                self.finalize_emotion_detection()

    def preprocess_face(self, face_image):
        face_image = cv2.resize(face_image, (48, 48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = face_image.astype("float32") / 255.0
        face_image = np.expand_dims(face_image, axis=-1)
        face_image = np.expand_dims(face_image, axis=0)
        return face_image

    def finalize_emotion_detection(self):
        global suggested_path  # Declare global variable

        if self.detected_emotions:
            most_common_emotion_index = Counter(self.detected_emotions).most_common(1)[0][0]
            detected_emotion = emotions[most_common_emotion_index]
            self.emotion_label.config(text=f"Detected Emotion: {detected_emotion}")
            global GEmotion
            GEmotion = detected_emotion
            suggested_path = self.select_song_based_on_emotion(detected_emotion)
            if suggested_path:
                self.music_info.config(text=f"Playing song: {suggested_path}")
            else:
                self.music_info.config(text="No suitable song found.")

    def select_song_based_on_emotion(self, emotion):
        if emotion == 'happy':
            emotion_filter = (song_data['valence'] > 3000) & \
                            (song_data['energy'] > 0.2) & \
                            (song_data['danceability'] > 0.15) & \
                            (song_data['tempo'] > 120)
        elif emotion == 'sad':
            emotion_filter = (song_data['valence'] < 2000) & \
                          (song_data['energy'] < 0.1)
        elif emotion == 'angry':
            emotion_filter = (song_data['valence'] > 2000) & (song_data['valence'] < 2500) & \
                            (song_data['tempo'] > 130)
        elif emotion == 'fear':
            emotion_filter = (song_data['valence'] < 2000) & \
                            (song_data['energy'] < 0.2) & \
                            (song_data['instrumentalness'] > 0.1)
        elif emotion == 'neutral':
            emotion_filter = (song_data['valence'] > 2000) & (song_data['valence'] < 3000) & \
                            (song_data['energy'] > 0.1) & (song_data['energy'] < 0.2) & \
                            (song_data['tempo'] > 100) & (song_data['tempo'] < 120)
        else:
            return None

        filtered_songs = song_data[emotion_filter]
        if not filtered_songs.empty:
            selected_song = filtered_songs.sample(1)
            return selected_song['path'].values[0]
        return None

    def play_song(self, song_path):
        if os.path.exists(song_path):
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
        else:
            self.music_info.config(text="Song path does not exist.")

    def on_close(self):
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()

        
class Main(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#987D9A", height=800, width=600, bd=1)
        main_frame.pack(fill="both", expand="true")
        self.title("Smart Emotion Detection System")
        self.geometry("1200x800")
        self.resizable(True, True)

        frame_001 = tk.Frame(main_frame, relief="groove", bd=0, bg="#111111")
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)

        self.camera_feed = CameraFeed(frame_001, width=575, height=780)

        text_styles = {"font": ("Verdana", 6),
                       "background": "#987D9A",
                       "foreground": "#987D9A"}

        title_stylesm = {"font": ("Verdana", 30, "bold"),
                         "background": "#987D9A",
                         "foreground": "#720455"
                         }
        label_styles1 = {"background": "#987D9A", }
        frame_11 = tk.Frame(main_frame, relief="groove",
                            bd=0)
        frame_11.place(rely=0.01, relx=0.509, height=780, width=575)
        frame_11.configure(bg="#987D9A")

        img_path = "img/img.gif"
        try:
            info = Image.open(img_path)
            frames = info.n_frames

            photoimage_objects = []
            for i in range(frames):
                info.seek(i)
                frame = info.copy().resize((250, 250))
                photoimage = ImageTk.PhotoImage(frame.convert("RGBA"))
                photoimage_objects.append(photoimage)

            current_frame = 0
            gif_label = Label(frame_11)
            gif_label.pack()
            gif_label.place(relx=0.5, rely=0.1, anchor="center")
            gif_label.configure(background="#987D9A")

            def update_frame():
                nonlocal current_frame
                gif_label.config(image=photoimage_objects[current_frame])
                current_frame = (current_frame + 1) % frames
                self.after(50, update_frame)

            update_frame()

        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")

        img_path = "img/start3.png"

        try:
            image = Image.open(img_path)
            image = image.resize((200, 200))
            image1 = image.convert("RGBA")
            photo = ImageTk.PhotoImage(image1)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return
            
        buttton = tk.Button(frame_11, image=photo, bd=0, background="#047e82", command=lambda: start())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.6, relx=0.32)
        buttton.configure(activebackground="#987D9A", background="#987D9A")

        #button = ttk.Button(frame_11, text="start", command=lambda: start())
        #button.place(rely=0.7, relx=0.4, height=100, width=150)

        label_title = tk.Label(frame_11, title_stylesm, text="Emotion recognition")
        label_title.place(rely=0.43, relx=0.13)

        label_title = tk.Label(frame_11, title_stylesm, text="smart Mirror ")
        label_title.place(rely=0.5, relx=0.28)

        img_path = "img/logo_cc.png"

        try:
            image = Image.open(img_path)
            image1 = image.convert("RGBA")
            photo = ImageTk.PhotoImage(image1)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return

        label = tk.Label(frame_11, image=photo, bd=0)
        label.image = photo
        label.pack()
        label.place(rely=0.85, relx=0.42)
        label.configure(background="#987D9A", anchor="center")

        label_details = tk.Label(main_frame, text_styles, text="Emotion recognition smart Mirror ")
        label_details.place(rely=0.98, relx=0.40)

        label_details = tk.Label(main_frame, text_styles, text="By codercomrades ")
        label_details.place(rely=1, relx=0.45)

        
        def start():
            self.camera_feed.stop()
            Main.destroy(self)
            #dashboard()
            LoginPage()
            
        def update_frame():
            nonlocal current_frame
            gif_label.config(image=photoimage_objects[current_frame])
            current_frame = (current_frame + 1) % frames
            self.after(50, update_frame)
    
    def on_close(self):
        self.camera_feed.stop()
        self.destroy()

        

class LoginPage(tk.Tk):
   def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#BB9AB1", height=800, width=600, bd=1)
        main_frame.pack(fill="both", expand="true")
        self.title("Smart Emotion Detection System")
        self.geometry("1200x800")
        self.resizable(True, True)

        frame_001 = tk.Frame(main_frame, relief="groove", bd=0, bg="#BB9AB1")
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)

        self.camera_feed = CameraFeed(frame_001) 

        text_styles = {"font": ("Verdana", 15),
                       "background": "#BB9AB1",
                       "foreground": "#f5f5f5"}

        error = {"font": ("Verdana", 10),
                 "background": "#BB9AB1",
                 "foreground": "#f5f5f5"}

        frame_login = tk.Frame(main_frame, bg="#BB9AB1", relief="groove",
                               bd=0)
        frame_login.place(rely=0.01, relx=0.509, height=780, width=600)

        img_path = "img/back1.png"

        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return 

        ba_btn = tk.Button(frame_login, image=photo, bd=0, background="#BB9AB1", command=lambda: back())
        ba_btn.image = photo
        ba_btn.pack()
        ba_btn.place(rely=0.011, relx=0.01)
        ba_btn.configure(activebackground="#BB9AB1", background="#BB9AB1")

        img_path = "img/bl.png" 

        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return


        label = tk.Label(frame_login, image=photo, bd=0, background="#BB9AB1")
        label.image = photo
        label.pack()
        label.place(rely=0.008, relx=0.09)

        label_user = tk.Label(frame_login, text_styles, text="Username:")
        label_user.place(rely=0.38, relx=0.05)

        label_pw = tk.Label(frame_login, text_styles, text="Password:")
        label_pw.place(rely=0.54, relx=0.05)

        label_show = tk.Label(main_frame, error, text="")
        label_show.place(rely=0.01, relx=0.6)

        # Username Entry with Hint
        username_var = tk.StringVar(value="Username")  # Variable to store username and initial hint text

        username_entry = tk.Entry(frame_login, width=4, font=("Arial", 12))
        username_entry.place(rely=0.4, relx=0.4)
        username_entry.config(foreground="#BB9AB1", background="#BB9AB1")

        def on_enter_username(event=None):
            if username_var.get() == "Username":
                username_entry.delete(0, tk.END)  # Clear hint text on focus
                username_var.set("")  # Set variable to empty string

        def on_leave_username(event=None):
            if username_var.get() == "":
                username_entry.delete(0, tk.END)  
                username_var.set("Username") 

        username_entry.bind("<FocusIn>", on_enter_username)
        username_entry.bind("<FocusOut>", on_leave_username)

        
        password_var = tk.StringVar(value="Password")  

        password_entry = ttk.Entry(frame_login, width=4, font=("Arial", 12), show="")
        password_entry.place(rely=0.55, relx=0.39)
        password_entry.config(foreground="#BB9AB1")  

        def on_enter_password(event=None):
            if password_var.get() == "Password":
                password_entry.delete(0, tk.END)  
                password_var.set("") 
                password_entry.config(show="*")  
        def on_leave_password(event=None):
            if password_var.get() == "":
                password_entry.delete(0, tk.END)  
                password_var.set("Password")  
                password_entry.config(show="") 

        password_entry.bind("<FocusIn>", on_enter_password)
        password_entry.bind("<FocusOut>", on_leave_password)

        entry_user = ttk.Entry(frame_login, width=40, cursor="xterm")
        entry_user.place(rely=0.4, relx=0.4)
        entry_user.config(foreground="#BB9AB1", background="#BB9AB1")

        entry_pw = ttk.Entry(frame_login, width=40, cursor="xterm", show="*")
        entry_pw.place(rely=0.55, relx=0.4)
        entry_user.config(foreground="#BB9AB1", background="#BB9AB1")

        # button = ttk.Button(frame_login, text="Login", command=lambda: getlogin())
        # button.place(rely=0.55, relx=0.350, height=70, width=150)

        img_path = r"img/login11.png"  

       
        try:
            image = Image.open(img_path)
            image = image.resize((250, 80))
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_login, image=photo, bd=0, background="#BB9AB1", command=lambda: getlogin())
        button.image = photo
        button.pack()
        button.place(rely=0.72, relx=0.350)
        button.configure(activebackground="#BB9AB1", background="#BB9AB1")

        # signup_btn = ttk.Button(frame_login, text="Reset ", command=lambda: reset_())
        # signup_btn.place(rely=0.80, relx=0.420)

        img_path = "img/forget.png"  

      
        try:
            image = Image.open(img_path)
            image = image.resize((120, 80))
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return

       
        button = tk.Button(frame_login, image=photo, bd=0, background="#BB9AB1", command=lambda: reset_())
        button.image = photo
        button.pack()
        button.place(rely=0.88, relx=0.74)
        button.configure(activebackground="#BB9AB1", background="#BB9AB1")

        # signup_btn = ttk.Button(frame_login, text="Register", command=lambda: get_signup())
        # signup_btn.place(rely=0.75, relx=0.420)

        img_path = "img/Create.png" 

       
        try:
            image = Image.open(img_path)
            image = image.resize((250, 110))
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

       
        button = tk.Button(frame_login, image=photo, bd=0, background="#987D9A", command=lambda: get_signup())
        button.image = photo
        button.pack()
        button.place(rely=0.88, relx=0.01)
        button.configure(activebackground="#BB9AB1", background="#BB9AB1")

        img_path = "img/img.png" 

        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return

        label = tk.Label(frame_login, image=photo, bd=0, background="#BB9AB1")
        label.image = photo
        label.pack()
        label.place(rely=0.0001, relx=0.24)
        label.configure(anchor="center", background="#BB9AB1")


        def back():
            self.camera_feed.stop()
            LoginPage.destroy(self)
            Main()

        def reset_():
            self.camera_feed.stop()
            reset()
            LoginPage.destroy(self)


        def get_signup():
            self.camera_feed.stop()
            LoginPage.destroy(self)
            SignupPage()

        def getlogin():
            username = entry_user.get()
            password = entry_pw.get()
            # if your want to run the script as it is set validation = True
            validation = validate(username, password)
            if validation:
                get_u(username)
                get_id(username)
                self.camera_feed.stop()
                LoginPage.destroy(self)
                dashboard()
            else:
                label_show.configure(text="Invalid Username or Password", fg="red")
                global count
                count += 1
                if count >= 3:
                    use_in = messagebox.askyesno("Information", "Did you forgot your password?")
                    if use_in:
                        LoginPage.destroy(self)
                        reset()

                    else:
                        tk.messagebox.showerror("Information",
                                                "The Username or Password you have entered are incorrect ")

        def validate(username, password):
            query = "select* from user where user_name = '{}' and password = '{}'".format(username, password)
            mycursor.execute(query)
            result = mycursor.fetchall()
            if result:
                return True
            else:
                return False
            
def on_close(self):
    self.camera_feed.stop()
    self.destroy()
            

def on_change():
    pass


def check_1(val, email_v):
    if re.search(regex, val):
        email_v.configure(text="valid")
        return True
    else:
        email_v.configure(text="Invalid")
        return False


count = 0
user_n = 0
user1 = 0


def get_u(username):
    global user_n
    global user1
    user1 = username
    query = "select first_name from user where user_name = '{}'".format(username)
    mycursor.execute(query)
    result = mycursor.fetchall()
    if result:
        user_n = result[0][0]
    else:
        user_n = username


user_id = 0


def get_id(username):
    global user_id
    query = "select User_id from user where user_name = '{}'".format(username)
    mycursor.execute(query)
    result = mycursor.fetchall()
    if result:
        user_id = result[0][0]
        
"""
import hashlib
import os

def encrypt_password(password: str) -> str:
    salt = os.urandom(16) 
    hashed_password = hashlib.sha256(salt + password.encode('utf-8')).hexdigest()
    return salt.hex() + ':' + hashed_password

password = 'my_secure_password'
encrypted_password = encrypt_password(password)
print(f"Encrypted password: {encrypted_password}")



def verify_password(stored_password: str, input_password: str) -> bool:
    salt_hex, stored_hash = stored_password.split(':')
    salt = bytes.fromhex(salt_hex)
    
    input_hash = hashlib.sha256(salt + input_password.encode('utf-8')).hexdigest()
    
    return input_hash == stored_hash

input_password = 'my_secure_password'
is_valid = verify_password(encrypted_password, input_password)
print(f"Password valid: {is_valid}"""

class SignupPage(tk.Tk):

    def __init__(self, check_email=None, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        self.root = root
        self.frame_sign = None
        main_frame = tk.Frame(self, bg="#BB9AB1", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="true")

        self.geometry("1200x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen
        self.title("Smart Emotion Detection System- Sign up")

        text_styles = {"font": ("Verdana", 11, "bold"),
                       "background": "#BB9AB1",
                       "foreground": "#000000"}
        label_fn = tk.Label(main_frame, text_styles, text="Welcome to the SMART MIRROR:")
        label_fn.place(rely=0.017, relx=0.509)

        img_path = "img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        but = tk.Button(main_frame, image=photo, bd=0, background="#BB9AB1", command=lambda: des())
        but.image = photo
        but.pack()
        but.place(rely=0.011, relx=0.01)

        img_path = "img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(main_frame, image=photo, bd=0, background="#BB9AB1")
        label.image = photo
        label.pack()
        label.place(rely=0.008, relx=0.09)

        frame_sign = tk.Frame(main_frame, bg="#BB9AB1", relief="groove",
                              bd=0)  # this is the frame that holds all the login details and buttons
        frame_sign.place(rely=0.057, relx=0.300, height=700, width=450)

        def check(val, email_v, lable_v):
            regex1 = '^[a-z0-9]+[._]?[a-z0-9]+[@]w+[.]w{2,3}$'
            check_1(regex1, val)
            if check_1:
                self.lable_v.configure(text="valid")
            else:
                self.lable_v.configure(text="Invalid")

        label_fn = tk.Label(frame_sign, text_styles, text="First Name:")
        label_fn.place(rely=0.06, relx=0.2)

        label_ln = tk.Label(frame_sign, text_styles, text="Last Name:")
        label_ln.place(rely=0.06, relx=0.5)

        entry_fn = ttk.Entry(frame_sign, width=20, cursor="xterm")
        entry_fn.place(rely=0.12, relx=0.2)

        entry_ln = ttk.Entry(frame_sign, width=20, cursor="xterm")
        entry_ln.place(rely=0.12, relx=0.5)

        label_email = tk.Label(frame_sign, text_styles, text="E-mail:")
        label_email.place(rely=0.2, relx=0.2)

        entry_email = ttk.Entry(frame_sign, width=20, cursor="xterm", validate="focusout",
                                validatecommand=(check_email, '%s'))
        entry_email.place(rely=0.2, relx=0.5)

        self.validation_label = tk.Label(self, text="", fg="#111111", bg="#BB9AB1")
        self.validation_label.pack()
        self.validation_label.place(rely=0.2, relx=0.7)

        label_sq = tk.Label(frame_sign, text_styles, text="enter date of birth:")
        label_sq.place(rely=0.45, relx=0.2)

        cal = Calendar(frame_sign, selectmode="day", year=2000, month=1, day=1)  # Set initial date
        cal.pack(pady=10)
        cal.place(rely=0.5, relx=0.2)

        label_user = tk.Label(frame_sign, text_styles, text="Username:")
        label_user.place(rely=0.3, relx=0.2)

        label_pw = tk.Label(frame_sign, text_styles, text="Password:")
        label_pw.place(rely=0.4, relx=0.2)

        self.validation_label1 = tk.Label(self, text="", fg="#111111", bg="#BB9AB1")
        self.validation_label1.pack()
        self.validation_label1.place(rely=0.38, relx=0.6)

        entry_user = ttk.Entry(frame_sign, width=20, cursor="xterm")
        entry_user.place(rely=0.3, relx=0.5)

        entry_pw = ttk.Entry(frame_sign, width=20, cursor="xterm", show="*")
        entry_pw.place(rely=0.4, relx=0.5)

        label_user = tk.Label(frame_sign, text_styles, text="verification code:")
        label_user.place(rely=0.9, relx=0.1)

        entry_vary = ttk.Entry(frame_sign, width=20, cursor="xterm", show="*")
        entry_vary.place(rely=0.9, relx=0.4)

        button_login = ttk.Button(frame_sign, text="Varify email", width=15, command=lambda: varify())
        button_login.place(rely=0.9, relx=0.70)
        button_login.configure()

        label_user = tk.Label(frame_sign, text_styles, text="Already have a account")
        label_user.place(rely=0.95, relx=0.1)

        button_login = ttk.Button(frame_sign, text="Login", width=15, command=lambda: get_login())
        button_login.place(rely=0.95, relx=0.6)

        img_path = "img/Create1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            image = image.resize((190, 70))
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_sign, image=photo, bd=0, background="#BB9AB1", command=lambda: signup())
        button.image = photo
        button.pack()
        button.place(rely=0.8, relx=0.4)

        def signup():
            # Creates a text file with the Username and password
            user = entry_user.get()
            email1 = entry_email.get()
            fn = entry_fn.get()
            ln = entry_ln.get()
            dob = cal.selection_get()
            pw = entry_pw.get()
            em = validate_email(email1)
            pass1 = is_secure_password(pw, min_length=8, min_uppercase=1, min_lowercase=1, min_digit=1, min_special=1)
            validate_email_handler(self)
            check_password_handler(self)
            validation = validate_u(user)
            if em:
                if not validation:
                    tk.messagebox.showerror("Information", "That Username already exists")
                else:
                    if not pw or not ln or not fn or not email or not user:
                        tk.messagebox.showerror("Error", "Please enter your details first", )
                    else:
                        if pass1:
                            send_code()

                        else:
                            tk.messagebox.showerror("Information", "Your password needs to select strong password")

                if not validation:
                    tk.messagebox.showerror("Information", "That Username already exists")
            else:
                tk.messagebox.showerror("Information", "Enter correct email")

        def validate_u(username):
            query = "select* from user where user_name = '{}'".format(username)
            mycursor.execute(query)
            result = mycursor.fetchall()
            if result:
                return False
            else:
                return True

        def get_login():
            SignupPage.destroy(self)
            LoginPage()

        def des():
            SignupPage.destroy(self)
            LoginPage()

        def validate_email_handler(self):
            emailw = entry_email.get()

            if validate_email(emailw):
                if check_email():
                    self.validation_label.config(text="Valid email address", fg="green")
                    return True
                else:
                    self.validation_label1.config(text="Invalid email: Missing '@' symbol")
            else:
                self.validation_label.config(text="Invalid email address", fg="red")
                return False

        def check_password_handler(self):
            password = entry_pw.get()

            if is_secure_password(password):
                self.validation_label1.config(text="Password is Secure!", fg="green")
            else:
                self.validation_label1.config(text="Password Does Not Meet Requirements", fg="red")

        def send_code():
            email2 = entry_email.get()
            code = generate_verification_code()
            get1(code)
            send_verification_email1(email2, code)
            # Display a success message or prompt for further verification steps
            tk.messagebox.showinfo("Information", "Your varification code was sent")  # Clear username entry

        def generate_verification_code(length=6):
            """
            Generates a random alphanumeric verification code of specified length.

            Args:
                length (int, optional): The length of the desired verification code. Defaults to 6.

            Returns:
                str: The generated verification code.
            """
            digits = string.ascii_letters + string.digits
            code = ''.join(random.choice(digits) for _ in range(length))
            return code

        def varify():
            x = entry_vary.get()
            if x == v_code12:
                finish()
            else:
                tk.messagebox.showerror("Error", "Please enter correct code")

        def finish():
            user = entry_user.get()
            email1 = entry_email.get()
            fn = entry_fn.get()
            ln = entry_ln.get()
            dob = cal.selection_get()
            pw = entry_pw.get()
            mycursor.execute(
                "INSERT INTO user (First_Name, Last_Name, User_Name, Password, Date_of_Birth, Email) VALUES(%s, %s, %s, %s, %s, %s)",
                (fn, ln, user, pw, dob, email1))
            mydb.commit()
            tk.messagebox.showinfo("Information", "Your account has been created!\n now you can log in")
            SignupPage.destroy(self)
            LoginPage()

        def check_email():
            email12 = entry_email.get()
            if '@' not in email12:
                return False
            else:
                return True


def send_verification_email1(email1, code):
    smtp_server = 'smtp.gmail.com'
    port = 587

    email2 = 'codercomradesverify@gmail.com'
    password = 'dtom iken jfmy fzsy'

    server = smtplib.SMTP(smtp_server, port)

    server.starttls()

    server.login(email2, password)

    # Create email message
    msg = EmailMessage()
    msg['From'] = email2
    msg['To'] = email1
    msg['Subject'] = 'Verification Code'
    msg.set_content(f'dont reply this message.this is auto genarated message\n Your verification code is: {code}')

    sender = 'codercomradesverify@gmail.com'
    receiver = email1

    # server.sendmail(sender, receiver, msg)
    server.sendmail(msg['From'], [msg['To']], msg.as_string())

    server.quit()


v_code12 = 0


def get1(v_code1):
    global v_code12
    v_code12 = v_code1


def validate_email(email):
    regex1 = r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$"
    if re.fullmatch(regex1, email):
        return True
    else:
        return False


def is_secure_password(password, min_length=8, min_uppercase=1, min_lowercase=1, min_digit=1, min_special=1):
    special_chars_regex = r"[!@#$%^&*()-_=+[{\]};:'\",<.>/?|`~]"

    criteria = [
        ("Length (min 8 chars)", len(password) >= min_length),
        ("Uppercase letter", any(char.isupper() for char in password)),
        ("Lowercase letter", any(char.islower() for char in password)),
        ("Digit", any(char.isdigit() for char in password)),
        ("Special character", bool(re.search(special_chars_regex, password))),
    ]

    return all(met for _, met in criteria)


height = 50
corner_radius = 10


class reset(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#BB9AB1", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="true")

        self.geometry("1200x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen
        self.title("Smart Emotion Detection System- Reset password")

        text_styles = {"font": ("Verdana", 15),
                       "background": "#BB9AB1",
                       "foreground": "#7A627C"}


        # Text field and scrollbar for displaying instructions
        text_frame = tk.Frame(main_frame, bg="#BB9AB1", relief="groove", bd=0)
        text_frame.place(rely=0.23, relx=0.310, height=200, width=400)

        text_instructions = tk.Text(text_frame, wrap=tk.WORD, width=45, height=40, font=("Verdana", 10), bd=0)
        text_instructions.insert(tk.INSERT, "Enter your registered Username Below.")
        text_instructions.config(state=tk.DISABLED)  # Disable editing of instructions
        text_instructions.config(bg="#E9DCE4", font=("Verdana", 10), fg="black")

        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_instructions.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_instructions.config(yscrollcommand=scrollbar.set)
        text_instructions.pack(fill=tk.BOTH, expand=True)  # Fill available space

        frame_reset = tk.Frame(main_frame, bg="#BB9AB1", relief="groove",
                               bd=0)  # this is the frame that holds all the login details and buttons
        frame_reset.place(rely=0.055, relx=0.300, height=700, width=450)

        label_user = tk.Label(frame_reset, text_styles, text="Reset Password ")
        label_user.place(rely=0.06, relx=0.38)

        """label_user = tk.Label(frame_reset, title_styles, text="Reset Password ")
        label_user.place(rely=0.09, relx=0.12)"""

        label_user = tk.Label(frame_reset, text_styles, text="Username:")
        label_user.place(rely=0.642, relx=0.16)

        entry_user = ttk.Entry(frame_reset, width=20, cursor="xterm", font=("verdana", 10))
        entry_user.place(rely=0.65, relx=0.5)
        entry_user.config(foreground="black")

        self.label_e = tk.Label(frame_reset, text_styles, text="")
        self.label_e.place(rely=0.75, relx=0.2)

        button = ttk.Button(frame_reset, text="Search", width=15, command=lambda: search())
        button.place(rely=0.9, relx=0.4)

        button = ttk.Button(frame_reset, text="back", width=15, command=lambda: back())
        button.place(rely=0.94, relx=0.4)


        def back():
            LoginPage()
            reset.destroy(self)

        def search():
            user1 = entry_user.get()
            validation = validate_us(user1)
            if not validation:
                return False
            else:
                send_code()
                reset.destroy(self)
                reset_pass()

        def reset_pass():
            ch_pass()

        def validate_us(username):
            query = "select* from user where user_name = '{}'".format(username)
            mycursor.execute(query)
            result = mycursor.fetchall()
            if result:
                return True
            else:
                self.label_e.config(text=f"Your account not found", fg="red")
                return False

        def get_email_from_username(username):
            sql1 = "SELECT e_mail FROM user WHERE User_name = %s"
            val = (username,)
            mycursor.execute(sql1, val)
            result = mycursor.fetchone()
            if result:
                get_user(username)
                self.label_e.config(text=f"Your email is:  {result}", fg="green")  # Return the email address
                return result[0]
            else:
                self.label_e.config(text=f"Your email address not found", fg="red")
                return None  # Username not found

        def send_code():
            username = entry_user.get()
            email2 = get_email_from_username(username)

            if email:
                code = generate_verification_code()
                get(code)
                send_verification_email(email2, code)
                # Display a success message or prompt for further verification steps
                entry_user.delete(0, tk.END)
                tk.messagebox.showinfo("Information", "Your varification code was sent")  # Clear username entry
            else:
                # Display an error message indicating username not found
                pass  # Implement appropriate error handling

        def generate_verification_code(length=6):
            """
            Generates a random alphanumeric verification code of specified length.

            Args:
                length (int, optional): The length of the desired verification code. Defaults to 6.

            Returns:
                str: The generated verification code.
            """
            digits = string.ascii_letters + string.digits
            code = ''.join(random.choice(digits) for _ in range(length))
            return code


v_code = 0
user_ = 0


def get(v_code1):
    global v_code
    v_code = v_code1


def get_user(user_name1):
    global user_
    user_ = user_name1


def send_verification_email(email1, code):
    smtp_server = 'smtp.gmail.com'
    port = 587

    email2 = 'codercomradesverify@gmail.com'
    password = 'dtom iken jfmy fzsy'

    server = smtplib.SMTP(smtp_server, port)

    server.starttls()

    server.login(email2, password)

    # Create email message
    msg = EmailMessage()
    msg['From'] = email2
    msg['To'] = email1
    msg['Subject'] = 'Verification Code'
    msg.set_content(f'dont reply this message.this is auto genarated message\n Your verification code is: {code}')

    sender = 'codercomradesverify@gmail.com'
    receiver = email1

    # server.sendmail(sender, receiver, msg)
    server.sendmail(msg['From'], [msg['To']], msg.as_string())

    server.quit()


class ch_pass(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        main_frame = tk.Frame(self, bg="#BB9AB1", height=800, width=1200)  # this is the background
        main_frame.pack(fill="both", expand="true")

        self.geometry("1200x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen
        self.title("Password Reset")

        text_styles = {"font": ("Verdana", 15),
                       "background": "#BB9AB1",
                       "foreground": "#f5f5f5"}
        img_path = "img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        b_button = tk.Button(main_frame, image=photo, bd=0, background="#BB9AB1", command=lambda: back())
        b_button.image = photo
        b_button.pack()
        b_button.place(rely=0.011, relx=0.01)

        label_user = tk.Label(main_frame, text_styles, text="Change password ")
        label_user.place(rely=0.0197, relx=0.3)

        frame_reset1 = tk.Frame(main_frame, bg="#BB9AB1", relief="groove", bd=0)
        frame_reset1.place(rely=0.3, relx=0.55, height=380, width=405)

        label_fn = tk.Label(frame_reset1, text_styles, text="verification code")
        label_fn.place(rely=0.09, relx=0.32)

        entry_vary = ttk.Entry(frame_reset1, width=20, cursor="xterm")
        entry_vary.place(rely=0.19, relx=0.38)

        label_pw = tk.Label(frame_reset1, text_styles, text="Enter New password")
        label_pw.place(rely=0.4, relx=0.265)

        entry_pw = ttk.Entry(frame_reset1, width=20, cursor="xterm", show="*")
        entry_pw.place(rely=0.5, relx=0.38)

        label_pw1 = tk.Label(frame_reset1, text_styles, text="Enter password again")
        label_pw1.place(rely=0.6, relx=0.26)

        entry_pw1 = ttk.Entry(frame_reset1, width=20, cursor="xterm", show="*")
        entry_pw1.place(rely=0.7, relx=0.38)

        reset_btn = tk.Button(frame_reset1, text="Change Password", command=lambda: change())
        reset_btn.place(rely=0.8, relx=0.4, width=100, height=30)

        label_msg = tk.Label(frame_reset1, text_styles, text="")
        label_msg.place(rely=0.1, relx=0.1)

        frame_001 = tk.Frame(main_frame, relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)
        frame_001.configure(bg="#BB9AB1")

        def back():
            user_in = tk.messagebox.askyesno("cancel Password rest process ",
                                             "Do you want to cancel  password reset process?")
            if user_in:
                ch_pass.destroy(self)
                reset()

        def change():
            variefy = entry_vary.get()
            password = entry_pw.get()
            password1 = entry_pw1.get()
            if check(variefy):
                if password1 == password:
                    if is_secure_password(password):
                        if up(user_, password):
                            ch_pass.destroy(self)
                            LoginPage()
                        else:
                            tk.messagebox.showerror("Error", "System error try later")

                    else:
                        tk.messagebox.showerror("Error", "password doesn't meet the security requirements")

                else:
                    tk.messagebox.showerror("Error", "Password not match")

            else:
                tk.messagebox.showerror("Error", "Enter correct verification code")

        def check(variefy):
            global v_code
            if variefy == v_code:
                return True
            else:
                return False

        def up(user, password):
            if update1(user, password):
                return True


def update1(user, password):
    sql = "UPDATE user SET password = %s WHERE User_name = %s"
    mycursor.execute(sql, (password, user))
    mydb.commit()
    return True


emotion = 0


def show_pop1(event, popup_manu=None):
    popup_manu.post(event.x_root, event.y_root)


def update_frame(canvas=None):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (500, 430))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(10, 10, anchor=tk.NW, image=photo)
        canvas.image = photo
        canvas.after(10, update_frame())


cap = cv2.VideoCapture(0)


class Loading(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#474849", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("1200x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "white",
                       "foreground": "#000000"}
        text_styles1 = {"font": ("Verdana", 10),
                        "background": "#43ffff",
                        "foreground": "#111111"}

        text_styles2 = {"font": ("Verdana", 6),
                        "background": "#43ffff",
                        "foreground": "#111111"}

        text_styles3 = {"font": ("Verdana", 6),
                        "background": "yellow",
                        "foreground": "#111111"}
        title_style_read = {"font": ("Verdana", 10),
                            "background": "#ffffff",
                            "foreground": "#111111"}

        frame_scan = tk.Frame(main_frame, bg="#B9A6BA", relief="groove", bd=0)
        frame_scan.place(rely=0.090, relx=0.509, height=750, width=580)

        frame_001 = tk.Frame(main_frame, relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)
        frame_001.configure(bg="#111111")
        
        label_fn = tk.Label(frame_scan, text_styles, text="Welcome to the SMART MIRROR:")
        label_fn.place(rely=0.001, relx=0.15)

        canvas = tk.Canvas(frame_scan, bg="#474849", height=500, width=600, bd=1)
        canvas.pack(fill="both", expand="20")

        canvas.create_oval(500, 500, 220, 170, outline="#000000", width=2)

        update_frame()




def on_enter(event):
    pass


class dashboard(tk.Tk):
    def __init__(self, show_pop1=None, on_enter=None, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#BB9AB1", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("1200x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "#BB9AB1",
                       "foreground": "#ffffff"}
        text_styles1 = {"font": ("Verdana", 10),
                        "background": "#BB9AB1",
                        "foreground": "#ffffff"}

        text_styles2 = {"font": ("Verdana", 6),
                        "background": "#BB9AB1",
                        "foreground": "#ffffff"}

        text_styles3 = {"font": ("Verdana", 6),
                        "background": "#BB9AB1",
                        "foreground": "#ffffff"}
        title_style_read = {"font": ("Verdana", 10),
                            "background": "#BB9AB1",
                            "foreground": "#ffffff"}
        
        frame_dash = tk.Frame(main_frame, bg="#BB9AB1", relief="groove", bd=0)
        frame_dash.place(rely=0.0001, relx=0.500, height=790, width=580)

        label_fn = tk.Label(frame_dash, text_styles, text="Welcome to the SMART MIRROR:")
        label_fn.place(rely=0.04, relx=0.3)
        label_fn.configure(background="#BB9AB1", font=("verdana", 12))

        img_path = "img/set.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            image = image.resize((30, 20))
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_dash, image=photo, bd=0, background="#BB9AB1", command=lambda: manu1())
        button.image = photo
        button.pack()
        button.place(rely=0.000001, relx=0.9)
        button.configure(activebackground="#BB9AB1", height=89, width=87)

        img_path = "img/logout.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            image = image.resize((200, 150))
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_dash, image=photo, bd=0, background="#BB9AB1", command=lambda: log())
        button.image = photo
        button.pack()
        button.place(rely=0.01, relx=0.02)
        button.configure(activebackground="#BB9AB1", height=45, width=87)
        

        label_fn = tk.Label(frame_dash, text_styles, text=f"{user_n} Now you are currently in:{emotion} ")
        label_fn.place(rely=0.14, relx=0.05)

        if c_emo == 5:
            label_fn.configure(foreground="blue", font=("verdana", 10, "bold"), bg="#BB9AB1")
        elif c_emo == 6:
            label_fn.configure(foreground="green", font=("verdana", 10, "bold"), bg="#BB9AB1")
        elif c_emo == 7:
            label_fn.configure(foreground="yellow", font=("verdana", 10, "bold"), bg="#BB9AB1")
        else:
            label_fn.configure(foreground="red", font=("verdana", 10, "bold"), bg="#BB9AB1")

        frame_de = tk.Frame(frame_dash, bg="#BB9AB1", relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_de.place(rely=0.17, relx=0.0290, height=150, width=520)

        
        frame_details = tk.LabelFrame(frame_de, bg="#ffffff", relief="groove", bd=0,
                                      text="You can move your way to the positive.Be emotionally positive.")  # this is the frame that holds all the login details and buttons
        frame_details.place(rely=0.2, relx=0.0350, height=150, width=520)

        img_path = "img/re.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            image = image.resize((300, 200))
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_dash, image=photo, bd=0, background="#BB9AB1", command=lambda: rescan())
        button.image = photo
        button.pack()
        button.place(rely=0.450, relx=0.4)
        button.configure(activebackground="#BB9AB1", height=55, width=100)

        frame_func = tk.Frame(frame_dash, bg="#BB9AB1", relief="groove",
                              bd=0)  # this is the frame that holds all the login details and buttons
        frame_func.place(rely=0.530, relx=0.0300, height=220, width=520)

        frame_light = tk.Frame(frame_func, bg="#BB9AB1", relief="groove",
                               bd=0)  # this is the frame that holds all the login details and buttons
        frame_light.place(rely=0.031, relx=0.005, height=150, width=510)

        label_light = tk.Label(frame_func, text_styles1, text="select room lightning")
        label_light.place(rely=0.00001, relx=0.05)

        label_li = tk.Label(frame_light, text_styles2,
                               text="select AI ambient light recommendation or select your own colour")
        label_li.place(rely=0.07, relx=0.05)

        img_path = "img/select_song.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            image = image.resize((100, 60))
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_func, image=photo, bd=0, background="#BB9AB1", command=lambda: choose_li())
        button.image = photo
        button.pack()
        button.place(rely=0.25, relx=0.7)
        button.configure(activebackground="#BB9AB1", height=36, width=87)

        img_path = "img/ai_light.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            image = image.resize((100, 60))
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_func, image=photo, bd=0, background="#BB9AB1", command=lambda: getailight())
        button.image = photo
        button.pack()
        button.place(rely=0.25, relx=0.16)
        button.configure(activebackground="#BB9AB1", height=36, width=87)

        frame_music = tk.Frame(frame_func, bg="#BB9AB1", relief="groove",
                               bd=0)  # this is the frame that holds all the login details and buttons
        frame_music.place(rely=0.505, relx=0.005, height=100, width=510)

        label_music = tk.Label(frame_music, text_styles1, text="select Music")
        label_music.place(rely=0.01, relx=0.05)

        label_music = tk.Label(frame_music, text_styles2,
                               text="choose AI music recommendation or select your own favourite song from list")
        label_music.place(rely=0.32, relx=0.05)

        img_path = "img/select_song.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            image = image.resize((100, 60))
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_func, image=photo, bd=0, background="#BB9AB1", command=lambda: sel_mu())
        button.image = photo
        button.pack()
        button.place(rely=0.75, relx=0.7)
        button.configure(activebackground="#BB9AB1", height=36, width=87)

        img_path = "img/ai_music.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            image = image.resize((100, 60))
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_func, image=photo, bd=0, background="#BB9AB1", command=lambda: ai_music())
        button.image = photo
        button.pack()
        button.place(rely=0.75, relx=0.16)
        button.configure(activebackground="#BB9AB1", height=36, width=87)

        frame_read = tk.Frame(frame_dash, bg="#BB9AB1", relief="groove",
                              bd=0)  # this is the frame that holds all the login details and buttons
        frame_read.place(rely=0.8, relx=0.0400, height=60, width=520)

        if emotion == "happy":
            label_read = tk.Label(frame_read, title_style_read, text="How to increase our happy feeling",
                                  background="#BB9AB1")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "sad":
            label_read = tk.Label(frame_read, title_style_read, text="how to control our sad feeling",
                                  background="#BB9AB1")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "neutral":
            label_read = tk.Label(frame_read, title_style_read, text="how to increase our positive emotions",
                                  background="#BB9AB1")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "angry":
            label_read = tk.Label(frame_read, title_style_read, text=" How to controll our anger", background="#B9A6BA")
            label_read.place(rely=0.1, relx=0.05)

        label_read1 = tk.Label(frame_read, text_styles3, text="If you free  you can read this")
        label_read1.place(rely=0.5, relx=0.05)

        img_path = "img/read1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        read_btn = tk.Button(frame_read, image=photo, bd=0, background="#BB9AB1", command=lambda: read1())
        read_btn.image = photo
        read_btn.pack()
        read_btn.place(rely=0.15, relx=0.8, )
        read_btn.configure(activebackground="#BB9AB1")

        frame_read1 = tk.Frame(frame_dash, bg="#BB9AB1", relief="groove",
                               bd=0)  # this is the frame that holds all the login details and buttons
        frame_read1.place(rely=0.87, relx=0.0400, height=60, width=520)

        if emotion == "happy":
            label_read = tk.Label(frame_read1, title_style_read, text="importance of be happy", background="#B9A6BA")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "sad":
            label_read = tk.Label(frame_read1, title_style_read, text="importance of be avoid sad",
                                  background="#B9A6BA")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "neutral":
            label_read = tk.Label(frame_read1, title_style_read, text="how to increase our positive emotions",
                                  background="#B9A6BA")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "angry":
            label_read = tk.Label(frame_read1, title_style_read, text="importance of controlling our anger",
                                  background="#474849")
            label_read.place(rely=0.1, relx=0.05)

        label_read1 = tk.Label(frame_read1, text_styles3, text="If you free  you can read this")
        label_read1.place(rely=0.5, relx=0.05)
        label_read1.configure(activeforeground="#B9A6BA")

        img_path = "img/read1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        read1_btn = tk.Button(frame_read1, image=photo, bd=0, background="#BB9AB1", command=lambda: read2())
        read1_btn.image = photo
        read1_btn.pack()
        read1_btn.place(rely=0.25, relx=0.8)
        read_btn.configure(activebackground="#BB9AB1")

        frame_001 = tk.Frame(main_frame, relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)
        frame_001.configure(bg="#BB9AB1")

        label_read1.bind("Enter", on_enter)

        def rescan():
            dashboard.destroy(self)
            EmotionMusicApp()

        def getailight():
            dashboard.destroy(self)
            Ambient_ai()

        def choose_li():
            dashboard.destroy(self)
            sel_light()

        def ai_music():
            dashboard.destroy(self)
            player()

        def sel_mu():
            dashboard.destroy(self)
            selec_music()

        def manu1():
            dashboard.destroy(self)
            Setting1()

        def log():
            dashboard.destroy(self)
            LoginPage()

        def on_enter(event):
            read_btn.configure(foreground="red")

        def on_leave(event):
            read_btn.configure(foreground="black")

        def read1():
            readPage()
            dashboard.destroy(self)

        def read2():
            readPage()
            dashboard.destroy(self)

    def show_frame(self, Some_Widgets):
        pass


emotion = 'neutral'
if emotion == "neutral":
    c_emo = 7
if c_emo == 7:
    light_link = "img/ye1.png"
    mu_link = suggested_path
    name = "uptown"
    id_ = 3

def restart():
    global root
    root.destroy()
    root = tk.Tk()
    Main()


class Del_acc(tk.Tk):
    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#BB8493", height=800, width=600, bd=0) 
        main_frame.pack(fill="both", expand=5)

        self.geometry("1200x800")
        self.resizable(True, True) 
        title_styles = {"font": ("Trebuchet MS Bold", 30), "background": "white"}

        text_styles = {"font": ("Verdana", 10),
                       "background": "#BB8493",
                       "foreground": "#ffffff"}

        frame_del = tk.Frame(main_frame, bg="#BB8493", relief="groove",
                             bd=0) 
        frame_del.place(rely=0.3, relx=0.509, height=500, width=450)

        img_path = "img/back1.png"
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  

        ba_btn = tk.Button(main_frame, image=photo, bd=0, background="#BB8493", command=lambda: back())
        ba_btn.image = photo
        ba_btn.pack()
        ba_btn.place(rely=0.011, relx=0.01)
        ba_btn.configure(activeforeground="#BB8493")

        img_path = "img/bl.png" 

        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return 

        label = tk.Label(main_frame, image=photo, bd=0, background="#BB8493")
        label.image = photo
        label.pack()
        label.place(rely=0.008, relx=0.09)

        label_user = tk.Label(main_frame, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)

        label_title = tk.Label(frame_del, title_styles, text="Delete Account ")
        label_title.place(rely=0.1, relx=0.3)
        label_title.configure(background="#BB8493")

        label_user = tk.Label(frame_del, text_styles, text="Username:")
        label_user.place(rely=0.29, relx=0.05)

        label_pw = tk.Label(frame_del, text_styles, text="Password:")
        label_pw.place(rely=0.45, relx=0.05)

        entry_user = ttk.Entry(frame_del, width=40, cursor="xterm")
        entry_user.place(rely=0.3, relx=0.4)

        entry_pw = ttk.Entry(frame_del, width=40, cursor="xterm", show="*")
        entry_pw.place(rely=0.45, relx=0.4)

        button = ttk.Button(frame_del, text="Delete Account", command=lambda: del_acc())
        button.place(rely=0.55, relx=0.350, height=70, width=150)
        button.configure()

        label_re = tk.Label(frame_del, text_styles, text="Don't know password")
        label_re.place(rely=0.89, relx=0.01)
        label_re.configure(background="#BB8493", foreground="#ffffff")

        signup_btn = ttk.Button(frame_del, text="Reset ", command=lambda: reset_())
        signup_btn.place(rely=0.88, relx=0.420)
        signup_btn.configure()

        frame_001 = tk.Frame(main_frame, relief="groove",
                            bd=0) 
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)
        frame_001.configure(bg="#BB8493")

        def back():
            Del_acc.destroy(self)
            dashboard()

        def reset_():
            Del_acc.destroy(self)
            reset()

        def del_acc():
            username = entry_user.get()
            password = entry_pw.get()
            validation = validate1(username, password)
            if validation:
                del_account(username)

            else:
                tk.messagebox.showerror("Information", "The Username or Password you have entered are incorrect ")

        def validate1(username, password):
            query = "select* from user where User_Name = '{}' and password = '{}'".format(username, password)
            mycursor.execute(query)
            result = mycursor.fetchall()
            if result:
                return True
            else:
                return False

        def del_account(username, self=None):
            query = "DELETE FROM user WHERE User_Name = %s"
            try:
                mycursor.execute(query, (username,))
                mydb.commit() 

                affected_rows = mycursor.rowcount
                
                if affected_rows > 0:
                    tk.messagebox.showinfo(
                        "Account Delete Success",
                        f"{username}, your account has been deleted and all of your data cleared."
                    )
                    close() 
                else:
                    tk.messagebox.showinfo(
                        "Account Delete Unsuccessful",
                        f"No account found with the username '{username}'."
                    )
            except mysql.connector.Error as e:
                tk.messagebox.showerror("Database Error", f"An error occurred: {str(e)}")

        def close():
            if self is not None:
                Del_acc.destroy(self)
                Main()



class Ambient_ai(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#BB8493", height=800, width=600)
        main_frame.pack(fill="both", expand="20")

        text_styles = {"font": ("Verdana", 15),
                       "background": "#BB8493",
                       "foreground": "#ffffff"}

        self.geometry("1200x800")
        self.resizable(True, True)

        frame_light = tk.Frame(main_frame, relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_light.place(rely=0.01, relx=0.509, height=780, width=575)
        frame_light.configure(bg="#BB8493")

        label_light = tk.Label(frame_light, text_styles, text="Ambient light auto controlling")
        label_light.place(rely=0.25, relx=0.258)

        img_path = "img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_light, image=photo, bd=0, background="#BB8493", command=lambda: back())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.6)

        img_path = "img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(frame_light, image=photo, bd=0, background="#BB8493")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.06)


        label_user = tk.Label(frame_light, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)

        img_path = light_link

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(frame_light, image=photo, bd=0, background="#BB8493")
        label.image = photo
        label.pack()
        label.place(rely=0.33, relx=0.31)

        label_light = tk.Label(frame_light, text_styles, text="Not satisfied by light, then try your choice")
        label_light.place(rely=0.65, relx=0.258)

        select_btn = tk.Button(frame_light, text="Choose", command=lambda: select())
        select_btn.place(rely=0.7, relx=0.4, width=100, height=30)
        select_btn.configure(background="#BB8493", activebackground="pink", foreground="purple",
                             activeforeground="blue")
        
        frame_001 = tk.Frame(main_frame, relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)
        frame_001.configure(bg="#111111")

        def select():
            Ambient_ai.destroy(self)
            sel_light()

        def back():
            Ambient_ai.destroy(self)
            dashboard()


class sel_light(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#C8ACD6", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("1200x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "#C8ACD6",
                       "foreground": "#ffffff"}
        text_styles1 = {"font": ("Verdana", 10),
                        "background": "#C8ACD6",
                        "foreground": "#ffffff"}

        text_styles2 = {"font": ("Verdana", 6),
                        "background": "#43ffff",
                        "foreground": "#111111"}

        title_styles = {"font": ("arial-bold", 15),
                        "background": "#C8ACD6",
                        "foreground": "#ffffff"}
        

        frame_01 = tk.Frame(main_frame, bg="#C8ACD6", height=800, width=600)  # this is the background
        frame_01.pack(fill="both", expand="20")
        frame_01.place(rely=0.2, relx=0.509)


        label_fn = tk.Label(frame_01, title_styles, text="Select Your disired ambient light")
        label_fn.place(rely=0.2, relx=0.19)

        img_path = "img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_01, image=photo, bd=0, background="#C8ACD6", command=lambda: back())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)

        img_path = "img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(frame_01, image=photo, bd=0, background="#C8ACD6")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.09)

        label_user = tk.Label(frame_01, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)

        img_path = "img/gr11.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(frame_01, image=photo, bd=0, background="#C8ACD6", command=lambda: sel_1())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.33, relx=0.06)

        img_path = "img/libl.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(frame_01, image=photo, bd=0, background="#C8ACD6", command=lambda: sel_2())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.33, relx=0.46)

        img_path = "img/re11.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(frame_01, image=photo, bd=0, background="#C8ACD6", command=lambda: sel_3())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.33, relx=0.8)

        img_path = "img/bl11.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(frame_01, image=photo, bd=0, background="#C8ACD6", command=lambda: sel_4())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.45, relx=0.06)

        img_path = "img/pi11.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(frame_01, image=photo, bd=0, background="#C8ACD6", command=lambda: sel_5())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.45, relx=0.45)

        img_path = "img/ye11.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(frame_01, image=photo, bd=0, background="#C8ACD6", command=lambda: sel_6())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.45, relx=0.8)

        img_path = "img/gr11.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(frame_01, image=photo, bd=0, background="#C8ACD6", command=lambda: sel_7())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.45, relx=0.8)

        label_fn = tk.Label(frame_01, title_styles, text="Or try recommendation")
        label_fn.place(rely=0.55, relx=0.1)

        img_path = "img/pi11.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(frame_01, image=photo, bd=0, background="#C8ACD6", command=lambda: sel_5())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.6, relx=0.49)

        frame_001 = tk.Frame(main_frame, relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)
        frame_001.configure(bg="#111111")


        def sel_1():
            sel_light.destroy(self)
            assign(1)
            Ambient_se()

        def sel_2():
            sel_light.destroy(self)
            assign(2)
            Ambient_se()

        def sel_3():
            sel_light.destroy(self)
            assign(3)
            Ambient_se()

        def sel_4():
            sel_light.destroy(self)
            assign(4)
            Ambient_se()

        def sel_5():
            sel_light.destroy(self)
            assign(5)
            Ambient_se()

        def sel_6():
            sel_light.destroy(self)
            assign(6)
            Ambient_se()

        def sel_7():
            sel_light.destroy(self)
            assign(7)
            Ambient_se()

        def back():
            sel_light.destroy(self)
            dashboard()


light_link1 = "img/gr1"


def assign(nb):
    global light_link1
    if nb == 1:
        light_link1 = "img/gr2.png"
    elif nb == 2:
        light_link1 = "img/lib.png"
    elif nb == 3:
        light_link1 = "img/re1.png"
    elif nb == 4:
        light_link1 = "img/b1.png"
    elif nb == 5:
        light_link1 = "img/pi1.png"
    elif nb == 6:
        light_link1 = "img/ye1.png"
    else:
        light_link1 = "img/gr2.png"


class Ambient_se(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#B692C2", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        text_styles = {"font": ("Verdana", 15),
                       "background": "#B692C2",
                       "foreground": "#ffffff"}

        self.geometry("1200x800")
        self.resizable(True, True)

        frame_01 = tk.Frame(main_frame, bg="#C8ACD6", height=800, width=600)  # this is the background
        frame_01.pack(fill="both", expand="20")
        frame_01.place(rely=0.2, relx=0.509)

        label_light = tk.Label(frame_01, text_styles, text="Ambient light  controlling")
        label_light.place(rely=0.25, relx=0.258)

        img_path = "img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_01, image=photo, bd=0, background="#474849", command=lambda: back())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)

        img_path = "img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(frame_01, image=photo, bd=0, background="#B692C2")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.09)

        label_user = tk.Label(frame_01, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)

        img_path = light_link1

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(frame_01, image=photo, bd=0, background="#B692C2")
        label.image = photo
        label.pack()
        label.place(rely=0.33, relx=0.31)

        label_light = tk.Label(frame_01, text_styles, text="If not satisfy choose another colour")
        label_light.place(rely=0.65, relx=0.258)

        select_btn = tk.Button(frame_01, text="Choose", command=lambda: select())
        select_btn.place(rely=0.7, relx=0.4, width=100, height=30)
        select_btn.configure(background="#B692C2", activebackground="#B692C2", foreground="#f8a842",
                             activeforeground="blue")
        
        frame_001 = tk.Frame(main_frame, relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)
        frame_001.configure(bg="#111111")


        def select():
            Ambient_ai.destroy(self)
            sel_light()

        def back():
            Ambient_ai.destroy(self)
            dashboard()


def play_music():
    mixer.init()
    mixer.music.set_volume(0.5)
    mixer.music.load(suggested_path)
    mixer.music.play()


def stop():
    mixer.music.stop()


class player(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#987D9A", height=800, width=600, bd=1)
        main_frame.pack(fill="both", expand="true")
        self.title("Smart Emotion Detection System")
        self.geometry("1200x800")
        self.resizable(True, True)

        frame_001 = tk.Frame(main_frame, relief="groove", bd=0, bg="#111111")
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)

        self.camera_feed = CameraFeed(frame_001)   # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "white",
                       "foreground": "#000000"}
        text_styles1 = {"font": ("Verdana", 10),
                        "background": "#704264",
                        "foreground": "#111111"}

        text_styles2 = {"font": ("Verdana", 6),
                        "background": "#704264",
                        "foreground": "#111111"}

        title_styles = {"font": ("arial-bold", 15),
                        "background": "#ffffEE",
                        "foreground": "#111111"}
        
        frame_01 = tk.Frame(main_frame, bg="#BB9AB1", height=800, width=600)  # this is the background
        frame_01.pack(fill="both", expand="20")
        frame_01.place(rely=0.001, relx=0.509)

        label_fn = tk.Label(frame_01, title_styles, text="Music Player")
        label_fn.place(rely=0.2, relx=0.4)
        label_fn.configure(background="#B9A6BA")

        img_path = "img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_01, image=photo, bd=0, background="#BB9AB1", command=lambda: back_music())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)

        img_path = "img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(frame_01, image=photo, bd=0, background="#BB9AB1")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.09)

        label_user = tk.Label(frame_01, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)
        label_user.configure(background="#BB9AB1")

        song_label = tk.Label(frame_01, text="Music Player")
        song_label.pack()
        song_label.place(rely=0.1, relx=0.4)
        song_label.config(font=("Verdana", 15, "bold"), background="#BB9AB1", foreground="green")

        music = tk.Frame(frame_01, background="#121212", width=500, height=500)
        music.place(rely=0.15, relx=0.08)

        self.song_label = tk.Label(music, text="Now playing", font=("Verdana", 15), background="#121212",
                                   foreground="green")
        self.song_label.pack()
        self.song_label.place(rely=0.1, relx=0.35)

        hide = tk.Frame(music, bg="#121212", height=500, width=500)

        img_path1 = "img/giff.gif"

        try:
            info = Image.open(img_path1)
            frames = info.n_frames

            photoimage_objects = []
            for i in range(frames):
                info.seek(i)
                frame = info.copy().resize((250, 250))
                photoimage = ImageTk.PhotoImage(frame.convert("RGBA"))
                photoimage_objects.append(photoimage)

            current_frame = 0
            gif_label = Label(music, image=photoimage_objects[current_frame], )
            gif_label.pack()
            gif_label.place(relx=0.53, rely=0.5, anchor="center")
            gif_label.configure(background="#121212")

            def update_frame():
                nonlocal current_frame
                gif_label.config(image=photoimage_objects[current_frame])
                current_frame = (current_frame + 1) % frames
                self.after(50, update_frame)

        except FileNotFoundError:

            print(f"Error: Image file not found: {img_path}")

        img_path = "img/bar1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        labal1 = tk.Label(music, image=photo, bd=0, background="#121212", width=300, height=100)
        labal1.image = photo
        labal1.pack()
        labal1.place(rely=0.66, relx=0.19)

        img_path = "img/play1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        btn_play = tk.Button(music, image=photo, bd=0, width=39, height=35, command=lambda: play())
        btn_play.image = photo
        btn_play.pack()
        btn_play.place(rely=0.722, relx=0.45)
        btn_play.configure(activebackground="#343434")

        img_path = "img/stop1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        btn_stop = tk.Button(music, image=photo, bd=0, width=38, height=35, command=lambda: stop_song(self))
        btn_stop.image = photo
        btn_stop.pack()
        btn_stop.place(rely=0.722, relx=0.65)
        btn_stop.configure(activebackground="#343434")

        img_path = "img/pause11.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        btn_pause = tk.Button(music, image=photo, bd=0, width=36, height=33, command=lambda: pause())
        btn_pause.image = photo
        btn_pause.pack()
        btn_pause.place(rely=0.7315, relx=0.25)
        btn_pause.configure(activebackground="#343434")

        img_path = "img/favo.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        btn_fav = tk.Button(music, image=photo, bd=0, width=60, height=35, command=lambda: add())
        btn_fav.image = photo
        btn_fav.pack()
        btn_fav.place(rely=0.86, relx=0.45)
        btn_fav.configure(activebackground="#121212", bg="#121212")


        def back_music():
            player.destroy(self)
            dashboard()

        def stop_song(self):
            pygame.mixer.music.stop()
            self.song_label.config(text="Music Stopped")
            self.song_label.place(rely=0.1, relx=0.35)
            gif_label.place_forget()

        def play():
            song_name = os.path.basename(suggested_path)  
            self.song_label.config(text="Now playing: " + song_name)
            self.song_label.place(rely=0.1, relx=0.2)
            play_music()
            update_frame()
            gif_label.place(relx=0.53, rely=0.5, anchor="center")

        def pause():
            pygame.mixer.music.pause()
            self.song_label.config(text="Music Paused")
            self.song_label.place(rely=0.1, relx=0.35)
            gif_label.place_forget()

        def add():
            id1 = user_id
            query = "INSERT INTO favourite_list(Link, User_id) VALUES(%s, %s)"
            mycursor.execute(query, (suggested_path, id1))
            mydb.commit()



def play_song1(song_path):
    try:
        # Initialize Pygame audio mixer
        pygame.mixer.init()

        # Load the song using pygame.mixer
        pygame.mixer.music.load(song_path)

        # Start playback
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.event.pump()

    except Exception as e:
        print(f"Error playing song: {e}")
    finally:
        # Clean up Pygame resources (optional)
        pygame.mixer.quit()


class selec_music(tk.Tk):
    def __init__(self, frame_11=None, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#BB9AB1", height=800, width=1200)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("1200x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "#C8ACD6",
                       "foreground": "#111111"}
        text_styles1 = {"font": ("Verdana", 10),
                        "background": "#704264",
                        "foreground": "#111111"}

        text_styles2 = {"font": ("Verdana", 6),
                        "background": "#111111",
                        "foreground": "#ffffff"}

        title_styles = {"font": ("arial-bold", 15),
                        "background": "#C8ACD6",
                        "foreground": "#111111"}
        
        frame_01 = tk.Frame(main_frame,bg="#BB9AB1", height=800, width=600)  # this is the background
        frame_01.pack(fill="both", expand="20")
        frame_01.place(rely=0.001, relx=0.509)

        label_fn = tk.Label(frame_01, title_styles, text="Music Player")
        label_fn.place(rely=0.2, relx=0.4)
        label_fn.configure(background="#BB9AB1")

        img_path = "img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_01, image=photo, bd=0, background="#BB9AB1", command=lambda: back_music())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)

        img_path = "img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(frame_01, image=photo, bd=0, background="#704264")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.09)

        label_user = tk.Label(frame_01, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)
        label_user.configure(background="#704264")

        song_label = tk.Label(frame_01, text="Music Player")
        song_label.pack()
        song_label.place(rely=0.1, relx=0.3)
        song_label.config(font=("Verdana", 15, "bold"), background="#704264", foreground="green")

        music = tk.Frame(frame_01, background="#121212", width=500, height=500)
        music.place(rely=0.016, relx=0.05)

        self.song_label = tk.Label(music, text="Now playing", font=("Verdana", 15), background="#121212",
                                   foreground="green")
        self.song_label.pack()
        self.song_label.place(rely=0.1, relx=0.35)

        hide = tk.Frame(music, bg="#121212", height=500, width=500)

        img_path1 = "img/giff.gif"

        try:
            info = Image.open(img_path1)
            frames = info.n_frames

            photoimage_objects = []
            for i in range(frames):
                info.seek(i)
                frame = info.copy().resize((250, 250))
                photoimage = ImageTk.PhotoImage(frame.convert("RGBA"))
                photoimage_objects.append(photoimage)

            current_frame = 0
            gif_label = Label(music, image=photoimage_objects[current_frame], )
            gif_label.pack()
            gif_label.place(relx=0.53, rely=0.5, anchor="center")
            gif_label.configure(background="#121212")

            def update_frame():
                nonlocal current_frame
                gif_label.config(image=photoimage_objects[current_frame])
                current_frame = (current_frame + 1) % frames
                self.after(50, update_frame)

        except FileNotFoundError:

            print(f"Error: Image file not found: {img_path}")

        img_path = "img/bar1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        labal1 = tk.Label(music, image=photo, bd=0, background="#121212", width=300, height=100)
        labal1.image = photo
        labal1.pack()
        labal1.place(rely=0.66, relx=0.19)

        img_path = "img/play1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        btn_play = tk.Button(music, image=photo, bd=0, width=39, height=35, command=lambda: play())
        btn_play.image = photo
        btn_play.pack()
        btn_play.place(rely=0.722, relx=0.45)
        btn_play.configure(activebackground="#343434")

        img_path = "img/stop1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        btn_stop = tk.Button(music, image=photo, bd=0, width=38, height=35, command=lambda: stop_song(self))
        btn_stop.image = photo
        btn_stop.pack()
        btn_stop.place(rely=0.722, relx=0.65)
        btn_stop.configure(activebackground="#343434")

        img_path = "img/pause11.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        btn_pause = tk.Button(music, image=photo, bd=0, width=36, height=33, command=lambda: pause())
        btn_pause.image = photo
        btn_pause.pack()
        btn_pause.place(rely=0.7315, relx=0.25)
        btn_pause.configure(activebackground="#343434")


        frame_list = tk.Frame(main_frame,bg="#111111", height=150, width=580)  # this is the background
        frame_list.pack(fill="both", expand="20")
        frame_list.place(rely=0.8, relx=0.509)

        label_s1 = tk.Label(frame_list,text_styles2 ,text="01-------------Dark song ", cursor="hand2")
        label_s1.place(rely=0.1, relx=0.2)

        label_s2 = tk.Label(frame_list,text_styles2 ,text="02-------------Uptown ", cursor="hand2")
        label_s2.place(rely=0.3, relx=0.2)
        
        label_s3 = tk.Label(frame_list,text_styles2 ,text="03------------- Girls ", cursor="hand2")
        label_s3.place(rely=0.5, relx=0.2)


        button2 = tk.Button(frame_01, text="Add to Fav", background="#C8ACD6", command=lambda: add())
        button2.place(rely=0.75, relx=0.55)
        button2.configure(bg="gray", activeforeground="black", activebackground="#BB8493")
    
    

        frame_001 = tk.Frame(main_frame, relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)
        frame_001.configure(bg="#111111")

        label_s1.bind("<Button-1>", lambda e: play1())
        label_s2.bind("<Button-1>", lambda e: play2())
        label_s3.bind("<Button-1>", lambda e: play3())

        def back_music():
            selec_music.destroy(self)
            dashboard()

        def stop_song(self):
            pygame.mixer.music.stop()
            self.song_label.config(text="Music Stopped")
            self.song_label.place(rely=0.1, relx=0.35)
            gif_label.place_forget()

        def play():
            #update_gif(label)
            #play_song12(song_path1)
            song_path = song_path1
            self.song_label.config(text="Now playing: " + song_path, )
            self.song_label.place(rely=0.1, relx=0.2)
            play_music1()
            update_frame()
            gif_label.place(relx=0.53, rely=0.5, anchor="center")

        def pause():
            pygame.mixer.music.pause()
            self.song_label.config(text="Music Paused")
            self.song_label.place(rely=0.1, relx=0.35)
            gif_label.place_forget()

        def play1():
            global song_path1
            song_path1 = "mp3/dark.mp3"
            play()

        def play2():
            global song_path1
            song_path1 = "mp3/Uptown.mp3"
            play()

        def play3():
            global song_path1
            song_path1 = "mp3/Girls.mp3"
            play()


        def add():
            id1 = user_id
            query = " INSERT INTO favourite_list(Link, User_ID)  VALUES(%s,%d)", (suggested_path, id1)
            mycursor.execute(query)
            mydb.commit()



song_path1 = "mp3/dark.mp3"


def play_music1():
    mixer.init()
    mixer.music.set_volume(0.5)
    mixer.music.load(suggested_path)
    mixer.music.play()


class Setting1(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#C8ACD6", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("1200x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "#C8ACD6",
                       "foreground": "#000000"}
        text_styles1 = {"font": ("Verdana", 10),
                        "background": "#BB8493",
                        "foreground": "#111111"}

        manu_styles = {"font": ("Verdana", 15),
                       "background": "#C8ACD6",
                       "foreground": "#0000ff"}

        title_styles = {"font": ("arial-bold", 20),
                        "background": "#C8ACD6",
                        "foreground": "green"}
        


        frame_01 = tk.Frame(main_frame,bg="#C8ACD6", height=800, width=600)  # this is the background
        frame_01.pack(fill="both", expand="20")
        frame_01.place(rely=0.001, relx=0.509)

        img_path = "img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_01, image=photo, command=lambda: back_set(), bd=0)
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)
        button.configure(background="#C8ACD6", activebackground="#BB8493")

        img_path = "img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(frame_01, image=photo, bd=0, background="#C8ACD6")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.09)
        lable.configure(background="#C8ACD6")

        label_fn = tk.Label(frame_01, title_styles, text="Settings")
        label_fn.place(rely=0.15, relx=0.4)

        frame_de = tk.Frame(frame_01, bg="#C8ACD6", relief="groove", bd=0)
        frame_de.place(rely=0.25, relx=0.0290, height=600, width=560)

        label_s1 = tk.Label(frame_de, manu_styles, text="01   Update favourite list ", cursor="hand2")
        label_s1.place(rely=0.1, relx=0.17)
        label_s1.pack()

        label_s2 = tk.Label(frame_de, manu_styles, text="02   Recommendation History ", cursor="hand2")
        label_s2.place(rely=0.3, relx=0.17)
        label_s2.pack()

        label_s2 = tk.Label(frame_de, manu_styles, text="03   Delete Account ", cursor="hand2")
        label_s2.place(rely=0.5, relx=0.17)
        label_s2.pack()

        label_s1.bind("<Button-1>", lambda e: recomend())
        label_s2.bind("<Button-1>", lambda e: back_set())
        label_s2.bind("<Button-1>", lambda e: del_acc())

        text_frame = tk.Frame(frame_de, bg="#C8ACD6", relief="groove", bd=0)
        text_frame.place(rely=0.5, relx=0.15, height=200, width=400)

        text_instructions = tk.Text(text_frame, wrap=tk.WORD, width=45, height=200, font=("Verdana", 10), bd=1)
        text_instructions.insert(tk.INSERT, "Users can add or remove songs from their favorite list to personalize their music experience.\n"
"The system tracks and displays a history of detected emotions for the user to review past interactions.\n"
"Users can permanently delete their account and associated data for privacy and control.\n")
        text_instructions.config(state=tk.DISABLED)  # Disable editing of instructions
        text_instructions.config(bg="#C8ACD6", font=("Verdana", 10), fg="#ffffff")
        text_instructions.pack()

        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_instructions.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_instructions.config(yscrollcommand=scrollbar.set)
        text_instructions.pack(fill=tk.BOTH, expand=True)  # Fill available space

        frame_001 = tk.Frame(main_frame, relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)
        frame_001.configure(bg="#C8ACD6")

        def back_set():
            Setting1.destroy(self)
            dashboard()

        def del_acc():
            Setting1.destroy(self)
            Del_acc()

        def recomend():
            Setting1.destroy(self)
            Favo_up()


class Favo_up(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#C8ACD6", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("1200x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "#BB8493",
                       "foreground": "#000000"}
        text_styles1 = {"font": ("Verdana", 10),
                        "background": "#BB8493",
                        "foreground": "#111111"}

        manu_styles = {"font": ("Verdana", 10),
                       "background": "#ffffff",
                       "foreground": "#0000ff"}

        title_styles = {"font": ("arial-bold", 15),
                        "background": "#C8ACD6",
                        "foreground": "#111111"}
        

        frame_01 = tk.Frame(main_frame,bg="#C8ACD6", height=800, width=1200)  # this is the background
        frame_01.pack(fill="both", expand="20")
        frame_01.place(rely=0.001, relx=0.505)


        img_path = "img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_01, image=photo, bd=0, background="#C8ACD6", command=lambda: back_setting())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)

        img_path = "img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(frame_01, image=photo, bd=0, background="#C8ACD6")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.07)

        label_fn = tk.Label(frame_01, title_styles, text="Favourite Music")
        label_fn.place(rely=0.19, relx=0.17)

        tree = tk.ttk.Treeview(frame_01, columns=("name", "content id"), show="headings")
        tree.place(rely=0.3, relx=0.066)
        tree.column("#000000", minwidth=2, stretch=tk.NO)

        button1 = tk.Button(frame_01, text="show", background="#C8ACD6", command=lambda: display())
        button1.place(rely=0.25, relx=0.21)
        button1.configure(bg="#C8ACD6", activeforeground="pink", activebackground="#BB8493")

        frame_manu = tk.Frame(frame_01, bg="#C8ACD6", height=100, width=75)  # this is the background
        frame_manu.pack(fill="both", expand="20")
        frame_manu.place(rely=0.6, relx=0.25)

        frame_001 = tk.Frame(main_frame, relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_001.place(rely=0.01, relx=0.019, height=780, width=575)
        frame_001.configure(bg="#111111")

        entry_id = ttk.Entry(frame_01, width=40, cursor="xterm")
        entry_id.place(rely=0.7, relx=0.21)

        button2 = tk.Button(frame_01, text="show", background="#C8ACD6", command=lambda: remove())
        button2.place(rely=0.8, relx=0.21)
        button2.configure(bg="#C8ACD6", activeforeground="pink", activebackground="#BB8493")



        def display():
            id1 = user_id
            query = "SELECT * FROM `favourite_list` WHERE `User_ID` = %s"
            mycursor.execute(query, (id1,))
            data1 = mycursor.fetchall()
            tree.delete(*tree.get_children())

            for row in data1:
                print("Row data:", row) 

                file_path = row[1]

                if isinstance(file_path, str):
                    file_name = os.path.basename(file_path)
                else:
                    print(f"Invalid file path: {file_path}")
                    file_name = "Unknown File"

                tree.insert("", "end", values=(file_name,))

        def back_setting():
            Favo_up.destroy(self)
            Setting1()

        def get():
            id2 = entry_id.get()
            return id2


        def remove():
            id2 = get()
            query ="DELETE FROM favourite_list WHERE Favourite_ID = %s;"
            mycursor.execute(query, (id2))
            mydb.commit()
            result = mycursor.fetchall()  # Commit changes to database
            if result:

                tk.messagebox.showinfo("content delete unsuccessfull",
                                       " {} Your content not deleted.Try again ".format(
                                           id2))


            else:
                tk.messagebox.showinfo("Account delete success",
                                       " {} Your account has deleted and all of your data cleared. You can log in again ".format(
                                           id2))


class readPage(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.geometry("1200x800")
        self.resizable(True, True)

        main_frame = tk.Frame(self, bg="#BB8493", height=800, width=1200)
        main_frame.pack(fill="both", expand="20")

        text_styles = {"font": ("Verdana", 10),
                       "background": "#BB8493",
                       "foreground": "#ffffff"}

        title_styles = {"font": ("arial-bold", 15),
                        "background": "#BB8493",
                        "foreground": "#111111"}

        label_fn = tk.Label(main_frame, title_styles, text="Our Suggestion")
        label_fn.place(rely=0.2, relx=0.4)

        label_user = tk.Label(main_frame, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.07)

        frame_de = tk.Frame(main_frame, bg="#BB8493", relief="groove", bd=0)
        frame_de.place(rely=0.25, relx=0.0290, height=500, width=560)

        text_frame = tk.Frame(self, bg="#ffffff", relief="groove", bd=0)
        text_frame.place(rely=0.25, relx=0.04, height=500, width=545)

        text_instructions = tk.Text(text_frame, wrap=tk.WORD, width=45, height=40, font=("Verdana", 10), bd=0)
        text_instructions.insert(tk.INSERT,
                                 "Feeling neutral and looking for ways to boost your emotions in a \npositive direction? There are numerous activities and practices you can incorporate\n into your daily routine to uplift your mood and enhance your overall well-being."


                                 "\nEngaging in physical activities like walking, jogging, or yoga can release endorphins, which are natural\n mood enhancers. Exercise not only improves physical health but also has\n a significant impact on mental health, reducing stress and anxiety.\n Spending time outdoors and enjoying nature can also have a profound \neffect on your mental state. Whether it's a hike in the woods,\n a stroll in the park, or simply sitting in your backyard, connecting with nature can \nprovide a sense of peace and relaxation."


                                 "\nPracticing meditation and mindfulness helps you focus on the present moment and cultivate a positive mindset.\n These practices can reduce stress, improve concentration, and increase self-awareness. \nKeeping a gratitude journal is another effective way to shift your focus to the positive\n aspects of your life. By writing down things you are grateful for each day, you can foster a\n sense of appreciation and contentment."


                                 "Socializing with friends or family can provide emotional support and joy.\n Human connections are vital for emotional well-being, and spending quality \ntime with loved ones can significantly boost your mood. Whether it's a phone call,\n a video chat, or a face-to-face meeting, maintaining social connections is crucial for happiness.")
        text_instructions.config(state=tk.DISABLED)  # Disable editing of instructions
        text_instructions.config(bg="#ffffff", font=("Verdana", 10), fg="purple")

        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_instructions.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_instructions.config(yscrollcommand=scrollbar.set)
        text_instructions.pack(fill=tk.BOTH, expand=True)  # Fill available space

        btn_b = tk.Button(main_frame, text="close", command=lambda: back())
        btn_b.place(rely=0.9, relx=0.43)
        btn_b.configure(activeforeground="red", activebackground="pink", background="purple", foreground="white")

        def back():
            readPage.destroy(self)
            dashboard()





data = 0
q1 = 0
regex = '^[a-z0-9]+[._]?[a-z0-9]+[@]w+[.]w{2,3}$'
top = Main()
top.title(" SMART MIRROR")
root = tk.Tk()
root.withdraw()
root.title("ui smart mirror")
root.mainloop()
