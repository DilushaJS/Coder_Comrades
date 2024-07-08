import tkinter as tk
from doctest import master

import customtkinter as ctk
from customtkinter import *
import pygame
from PIL import Image, ImageTk
import mysql.connector
from tkinter import Label, PhotoImage, messagebox
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

"""
Useful Links:
https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter Most useful in my opinion
https://www.tutorialspoint.com/python/python_gui_programming.htm
https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/index.html
https://www.youtube.com/watch?v=HjNHATw6XgY&list=PLQVvvaa0QuDclKx-QpC9wntnURXVJqLyk
"""

# You can also use a pandas dataframe for pokemon_info.
# you can convert the dataframe using df.to_numpy.tolist()
ctk.set_default_color_theme("blue")

frame_styles = {"relief": "groove",
                "bd": 3, "bg": "#ffffff",
                "fg": "#073bb3", "font": ("Arial", 30, "bold")}

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="emotion"
)
mycursor = mydb.cursor()


class Main(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffff", height=800, width=600, bd=0)  # this is the background
        main_frame.pack(fill="both", expand="true")

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen
        title_styles = {"font": ("Trebuchet MS Bold", 20), "background": "white"}

        text_styles = {"font": ("Verdana", 6),
                       "background": "white",
                       "foreground": "#111111"}
        frame_11 = tk.Frame(main_frame, bg="#ffffff", relief="groove",
                            bd=1)  # this is the frame that holds all the login details and buttons
        frame_11.place(rely=0.01, relx=0.015, height=780, width=575)
        frame_11.configure(bg="#ffffff")

        img_path = "E:/gui/img/Start1.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            image1 = image.convert("RGBA")
            photo = ImageTk.PhotoImage(image1)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(frame_11, image=photo, bd=0, background="#ffffff", command=lambda: start())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.7, relx=0.4)

        #button = ttk.Button(frame_11, text="start", command=lambda: start())
        #button.place(rely=0.7, relx=0.4, height=100, width=150)

        frame_12 = tk.Frame(main_frame, bg="#ffffff", relief="groove",
                            bd=0)  # this is the frame that holds all the login details and buttons
        frame_12.place(rely=0.003, relx=0.015, height=250, width=570)

        img_path = "../gui/img/img.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            image1 = image.convert("RGBA")
            photo = ImageTk.PhotoImage(image1)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(frame_12, image=photo, bd=0, background="#ffffff")
        label.image = photo
        label.pack()
        label.place(rely=0.025, relx=0.33)

        label_title = tk.Label(frame_11, title_styles, text="Emotion recognition smart Mirror ")
        label_title.place(rely=0.4, relx=0.15)

        label_details = tk.Label(main_frame, text_styles, text="Emotion recognition smart Mirror ")
        label_details.place(rely=0.94, relx=0.40)

        label_details = tk.Label(main_frame, text_styles, text="By codercomrades ")
        label_details.place(rely=0.96, relx=0.45)

        def start():
            Main.destroy(self)
            LoginPage()


class LoginPage(tk.Tk):
    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffff", height=800, width=600, bd=0)  # this is the background
        main_frame.pack(fill="both", expand=5)

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen
        title_styles = {"font": ("Trebuchet MS Bold", 30), "background": "white"}

        text_styles = {"font": ("Verdana", 10),
                       "background": "white",
                       "foreground": "#111111"}

        frame_login = tk.Frame(main_frame, bg="#ffffff", relief="groove",
                               bd=0)  # this is the frame that holds all the login details and buttons
        frame_login.place(rely=0.3, relx=0.130, height=500, width=450)

        img_path = "E:/gui/img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        ba_btn = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: back())
        ba_btn.image = photo
        ba_btn.pack()
        ba_btn.place(rely=0.011, relx=0.01)

        img_path = "E:/gui/img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        label.image = photo
        label.pack()
        label.place(rely=0.008, relx=0.09)

        label_user = tk.Label(main_frame, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)

        label_title = tk.Label(frame_login, title_styles, text="Login ")
        label_title.place(rely=0.1, relx=0.4)

        label_user = tk.Label(frame_login, text_styles, text="Username:")
        label_user.place(rely=0.29, relx=0.05)

        label_pw = tk.Label(frame_login, text_styles, text="Password:")
        label_pw.place(rely=0.45, relx=0.05)

        entry_user = ttk.Entry(frame_login, width=40, cursor="xterm")
        entry_user.place(rely=0.3, relx=0.4)

        entry_pw = ttk.Entry(frame_login, width=40, cursor="xterm", show="*")
        entry_pw.place(rely=0.45, relx=0.4)

        # button = ttk.Button(frame_login, text="Login", command=lambda: getlogin())
        # button.place(rely=0.55, relx=0.350, height=70, width=150)

        img_path = "E:/gui/img/Login.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_login, image=photo, bd=0, background="#ffffff", command=lambda: getlogin())
        button.image = photo
        button.pack()
        button.place(rely=0.55, relx=0.350)

        label_re = tk.Label(frame_login, text_styles, text="don't know password")
        label_re.place(rely=0.89, relx=0.02)

        # signup_btn = ttk.Button(frame_login, text="Reset ", command=lambda: reset_())
        # signup_btn.place(rely=0.80, relx=0.420)

        img_path = "E:/gui/img/forget.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_login, image=photo, bd=0, background="#ffffff", command=lambda: reset_())
        button.image = photo
        button.pack()
        button.place(rely=0.88, relx=0.370)

        # signup_btn = ttk.Button(frame_login, text="Register", command=lambda: get_signup())
        # signup_btn.place(rely=0.75, relx=0.420)

        img_path = "E:/gui/img/Create.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_login, image=photo, bd=0, background="#ffffff", command=lambda: get_signup())
        button.image = photo
        button.pack()
        button.place(rely=0.78, relx=0.370)

        img_path = "E:/gui/img/img.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        label.image = photo
        label.pack()
        label.place(rely=0.06, relx=0.35)

        def back():
            LoginPage.destroy(self)
            Main()

        def reset_():
            LoginPage.destroy(self)
            reset()

        def get_signup():
            LoginPage.destroy(self)
            SignupPage()

        def getlogin():
            username = entry_user.get()
            password = entry_pw.get()
            # if your want to run the script as it is set validation = True
            validation = validate(username, password)
            if validation:
                tk.messagebox.showinfo("Login Successful",
                                       "Welcome {}".format(username))
                root.deiconify()
                LoginPage.destroy(self)
            else:
                tk.messagebox.showerror("Information", "The Username or Password you have entered are incorrect ")

        def validate(username, password):
            query = "select* from user where user_name = '{}' and password = '{}'".format(username, password)
            mycursor.execute(query)
            result = mycursor.fetchall()
            if result:
                return True
            else:
                return False


def on_change():
    pass


def check_1(val, email_v):
    if re.search(regex, val):
        email_v.configure(text="valid")
        return True
    else:
        email_v.configure(text="Invalid")
        return False



class SignupPage(tk.Tk):

    def __init__(self, check_email=None, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        self.root = root
        self.frame_sign = None
        main_frame = tk.Frame(self, bg="#ffffEE", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="true")

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 11, "bold"),
                       "background": "white",
                       "foreground": "#000000"}
        label_fn = tk.Label(main_frame, text_styles, text="Welcome to the SMART MIRROR:")
        label_fn.place(rely=0.017, relx=0.19)

        img_path = "E:/gui/img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        label.image = photo
        label.pack()
        label.place(rely=0.011, relx=0.01)

        img_path = "E:/gui/img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        label.image = photo
        label.pack()
        label.place(rely=0.008, relx=0.09)

        frame_sign = tk.Frame(main_frame, bg="#ffffff", relief="groove",
                              bd=1)  # this is the frame that holds all the login details and buttons
        frame_sign.place(rely=0.057, relx=0.133, height=700, width=450)

        def check(val, email_v, lable_v):
            regex1 = '^[a-z0-9]+[._]?[a-z0-9]+[@]w+[.]w{2,3}$'
            check_1(regex1, val)
            if check_1:
                self.lable_v.configure(text="valid")
            else:
                self.lable_v.configure(text="Invalid")

        label_fn = tk.Label(frame_sign, text_styles, text="First Nmae:")
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

        self.validation_label = tk.Label(self, text="", fg="#111111", bg="#ffffff")
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

        entry_user = ttk.Entry(frame_sign, width=20, cursor="xterm")
        entry_user.place(rely=0.3, relx=0.5)

        entry_pw = ttk.Entry(frame_sign, width=20, cursor="xterm", show="*")
        entry_pw.place(rely=0.4, relx=0.5)

        label_user = tk.Label(frame_sign, text_styles, text="Already have a account")
        label_user.place(rely=0.95, relx=0.1)

        button_login = ttk.Button(frame_sign, text="Login", width=15, command=lambda: get_login())
        button_login.place(rely=0.95, relx=0.54)

        img_path = "E:/gui/img/Create.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(frame_sign, image=photo, bd=0, background="#ffffff", command=lambda: signup())
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
            validate_email_handler(self)
            validation = validate_u(user)
            if em:
                if not validation:
                    tk.messagebox.showerror("Information", "That Username already exists")
                else:
                    if not pw or not ln or not fn or not email or not user:
                        tk.messagebox.showerror("Error", "Please enter your details first", )
                    else:
                        if len(pw) > 5:
                            mycursor.execute(
                                "INSERT INTO user (User_name, first_name, last_name, e_mail, date_of_birth,"
                                " password) VALUES(%s, %s, %s, %s, %s, %s)",
                                (user, fn, ln, email1, dob, pw))
                            mydb.commit()
                            tk.messagebox.showinfo("Information", "Your account has been created!")
                            SignupPage.destroy(self)
                            LoginPage()

                        else:
                            tk.messagebox.showerror("Information", "Your password needs to be longer than 3 values.")

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
                self.validation_label.config(text="Valid email address", fg="green")
                return True

            else:
                self.validation_label.config(text="Invalid email address", fg="red")
                return False




def validate_email(email):
    """
    Validates an email address using a regular expression.

    Args:
        email (str): The email address to validate.

    Returns:
        bool: True if the email is valid, False otherwise.
    """

    regex1 = r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$"
    if re.fullmatch(regex1, email):
        return True
    else:
        return False




height = 50
corner_radius = 10


class reset(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffEE", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="true")

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "white",
                       "foreground": "#000000"}
        label_fn = tk.Label(main_frame, text_styles, text="Reset password:")
        label_fn.place(rely=0.001, relx=0.15)

        img_path = "E:/gui/img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        label.image = photo
        label.pack()
        label.place(rely=0.011, relx=0.01)

        img_path = "E:/gui/img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        label.image = photo
        label.pack()
        label.place(rely=0.008, relx=0.09)

        label_user = tk.Label(main_frame, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)

        frame_reset = tk.Frame(main_frame, bg="#ffffff", relief="groove",
                               bd=1)  # this is the frame that holds all the login details and buttons
        frame_reset.place(rely=0.049, relx=0.155, height=700, width=450)

        label_user = tk.Label(frame_reset, text_styles, text="Username:")
        label_user.place(rely=0.3, relx=0.2)

        entry_user = ttk.Entry(frame_reset, width=20, cursor="xterm")
        entry_user.place(rely=0.3, relx=0.5)

        button = ttk.Button(frame_reset, text="Search", width=15, command=lambda: search())
        button.place(rely=0.8, relx=0.4)

        def search():
            user1 = entry_user.get()
            validation = validate_us(user1)
            if not validation:
                tk.messagebox.showerror("Information", "That Username Not exists")
            else:
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
                return False


class ch_pass(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        main_frame = tk.Frame(self, bg="#ffffEE", height=500, width=430)  # this is the background
        main_frame.pack(fill="both", expand="true")

        self.geometry("500x430")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "white",
                       "foreground": "#000000"}
        label_fn = tk.Label(main_frame, text_styles, text="Answer Security question:")
        label_fn.place(rely=0.08, relx=0.15)

        img_path = "E:/gui/img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        label.image = photo
        label.pack()
        label.place(rely=0.011, relx=0.01)

        img_path = "E:/gui/img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        label.image = photo
        label.pack()
        label.place(rely=0.008, relx=0.09)

        label_user = tk.Label(main_frame, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)

        frame_reset1 = tk.Frame(main_frame, bg="#ffffff", relief="groove", bd=1)
        frame_reset1.place(rely=0.08, relx=0.12, height=410, width=405)

        label_fn = tk.Label(frame_reset1, text_styles, text=":")
        label_fn.place(rely=0.08, relx=0.15)


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

        main_frame = tk.Frame(self, bg="#ffffEE", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
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

        label_fn = tk.Label(main_frame, text_styles, text="Welcome to the SMART MIRROR:")
        label_fn.place(rely=0.001, relx=0.15)

        canvas = tk.Canvas(main_frame, bg="#ffffff", height=500, width=600, bd=1)
        canvas.pack(fill="both", expand="20")

        canvas.create_oval(500, 500, 220, 170, outline="#000000", width=2)

        update_frame()


class dashboard(tk.Tk):
    def __init__(self, show_pop1=None, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffEE", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
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

        label_fn = tk.Label(main_frame, text_styles, text="Welcome to the SMART MIRROR:")
        label_fn.place(rely=0.001, relx=0.15)

        label_user = tk.Label(main_frame, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)

        btn_manu = tk.Button(main_frame, text="Manu", command=lambda: manu1())
        btn_manu.place(rely=0.01, relx=0.9, width=100, height=30)

        frame_dash = tk.Frame(main_frame, bg="#ffffff", relief="groove", bd=1)
        frame_dash.place(rely=0.090, relx=0.0155, height=750, width=580)

        label_fn = tk.Label(frame_dash, text_styles, text=f"you are currently in:{emotion}")
        label_fn.place(rely=0.005, relx=0.05)

        frame_de = tk.Frame(frame_dash, bg="#ffffff", relief="groove",
                            bd=1)  # this is the frame that holds all the login details and buttons
        frame_de.place(rely=0.0390, relx=0.0290, height=220, width=520)

        frame_details = tk.LabelFrame(frame_dash, bg="#ffffff", relief="groove", bd=0,
                                      text="aaaaaaadddddddddffghjjjj")  # this is the frame that holds all the login details and buttons
        frame_details.place(rely=0.051, relx=0.0350, height=200, width=500)

        rescan_btn = tk.Button(main_frame, text="Scan again", command=lambda: rescan())
        rescan_btn.place(rely=0.430, relx=0.450)

        frame_func = tk.Frame(main_frame, bg="#ffffff", relief="groove",
                              bd=1)  # this is the frame that holds all the login details and buttons
        frame_func.place(rely=0.470, relx=0.0400, height=220, width=520)

        frame_light = tk.Frame(frame_func, bg="#43ffff", relief="groove",
                               bd=1)  # this is the frame that holds all the login details and buttons
        frame_light.place(rely=0.035, relx=0.005, height=100, width=510)

        label_light = tk.Label(frame_light, text_styles1, text="select room lightning")
        label_light.place(rely=0.01, relx=0.05)

        label_music = tk.Label(frame_light, text_styles2,
                               text="select AI abient light recomendation or select your own colour")
        label_music.place(rely=0.32, relx=0.05)

        li_ai_btn = tk.Button(frame_light, text="Change light", command=lambda: getailight())
        li_ai_btn.place(rely=0.68, relx=0.10, width=100, height=30)

        seli_btn = tk.Button(frame_light, text="Choose colour", command=lambda: choose_li())
        seli_btn.place(rely=0.68, relx=0.73, width=100, height=30)

        frame_music = tk.Frame(frame_func, bg="#43ffff", relief="groove",
                               bd=1)  # this is the frame that holds all the login details and buttons
        frame_music.place(rely=0.505, relx=0.005, height=100, width=510)

        label_music = tk.Label(frame_music, text_styles1, text="select Music")
        label_music.place(rely=0.01, relx=0.05)

        label_music = tk.Label(frame_music, text_styles2,
                               text="choose AI music recommendation or select your own favourite song from list")
        label_music.place(rely=0.32, relx=0.05)

        mu_ai_btn = tk.Button(frame_music, text="make me happy", command=lambda: ai_music())
        mu_ai_btn.place(rely=0.6, relx=0.10, width=100, height=30)

        mu_sel_btn = tk.Button(frame_music, text="Choose music", command=lambda: rescan())
        mu_sel_btn.place(rely=0.6, relx=0.73, width=100, height=30)

        label_read = tk.Label(main_frame, text_styles, text=f"you are currently in:{emotion}")
        label_read.place(rely=0.74, relx=0.05)

        frame_read = tk.Frame(main_frame, bg="yellow", relief="groove",
                              bd=1)  # this is the frame that holds all the login details and buttons
        frame_read.place(rely=0.77, relx=0.0400, height=60, width=520)

        if emotion == "happy":
            label_read = tk.Label(frame_read, title_style_read, text="How to increase our happy feeling")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "sad":
            label_read = tk.Label(frame_read, title_style_read, text="how to control our sad feeling")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "neutral":
            label_read = tk.Label(frame_read, title_style_read, text="how to increase our positive emotions")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "angry":
            label_read = tk.Label(frame_read, title_style_read, text=" How to controll our anger")
            label_read.place(rely=0.1, relx=0.05)

        label_read1 = tk.Label(frame_read, text_styles3, text="If you free  you can read this")
        label_read1.place(rely=0.5, relx=0.05)

        read_btn = tk.Button(frame_read, text="read more", command=lambda: rescan())
        read_btn.place(rely=0.4, relx=0.7, width=100, height=30)

        frame_read1 = tk.Frame(main_frame, bg="yellow", relief="groove",
                               bd=1)  # this is the frame that holds all the login details and buttons
        frame_read1.place(rely=0.87, relx=0.0400, height=60, width=520)

        if emotion == "happy":
            label_read = tk.Label(frame_read1, title_style_read, text="importance of be happy")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "sad":
            label_read = tk.Label(frame_read1, title_style_read, text="importance of be avoid sad")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "neutral":
            label_read = tk.Label(frame_read1, title_style_read, text="how to increase our positive emotions")
            label_read.place(rely=0.1, relx=0.05)
        elif emotion == "angry":
            label_read = tk.Label(frame_read1, title_style_read, text="importance of controlling our anger")
            label_read.place(rely=0.1, relx=0.05)

        label_read1 = tk.Label(frame_read1, text_styles3, text="If you free  you can read this")
        label_read1.place(rely=0.5, relx=0.05)

        read1_btn = tk.Button(frame_read1, text="read more", command=lambda: rescan())
        read1_btn.place(rely=0.4, relx=0.7, width=100, height=30)

        def rescan():
            dashboard()

        def getailight():
            dashboard.destroy(self)
            Ambient_ai()

        def choose_li():
            dashboard.destroy(self)
            sel_light()

        def ai_music():
            dashboard.destroy(self)
            player()

        def manu1():
            dashboard.destroy(self)
            manu()

    def show_frame(self, Some_Widgets):
        pass


emotion = 'angry'
if emotion == 'sad':
    c_emo = 5
elif emotion == "happy":
    c_emo = 6
elif emotion == "nutral":
    c_emo = 7
elif emotion == "anger":
    c_emo = 8
else:
    c_emo = 9
if c_emo == 5:
    light_link = "img/gr.png"
elif c_emo == 6:
    light_link = "img/bl.png"
elif c_emo == 7:
    light_link = "img/ye.png"
elif c_emo == 8:
    light_link = "img/pi.png"
else:
    light_link = "img/li_bl.png"


def restart():
    global root
    root.destroy()
    root = tk.Tk()
    Main()


class Del_acc(tk.Tk):
    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffff", height=800, width=600, bd=0)  # this is the background
        main_frame.pack(fill="both", expand=5)

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen
        title_styles = {"font": ("Trebuchet MS Bold", 30), "background": "white"}

        text_styles = {"font": ("Verdana", 10),
                       "background": "white",
                       "foreground": "#111111"}

        frame_del = tk.Frame(main_frame, bg="#ffffff", relief="groove",
                             bd=0)  # this is the frame that holds all the login details and buttons
        frame_del.place(rely=0.3, relx=0.130, height=500, width=450)

        img_path = "E:/gui/img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        ba_btn = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: back())
        ba_btn.image = photo
        ba_btn.pack()
        ba_btn.place(rely=0.011, relx=0.01)

        img_path = "E:/gui/img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        label = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        label.image = photo
        label.pack()
        label.place(rely=0.008, relx=0.09)

        label_user = tk.Label(main_frame, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)

        label_title = tk.Label(frame_del, title_styles, text="Login ")
        label_title.place(rely=0.1, relx=0.4)

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

        label_re = tk.Label(frame_del, text_styles, text="don't know password")
        label_re.place(rely=0.89, relx=0.02)

        signup_btn = ttk.Button(frame_del, text="Reset ", command=lambda: reset_())
        signup_btn.place(rely=0.88, relx=0.420)

        def back():
            Del_acc.destroy(self)
            dashboard()

        def reset_():
            Del_acc.destroy(self)
            reset()

        def del_acc():
            username = entry_user.get()
            password = entry_pw.get()
            # if your want to run the script as it is set validation = True
            validation = validate1(username, password)
            if validation:
                del_account(username)

            else:
                tk.messagebox.showerror("Information", "The Username or Password you have entered are incorrect ")

        def validate1(username, password):
            query = "select* from user where user_name = '{}' and password = '{}'".format(username, password)
            mycursor.execute(query)
            result = mycursor.fetchall()
            if result:
                return True
            else:
                return False

        def del_account(username, self=None):
            # Prepared statement for deletion to prevent SQL injection
            query = "DELETE FROM user WHERE User_name = %s"
            mycursor.execute(query, (username,))
            mydb.commit()
            result = mycursor.fetchall()  # Commit changes to database
            if result:

                tk.messagebox.showinfo("Account delete unsuccessfully",
                                       " {} Your account not deleted try again ".format(
                                           username))


            else:
                tk.messagebox.showinfo("Account delete success",
                                       " {} Your account has deleted and all of your data cleared You can log in again ".format(
                                           username))
                close()
                # restart()

        def close():
            Del_acc.destroy(self)
            Main()


class Ambient_ai(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffff", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        text_styles = {"font": ("Verdana", 15),
                       "background": "white",
                       "foreground": "#000000"}

        self.geometry("600x800")
        self.resizable(True, True)

        label_light = tk.Label(main_frame, text_styles, text="Ambient light auto controlling")
        label_light.place(rely=0.25, relx=0.258)

        img_path = "E:/gui/img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: back())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)

        img_path = "E:/gui/img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.09)

        label_user = tk.Label(main_frame, text_styles, text="Smart Mirror ")
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
        label = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        label.image = photo
        label.pack()
        label.place(rely=0.33, relx=0.31)

        label_light = tk.Label(main_frame, text_styles, text="not satisfied by light then try your choice")
        label_light.place(rely=0.65, relx=0.258)

        select_btn = tk.Button(main_frame, text="Choose", command=lambda: select())
        select_btn.place(rely=0.7, relx=0.4, width=100, height=30)

        def select():
            Ambient_ai.destroy(self)
            sel_light()

        def back():
            Ambient_ai.destroy(self)
            dashboard()


class sel_light(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffEE", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
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

        title_styles = {"font": ("arial-bold", 15),
                        "background": "#ffffEE",
                        "foreground": "#111111"}
        label_fn = tk.Label(main_frame, title_styles, text="Select Your disired ambient light")
        label_fn.place(rely=0.2, relx=0.19)

        img_path = "E:/gui/img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: back())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)

        img_path = "E:/gui/img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.09)

        label_user = tk.Label(main_frame, text_styles, text="Smart Mirror ")
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
        buttton = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: sel_1())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.33, relx=0.06)

        img_path = light_link

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: sel_2())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.33, relx=0.46)

        img_path = light_link

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: sel_3())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.33, relx=0.8)

        img_path = light_link

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: sel_4())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.45, relx=0.06)

        img_path = light_link

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: sel_5())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.45, relx=0.45)

        img_path = light_link

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: sel_6())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.45, relx=0.8)

        img_path = light_link

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: sel_7())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.45, relx=0.8)

        label_fn = tk.Label(main_frame, title_styles, text="Or try recomendation")
        label_fn.place(rely=0.55, relx=0.1)

        img_path = light_link

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: sel_6())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.65, relx=0.5)

        def sel_1():
            sel_light.destroy(self)
            sel_light()

        def sel_2():
            sel_light.destroy(self)
            sel_light()

        def sel_3():
            sel_light.destroy(self)
            sel_light()

        def sel_4():
            sel_light.destroy(self)
            sel_light()

        def sel_5():
            sel_light.destroy(self)
            sel_light()

        def sel_6():
            sel_light.destroy(self)
            sel_light()

        def sel_7():
            sel_light.destroy(self)
            sel_light()

        def back():
            sel_light.destroy(self)
            dashboard()


music_link = ""


def play_music():
    mixer.init()
    mixer.music.set_volume(0.5)
    mixer.music.load("mp3/song.mp3")
    mixer.music.play()


def stop():
    mixer.music.stop()


class player(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffEE", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
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

        title_styles = {"font": ("arial-bold", 15),
                        "background": "#ffffEE",
                        "foreground": "#111111"}
        label_fn = tk.Label(main_frame, title_styles, text="Music Player")
        label_fn.place(rely=0.2, relx=0.4)

        img_path = "E:/gui/img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: back_music())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)

        img_path = "E:/gui/img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.09)

        label_user = tk.Label(main_frame, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)

        song_label = tk.Label(main_frame, text="Music Player")
        song_label.pack()

        playbar = tk.Scale(main_frame, orient='horizontal', length=200, from_=0, to=100)
        playbar.pack()

        addButton = tk.Button(main_frame, text="Add Song", command=lambda: add)
        addButton.pack()

        playButton = tk.Button(main_frame, text="Play", command=lambda: play_song)
        playButton.pack()

        stopButton = tk.Button(main_frame, text="Stop", command=lambda: stop_song)
        stopButton.pack()

        nextButton = tk.Button(main_frame, text="Next Song", command=lambda: next_song)
        nextButton.pack()

        def back_music():
            music_player.destroy(self)
            dashboard()

        def add(self):
            song = filedialog.askopenfilename(filetypes=[("Music files", "*.mp3")])
            self.playlist.append(song)

        def play_song(self):
            if len(self.playlist) > 0:
                pygame.mixer.music.load(self.playlist[self.current_song_index])
                pygame.mixer.music.play()
                self.song_label.config(text="Now Playing: " + self.playlist[self.current_song_index])
                self.playbar.config(to=pygame.mixer.Sound(self.playlist[self.current_song_index]).get_length())
                self.root.after(1000, self.update_playbar)
            else:
                self.song_label.config(text="No songs in playlist")

        def stop_song(self):
            pygame.mixer.music.stop()
            self.song_label.config(text="Music Stopped")

        def next_song(self):
            if len(self.playlist) > 0:
                self.current_song_index += 1
                if self.current_song_index >= len(self.playlist):
                    self.current_song_index = 0
                pygame.mixer.music.load(self.playlist[self.current_song_index])
                pygame.mixer.music.play()
                self.song_label.config(text="Now Playing: " + self.playlist[self.current_song_index])
                self.playbar.config(to=pygame.mixer.Sound(self.playlist[self.current_song_index]).get_length())
                self.root.after(1000, self.update_playbar)

        def update_playbar(self):
            current_time = pygame.mixer.music.get_pos() / 1000
            self.playbar.set(current_time)
            self.root.after(1000, self.update_playbar)


class music_player(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffEE", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
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

        title_styles = {"font": ("arial-bold", 15),
                        "background": "#ffffEE",
                        "foreground": "#111111"}
        label_fn = tk.Label(main_frame, title_styles, text="Music Player")
        label_fn.place(rely=0.2, relx=0.4)

        img_path = "E:/gui/img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: back_music())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)

        img_path = "E:/gui/img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.09)

        label_user = tk.Label(main_frame, text_styles, text="Smart Mirror ")
        label_user.place(rely=0.0197, relx=0.17)

        frame_de = tk.Frame(main_frame, bg="#ffffff", relief="groove",
                            bd=1)
        frame_de.place(rely=0.25, relx=0.0290, height=200, width=560)

        img_path = "img/play.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(frame_de, image=photo, bd=0, background="#ffffff", command=lambda: play())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.8, relx=0.4)

        img_path = "img/pause.png"

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        buttton = tk.Button(frame_de, image=photo, bd=0, background="#ffffff", command=lambda: stop())
        buttton.image = photo
        buttton.pack()
        buttton.place(rely=0.8, relx=0.6)

        def play():
            play_music()

        def stop():
            stop()

        def back_music():
            music_player.destroy(self)
            dashboard()


class Setting1(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffEE", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "white",
                       "foreground": "#000000"}
        text_styles1 = {"font": ("Verdana", 10),
                        "background": "#43ffff",
                        "foreground": "#111111"}

        manu_styles = {"font": ("Verdana", 8),
                       "background": "#ffffff",
                       "foreground": "#0000ff"}

        title_styles = {"font": ("arial-bold", 15),
                        "background": "#ffffEE",
                        "foreground": "#111111"}

        img_path = "E:/gui/img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(main_frame, image=photo, command=lambda: back_set())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)

        img_path = "E:/gui/img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.09)

        label_fn = tk.Label(main_frame, title_styles, text="Settings")
        label_fn.place(rely=0.2, relx=0.4)

        frame_de = tk.Frame(main_frame, bg="#ffffff", relief="groove", bd=1)
        frame_de.place(rely=0.25, relx=0.0290, height=200, width=560)

        label_s1 = tk.Label(frame_de, manu_styles, text="01   Update favourite list ", cursor="hand2")
        label_s1.place(rely=0.1, relx=0.17)
        label_s1.pack()

        label_s2 = tk.Label(frame_de, manu_styles, text="02   Recommendation History ", cursor="hand2")
        label_s2.place(rely=0.2, relx=0.17)
        label_s2.pack()

        label_s2 = tk.Label(frame_de, manu_styles, text="03   Delete Account ", cursor="hand2")
        label_s2.place(rely=0.3, relx=0.17)
        label_s2.pack()

        label_s1.bind("<Button-1>", lambda e: back_set())
        label_s2.bind("<Button-1>", lambda e: back_set())
        label_s2.bind("<Button-1>", lambda e: del_acc())

        def back_set():
            Setting1.destroy(self)

        def del_acc():
            Setting1.destroy(self)
            Del_acc()


class manu(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffEE", height=800, width=600)  # this is the background
        main_frame.pack(fill="both", expand="20")

        self.geometry("600x800")  # Sets window size to 626w x 431h pixels
        self.resizable(True, True)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "white",
                       "foreground": "#000000"}
        text_styles1 = {"font": ("Verdana", 10),
                        "background": "#43ffff",
                        "foreground": "#111111"}

        manu_styles = {"font": ("Verdana", 10),
                       "background": "#ffffff",
                       "foreground": "#0000ff"}

        title_styles = {"font": ("arial-bold", 15),
                        "background": "#ffffEE",
                        "foreground": "#111111"}

        img_path = "E:/gui/img/bg.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        bg = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        bg.image = photo
        bg.pack()
        bg.place(rely=0, relx=0)

        img_path = "E:/gui/img/back1.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        button = tk.Button(main_frame, image=photo, bd=0, background="#ffffff", command=lambda: back_setting())
        button.image = photo
        button.pack()
        button.place(rely=0.011, relx=0.01)

        img_path = "E:/gui/img/bl.png"  # Adjust the path accordingly

        # Try loading the image
        try:
            image = Image.open(img_path)
            photo = ImageTk.PhotoImage(image)
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            return  # Exit if image is not found

        # Create a label to display the image
        lable = tk.Label(main_frame, image=photo, bd=0, background="#ffffff")
        lable.image = photo
        lable.pack()
        lable.place(rely=0.008, relx=0.09)

        label_fn = tk.Label(main_frame, title_styles, text="Manu")
        label_fn.place(rely=0.1, relx=0.4)

        frame_manu = tk.Frame(main_frame, bg="#ffffEE", height=100, width=75)  # this is the background
        frame_manu.pack(fill="both", expand="20")
        frame_manu.place(rely=0.3, relx=0.25)

        label_s1 = tk.Label(frame_manu, manu_styles, text="01   Settings ", cursor="hand2")
        label_s1.place(rely=0.1, relx=0.17)
        label_s1.pack()
        label_s1.bind("<Button-1>", lambda e: setting())

        def setting():
            manu.destroy(self)
            Setting1()

        def back_setting():
            manu.destroy(self)


class MenuBar(tk.Menu):
    def __init__(self, parent):
        tk.Menu.__init__(self, parent)

        menu_file = tk.Menu(self, tearoff=0)
        self.add_cascade(label="Menu1", menu=menu_file)
        menu_file.add_command(label="All Widgets", command=lambda: parent.show_frame(Some_Widgets))
        menu_file.add_separator()
        menu_file.add_command(label="Exit Application", command=lambda: parent.Quit_application())

        menu_orders = tk.Menu(self, tearoff=0)
        self.add_cascade(label="Menu2", menu=menu_orders)

        menu_pricing = tk.Menu(self, tearoff=0)
        self.add_cascade(label="Menu3", menu=menu_pricing)
        menu_pricing.add_command(label="Page One", command=lambda: parent.show_frame(PageOne))

        menu_operations = tk.Menu(self, tearoff=0)
        self.add_cascade(label="Menu4", menu=menu_operations)
        menu_operations.add_command(label="Page Two", command=lambda: parent.show_frame(PageTwo))
        menu_positions = tk.Menu(menu_operations, tearoff=0)
        menu_operations.add_cascade(label="Menu5", menu=menu_positions)
        menu_positions.add_command(label="Page Three", command=lambda: parent.show_frame(PageThree))
        menu_positions.add_command(label="Page Four", command=lambda: parent.show_frame(PageFour))

        menu_help = tk.Menu(self, tearoff=0)
        self.add_cascade(label="Menu6", menu=menu_help)
        menu_help.add_command(label="Open New Window", command=lambda: parent.OpenNewWindow())


class MyApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        main_frame = tk.Frame(self, bg="#84CEEB", height=600, width=1024)
        main_frame.pack_propagate(0)
        main_frame.pack(fill="both", expand="1", padx=5, pady=5)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        # self.resizable(0, 0) prevents the app from being resized
        # self.geometry("1024x600") fixes the applications size
        self.frames = {}
        pages = (Some_Widgets, PageOne, PageTwo, PageThree, PageFour)
        for F in pages:
            frame = F(main_frame, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(Some_Widgets)
        menubar = MenuBar(self)
        tk.Tk.config(self, menu=menubar)

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()

    def OpenNewWindow(self):
        OpenNewWindow()

    def Quit_application(self):
        self.destroy()


class GUI(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.main_frame = tk.Frame(self, bg="#BEB2A7", height=600, width=1024)
        # self.main_frame.pack_propagate(0)
        self.main_frame.pack(fill="both", expand="true")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)


class Some_Widgets(GUI):  # inherits from the GUI class
    def __init__(self, parent, controller):
        GUI.__init__(self, parent)

        frame1 = tk.LabelFrame(self, frame_styles, text="This is a LabelFrame containing a Treeview")
        frame1.place(rely=0.05, relx=0.02, height=400, width=400)

        frame2 = tk.LabelFrame(self, frame_styles, text="Some widgets")
        frame2.place(rely=0.05, relx=0.45, height=500, width=500)

        button1 = tk.Button(frame2, text="tk button", command=lambda: Refresh_data())
        button1.pack()
        button2 = ttk.Button(frame2, text="ttk button", command=lambda: Refresh_data())
        button2.pack()

        Var1 = tk.IntVar()
        Var2 = tk.IntVar()
        Cbutton1 = tk.Checkbutton(frame2, text="tk CheckButton1", variable=Var1, onvalue=1, offvalue=0)
        Cbutton1.pack()
        Cbutton2 = tk.Checkbutton(frame2, text="tk CheckButton2", variable=Var2, onvalue=1, offvalue=0)
        Cbutton2.pack()

        Cbutton3 = ttk.Checkbutton(frame2, text="ttk CheckButton1", variable=Var1, onvalue=1, offvalue=0)
        Cbutton3.pack()
        Cbutton3 = ttk.Checkbutton(frame2, text="ttk CheckButton2", variable=Var2, onvalue=1, offvalue=0)
        Cbutton3.pack()

        Lbox1 = tk.Listbox(frame2, selectmode="multiple")
        Lbox1.insert(1, "This is a tk ListBox")
        Lbox1.insert(2, "Github")
        Lbox1.insert(3, "Python")
        Lbox1.insert(3, "StackOverflow")
        Lbox1.pack(side="left")

        Var3 = tk.IntVar()
        R1 = tk.Radiobutton(frame2, text="tk Radiobutton1", variable=Var3, value=1)
        R1.pack()
        R2 = tk.Radiobutton(frame2, text="tk Radiobutton2", variable=Var3, value=2)
        R2.pack()
        R3 = tk.Radiobutton(frame2, text="tk Radiobutton3", variable=Var3, value=3)
        R3.pack()

        R4 = tk.Radiobutton(frame2, text="ttk Radiobutton1", variable=Var3, value=1)
        R4.pack()
        R5 = tk.Radiobutton(frame2, text="ttk Radiobutton2", variable=Var3, value=2)
        R5.pack()
        R6 = tk.Radiobutton(frame2, text="ttk Radiobutton3", variable=Var3, value=3)
        R6.pack()

        # This is a treeview.
        tv1 = ttk.Treeview(frame1)
        column_list_account = ["Name", "Type", "Base Stat Total"]
        tv1['columns'] = column_list_account
        tv1["show"] = "headings"  # removes empty column
        for column in column_list_account:
            tv1.heading(column, text=column)
            tv1.column(column, width=50)
        tv1.place(relheight=1, relwidth=0.995)
        treescroll = tk.Scrollbar(frame1)
        treescroll.configure(command=tv1.yview)
        tv1.configure(yscrollcommand=treescroll.set)
        treescroll.pack(side="right", fill="y")

        # def Load_data():
        # for row in pokemon_info:
        # tv1.insert("", "end", values=row)

        def Refresh_data():
            # Deletes the data in the current treeview and reinserts it.
            tv1.delete(*tv1.get_children())  # *=splat operator
        # Load_data()

    # Load_data()


class PageOne(GUI):
    def __init__(self, parent, controller):
        GUI.__init__(self, parent)

        label1 = tk.Label(self.main_frame, font=("Verdana", 20), text="Page One")
        label1.pack(side="top")


class PageThree(GUI):
    def __init__(self, parent, controller):
        GUI.__init__(self, parent)

        label1 = tk.Label(self.main_frame, font=("Verdana", 20), text="Page Three")
        label1.pack(side="top")


class PageFour(GUI):
    def __init__(self, parent, controller):
        GUI.__init__(self, parent)

        label1 = tk.Label(self.main_frame, font=("Verdana", 20), text="Page Four")
        label1.pack(side="top")


class PageTwo(GUI):
    def __init__(self, parent, controller):
        GUI.__init__(self, parent)

        label1 = tk.Label(self.main_frame, font=("Verdana", 20), text="Page Two")
        label1.pack(side="top")


class OpenNewWindow(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self)
        main_frame.pack_propagate(0)
        main_frame.pack(fill="both", expand="true")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        self.title("Here is the Title of the Window")
        self.geometry("500x500")
        self.resizable(True, True)

        frame1 = ttk.LabelFrame(main_frame, text="This is a ttk LabelFrame")
        frame1.pack(expand=True, fill="both")

        label1 = tk.Label(frame1, font=("Verdana", 20), text="OpenNewWindow Page")
        label1.pack(side="top")


data = 0
q1 = 0
regex = '^[a-z0-9]+[._]?[a-z0-9]+[@]w+[.]w{2,3}$'
top = Main()
top.title(" SMART MIRROR")
root = dashboard()
root.withdraw()
root.title("ui smart mirror")
root.mainloop()