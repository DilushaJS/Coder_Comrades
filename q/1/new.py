from logging import RootLogger
import tkinter as tk
from tkinter import Label, PhotoImage, messagebox
from tkinter import ttk


"""
Useful Links:
https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter Most useful in my opinion
https://www.tutorialspoint.com/python/python_gui_programming.htm
https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/index.html
https://www.youtube.com/watch?v=HjNHATw6XgY&list=PLQVvvaa0QuDclKx-QpC9wntnURXVJqLyk
"""

# You can also use a pandas dataframe for pokemon_info.
# you can convert the dataframe using df.to_numpy.tolist()
pokemon_info = [['Bulbasaur', 'Grass', '318'], ['Ivysaur', 'Grass', '405'], ['Venusaur', 'Grass', '525'], ['Charmander', 'Fire', '309'], ['Charmeleon', 'Fire', '405'], ['Charizard', 'Fire', '534'], ['Squirtle', 'Water', '314'], ['Wartortle', 'Water', '405'], ['Blastoise', 'Water', '530'], ['Caterpie', 'Bug', '195'], ['Metapod', 'Bug', '205'], ['Butterfree', 'Bug', '395'], ['Weedle', 'Bug', '195'], ['Kakuna', 'Bug', '205'], ['Beedrill', 'Bug', '395'], ['Pidgey', 'Normal', '251'], ['Pidgeotto', 'Normal', '349'], ['Pidgeot', 'Normal', '479'], ['Rattata', 'Normal', '253'], ['Raticate', 'Normal', '413'], ['Spearow', 'Normal', '262'], ['Fearow', 'Normal', '442'], ['Ekans', 'Poison', '288'], ['Arbok', 'Poison', '448'], ['Pikachu', 'Electric', '320'], ['Raichu', 'Electric', '485'], ['Sandshrew', 'Ground', '300'], ['Sandslash', 'Ground', '450'], ['Nidoran?', 'Poison', '275'], ['Nidorina', 'Poison', '365'], ['Nidoqueen', 'Poison', '505'], ['Nidoran?', 'Poison', '273'], ['Nidorino', 'Poison', '365'], ['Nidoking', 'Poison', '505'], ['Clefairy', 'Fairy', '323'], ['Clefable', 'Fairy', '483'], ['Vulpix', 'Fire', '299'], ['Ninetales', 'Fire', '505'], ['Jigglypuff', 'Normal', '270'], ['Wigglytuff', 'Normal', '435'], ['Zubat', 'Poison', '245'], ['Golbat', 'Poison', '455'], ['Oddish', 'Grass', '320'], ['Gloom', 'Grass', '395'], ['Vileplume', 'Grass', '490'], ['Paras', 'Bug', '285'], ['Parasect', 'Bug', '405'], ['Venonat', 'Bug', '305'], ['Venomoth', 'Bug', '450'], ['Diglett', 'Ground', '265'], ['Dugtrio', 'Ground', '425'], ['Meowth', 'Normal', '290'], ['Persian', 'Normal', '440'], ['Psyduck', 'Water', '320'], ['Golduck', 'Water', '500'], ['Mankey', 'Fighting', '305'], ['Primeape', 'Fighting', '455'], ['Growlithe', 'Fire', '350'], ['Arcanine', 'Fire', '555'], ['Poliwag', 'Water', '300'], ['Poliwhirl', 'Water', '385'], ['Poliwrath', 'Water', '510'], ['Abra', 'Psychic', '310'], ['Kadabra', 'Psychic', '400'], ['Alakazam', 'Psychic', '500'], ['Machop', 'Fighting', '305'], ['Machoke', 'Fighting', '405'], ['Machamp', 'Fighting', '505'], ['Bellsprout', 'Grass', '300'], ['Weepinbell', 'Grass', '390'], ['Victreebel', 'Grass', '490'], ['Tentacool', 'Water', '335'], ['Tentacruel', 'Water', '515'], ['Geodude', 'Rock', '300'], ['Graveler', 'Rock', '390'], ['Golem', 'Rock', '495'], ['Ponyta', 'Fire', '410'], ['Rapidash', 'Fire', '500'], ['Slowpoke', 'Water', '315'], ['Slowbro', 'Water', '490'], ['Magnemite', 'Electric', '325'], ['Magneton', 'Electric', '465'], ["Farfetch'd", 'Normal', '377'], ['Doduo', 'Normal', '310'], ['Dodrio', 'Normal', '470'], ['Seel', 'Water', '325'], ['Dewgong', 'Water', '475'], ['Grimer', 'Poison', '325'], ['Muk', 'Poison', '500'], ['Shellder', 'Water', '305'], ['Cloyster', 'Water', '525'], ['Gastly', 'Ghost', '310'], ['Haunter', 'Ghost', '405'], ['Gengar', 'Ghost', '500'], ['Onix', 'Rock', '385'], ['Drowzee', 'Psychic', '328'], ['Hypno', 'Psychic', '483'], ['Krabby', 'Water', '325'], ['Kingler', 'Water', '475'], ['Voltorb', 'Electric', '330'], ['Electrode', 'Electric', '490'], ['Exeggcute', 'Grass', '325'], ['Exeggutor', 'Grass', '530'], ['Cubone', 'Ground', '320'], ['Marowak', 'Ground', '425'], ['Hitmonlee', 'Fighting', '455'], ['Hitmonchan', 'Fighting', '455'], ['Lickitung', 'Normal', '385'], ['Koffing', 'Poison', '340'], ['Weezing', 'Poison', '490'], ['Rhyhorn', 'Ground', '345'], ['Rhydon', 'Ground', '485'], ['Chansey', 'Normal', '450'], ['Tangela', 'Grass', '435'], ['Kangaskhan', 'Normal', '490'], ['Horsea', 'Water', '295'], ['Seadra', 'Water', '440'], ['Goldeen', 'Water', '320'], ['Seaking', 'Water', '450'], ['Staryu', 'Water', '340'], ['Starmie', 'Water', '520'], ['Scyther', 'Bug', '500'], ['Jynx', 'Ice', '455'], ['Electabuzz', 'Electric', '490'], ['Magmar', 'Fire', '495'], ['Pinsir', 'Bug', '500'], ['Tauros', 'Normal', '490'], ['Magikarp', 'Water', '200'], ['Gyarados', 'Water', '540'], ['Lapras', 'Water', '535'], ['Ditto', 'Normal', '288'], ['Eevee', 'Normal', '325'], ['Vaporeon', 'Water', '525'], ['Jolteon', 'Electric', '525'], ['Flareon', 'Fire', '525'], ['Porygon', 'Normal', '395'], ['Omanyte', 'Rock', '355'], ['Omastar', 'Rock', '495'], ['Kabuto', 'Rock', '355'], ['Kabutops', 'Rock', '495'], ['Aerodactyl', 'Rock', '515'], ['Snorlax', 'Normal', '540'], ['Articuno', 'Ice', '580'], ['Zapdos', 'Electric', '580'], ['Moltres', 'Fire', '580'], ['Dratini', 'Dragon', '300'], ['Dragonair', 'Dragon', '420'], ['Dragonite', 'Dragon', '600'], ['Mewtwo', 'Psychic', '680'], ['Mew', 'Psychic', '600']]


frame_styles = {"relief": "groove",
                "bd": 3, "bg": "#ffffff",
                "fg": "#073bb3", "font": ("Arial", 30, "bold")}


class LoginPage(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffff", height=1920, width=720,bd=0)  # this is the background
        main_frame.pack(fill="both", expand="true")

        logo = PhotoImage(file="logo.jpg") # Replace "logo.png" with your logo file
        logo_label = tk.Label(main_frame, image=logo)
        logo_label.pack()
        logo_label.place(rely=0.025, relx=0.35)

        self.geometry("720x1920")  # Sets window size to 626w x 431h pixels
        self.resizable(0, 0)  # This prevents any resizing of the screen
        title_styles = {"font": ("Trebuchet MS Bold", 30), "background": "white"}

        text_styles = {"font": ("Verdana", 10),
                       "background": "white",
                       "foreground": "#111111"}
        frame_login = tk.Frame(main_frame, bg="#ffffff", relief="groove", bd=1)  # this is the frame that holds all the login details and buttons
        frame_login.place(rely=0.15, relx=0.200, height=500, width=450)
          
        label_title = tk.Label(frame_login, title_styles, text="Login ")
        label_title.place(rely=0.1, relx=0.4)

        label_user = tk.Label(frame_login, text_styles, text="Username:")
        label_user.place(rely=0.25
                         , relx=0.05)

        label_pw = tk.Label(frame_login, text_styles, text="Password:")
        label_pw.place(rely=0.45, relx=0.05)

        entry_user = ttk.Entry(frame_login, width=40, cursor="xterm")
        entry_user.place(rely=0.3, relx=0.4)

        entry_pw = ttk.Entry(frame_login, width=40, cursor="xterm", show="*")
        entry_pw.place(rely=0.5, relx=0.4)

        button = ttk.Button(frame_login, text="Login", command=lambda: getlogin())
        button.place(rely=0.65, relx=0.390,height=70,width=150)
    

        signup_btn = ttk.Button(frame_login, text="Register", command=lambda: get_signup())
        signup_btn.place(rely=0.8, relx=0.460)

        def get_signup():
            SignupPage()

        def getlogin():
            username = entry_user.get()
            password = entry_pw.get()
            # if your want to run the script as it is set validation = True
            validation = validate(username, password)
            if validation:
                tk.messagebox.showinfo("Login Successful",
                                       "Welcome {}".format(username))
            else:
                tk.messagebox.showerror("Information", "The Username or Password you have entered are incorrect ")

        def validate(username, password):
            # Checks the text file for a username/password combination.
            try:
                with open("credentials.txt", "r") as credentials:
                    for line in credentials:
                        line = line.split(",")
                        if line[1] == username and line[3] == password:
                            return True
                    return False
            except FileNotFoundError:
                print("You need to Register first or amend Line 71 to     if True:")
                return False


class SignupPage(tk.Tk):

    def __init__(self,*args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        main_frame = tk.Frame(self, bg="#ffffEE", height=1920, width=720)  # this is the background
        main_frame.pack(fill="both", expand="true")


        self.geometry("720x1920")  # Sets window size to 626w x 431h pixels
        self.resizable(0, 0)  # This prevents any resizing of the screen

        text_styles = {"font": ("Verdana", 10),
                       "background": "white",
                       "foreground": "#000000"}
        label_fn = tk.Label(main_frame, text_styles, text="Welcome to the SMART MIRROR:")
        label_fn.place(rely=0.001, relx=0.15)

        frame_sign = tk.Frame(main_frame, bg="#ffffff", relief="groove", bd=1)  # this is the frame that holds all the login details and buttons
        frame_sign.place(rely=0.015, relx=0.155, height=700, width=450)

        label_fn = tk.Label(frame_sign, text_styles, text="First Nmae:")
        label_fn.place(rely=0.06, relx=0.2)

        label_ln = tk.Label(frame_sign, text_styles, text="Last Name:")
        label_ln.place(rely=0.06, relx=0.5)

        entry_fn = ttk.Entry(frame_sign,width=20, cursor="xterm")
        entry_fn.place(rely=0.12, relx=0.2)

        entry_ln = ttk.Entry(frame_sign,width=20, cursor="xterm")
        entry_ln.place(rely=0.12, relx=0.5)

        label_email = tk.Label(frame_sign, text_styles, text="E-mail:")
        label_email.place(rely=0.2, relx=0.2)

        entry_email = ttk.Entry(frame_sign,width=20, cursor="xterm")
        entry_email.place(rely=0.2, relx=0.5)

        label_sq = tk.Label(frame_sign, text_styles, text="Security question:")
        label_sq.place(rely=0.55, relx=0.2)

        label_sq_a = tk.Label(frame_sign, text_styles, text="  answer:")
        label_sq_a.place(rely=0.55, relx=0.5)

        entry_sq = ttk.Entry(frame_sign,width=20, cursor="xterm")
        entry_sq.place(rely=0.6, relx=0.2)

        entry_sq_a = ttk.Entry(frame_sign,width=20, cursor="xterm")
        entry_sq_a.place(rely=0.6, relx=0.5)

        label_user = tk.Label(frame_sign, text_styles, text="Username:")
        label_user.place(rely=0.3, relx=0.2)

        label_pw = tk.Label(frame_sign, text_styles, text="Password:")
        label_pw.place(rely=0.4, relx=0.2)

        entry_user = ttk.Entry(frame_sign,width=20, cursor="xterm")
        entry_user.place(rely=0.3, relx=0.5)

        entry_pw = ttk.Entry(frame_sign,width=20, cursor="xterm", show="*")
        entry_pw.place(rely=0.4, relx=0.5)

        button = ttk.Button(frame_sign, text="Create Account",width=15,command=lambda: signup())
        button.place(rely=0.8, relx=0.4)

        label_user = tk.Label(frame_sign, text_styles, text="Already have a account")
        label_user.place(rely=0.95, relx=0.1)

        button_login = ttk.Button(frame_sign, text="Create Account",width=15,command=lambda: get_login())
        button_login.place(rely=0.95, relx=0.5)

        def signup():
            # Creates a text file with the Username and password
            user = entry_user.get()
            email = entry_user.get()
            fn = entry_user.get()
            sq = entry_user.get()
            sq_a = entry_user.get()
            ln = entry_user.get()
            pw = entry_pw.get()
            validation = validate_user(user)
            if not validation:
                tk.messagebox.showerror("Information", "That Username already exists")
            else:
                if len(pw) > 3:
                    credentials = open("credentials.txt", "a+")
                    credentials.write(f"First_name,{fn},Last_name,{ln},email,{email},secrity,{sq},answer,{sq_a},Username,{user},Password,{pw},\n")
                    credentials.close()
                    tk.messagebox.showinfo("Information", "Your account details have been stored.")
                    SignupPage.destroy(self)

                else:
                    tk.messagebox.showerror("Information", "Your password needs to be longer than 3 values.")

        def validate_user(username):
            # Checks the text file for a username/password combination.
            try:
                with open("credentials.txt", "r") as credentials:
                    for line in credentials:
                        line = line.split(",")
                        if line[1] == username:
                            return False
                return True
            except FileNotFoundError:
                return True
        def get_login():
            LoginPage()
root= LoginPage()
root.title(" SMART MIRROR")
root.withdraw()
root.mainloop()
