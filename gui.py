


import tkinter as tk
from PIL import Image, ImageTk


def main():
    # Create the main window
    root = tk.Tk()
    root.title("Image Viewer")

    # Set window size (optional)
    root.geometry("400x400")

    # Image path (replace with your image path)
    img_path = "E:/gui/img/img.png"  # Adjust the path accordingly

    # Try loading the image
    try:
        image = Image.open(img_path)
        photo = ImageTk.PhotoImage(image)
    except FileNotFoundError:
        print(f"Error: Image file not found: {img_path}")
        return  # Exit if image is not found

    # Create a label to display the image
    label = tk.Label(root, image=photo)
    label.pack()

    # Run the main event loop
    root.mainloop()


if __name__ == "__main__":
    main()
