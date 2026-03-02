import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib
import skimage

def main():
    # 1. Print verification lines to terminal
    print("Testing Environment Packages...")
    print(f"Tkinter version: {tk.TkVersion}")
    print(f"OpenCV (cv2) version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pillow (PIL) version: {Image.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Scikit-Image version: {skimage.__version__}")

    # 2. Create a basic test image using OpenCV/NumPy
    # Create a simple 400x400 black image and draw a white circle
    img_np = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img_np, (200, 200), 100, (255, 255, 255), 3)
    cv2.putText(img_np, "Env Setup Success!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 3. Initialize Tkinter
    root = tk.Tk()
    root.title("Environment Test Window")
    root.geometry("450x450")

    # Convert the BGR OpenCV image to RGB for Pillow, then to ImageTk
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    # Display it in Tkinter
    label = tk.Label(root, image=img_tk)
    label.pack(pady=20)

    # Notify success in UI
    def show_info():
        messagebox.showinfo("Success", "All core libraries loaded and UI rendered successfully!")

    btn = tk.Button(root, text="Click to Verify", command=show_info)
    btn.pack()

    print("\nAttempting to open the Tkinter GUI. Close the window to exit the script.")
    root.mainloop()

if __name__ == "__main__":
    main()
