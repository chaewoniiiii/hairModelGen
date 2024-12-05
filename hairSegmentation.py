import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os

# Global variables
img = None
img_copy = None
mask = None
bgd_model = None
fgd_model = None
drawing = False
mode = 'foreground' 
rect_over = False
rect = (0, 0, 1, 1) 
rect_drawing = False
rect_over = False

def open_image():
    global img, img_copy, mask, bgd_model, fgd_model, tk_img, canvas, rect, rect_over, file_name
    
    # Open the file dialog to select an image
    file_path = filedialog.askopenfilename()
    file_name = os.path.basename(file_path)
    print(file_name)
    if not file_path:
        return

    img = cv2.imread(file_path)
    img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    img_copy = img.copy()

    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (0, 0, 1, 1)
    rect_over = False 

    # Resize image to fit the canvas
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    img_resized = resize_image_to_canvas(img, canvas_width, canvas_height)

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tk_img = ImageTk.PhotoImage(img_pil)

    canvas.delete('all')  
    canvas.create_image(0, 0, anchor=NW, image=tk_img)
    canvas.image = tk_img  

def resize_image_to_canvas(img, canvas_width, canvas_height):
    """이미지를 캔버스 크기에 맞게 비율을 유지하면서 리사이즈하는 함수"""
    height, width = img.shape[:2]
    aspect_ratio = width / height

    if width > height:
        new_width = canvas_width
        new_height = int(canvas_width / aspect_ratio)
    else:
        new_height = canvas_height
        new_width = int(canvas_height * aspect_ratio)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_img

def draw(event):
    global drawing, mask, img, img_copy, mode, rect_drawing, rect

    if drawing:
        x, y = int(canvas.canvasx(event.x)), int(canvas.canvasy(event.y))

        if mode == 'foreground':
            canvas.create_oval(x-10, y-10, x+10, y+10, fill="white", outline="white")
            cv2.circle(img_copy, (int(x), int(y)), 10, (255, 255, 255), -1)  
            cv2.circle(mask, (int(x), int(y)), 10, cv2.GC_FGD, -1) 
        elif mode == 'background':
            canvas.create_oval(x-10, y-10, x+10, y+10, fill="black", outline="black")
            cv2.circle(img_copy, (int(x), int(y)), 10, (0, 0, 0), -1) 
            cv2.circle(mask, (int(x), int(y)), 10, cv2.GC_BGD, -1) 

    if rect_drawing:
        rect_x, rect_y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        canvas.delete("rect")  
        canvas.create_rectangle(rect[0], rect[1], rect_x, rect_y, outline='blue', tags="rect")


def start_draw(event):
    global drawing, rect_drawing, rect

    if rect_drawing:  
        rect = (int(canvas.canvasx(event.x)), int(canvas.canvasy(event.y)), 1, 1)
    else:
        drawing = True


def stop_draw(event):
    global drawing, rect_drawing, rect

    if rect_drawing:  
        x, y = int(canvas.canvasx(event.x)), int(canvas.canvasy(event.y))
        rect = (rect[0], rect[1], x - rect[0], y - rect[1])
        rect_drawing = False

    drawing = False


def set_foreground():
    global mode
    mode = 'foreground'
    print("Switched to foreground drawing mode.")

def set_background():
    global mode
    mode = 'background'
    print("Switched to background drawing mode.")

def draw_rectangle():
    global rect_drawing
    rect_drawing = True
    print("Draw a rectangle around the object.")


def run_grabcut():
    global img, img_copy, mask, bgd_model, fgd_model, rect, segmented_img, rect_over

    if not rect_over:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        rect_over = True  
    else:

        cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

    segmented_img = img * mask2[:, :, np.newaxis]

    img_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tk_segmented_img = ImageTk.PhotoImage(img_pil)

    canvas.delete('all')  
    canvas.create_image(0, 0, anchor=NW, image=tk_segmented_img)
    canvas.image = tk_segmented_img 
    segmented_img = cv2.resize(segmented_img, dsize= (0, 0),fx = 2, fy = 2)


def save_image():
    global file_name
    output_path = './images/results/'
    cv2.imwrite(output_path + file_name, segmented_img)
    print(f"Segmented image saved to {output_path}")


root = Tk()
root.title("Interactive Image Segmentation with Tkinter")

btn_open = Button(root, text="Open Image", command=open_image)
btn_open.pack(side='top', padx=5, pady=5)

frame_bottom = Frame(root)
frame_bottom.pack(side='top')

canvas = Canvas(root, width=800, height=600)
canvas.pack(side='top')

btn_fg = Button(frame_bottom, text="Foreground", command=set_foreground)
btn_fg.grid(row=1, column=1, padx=5, pady=5)

btn_bg = Button(frame_bottom, text="Background", command=set_background)
btn_bg.grid(row=1, column=2, padx=5, pady=5)

btn_rect = Button(frame_bottom, text="Draw Rectangle", command=draw_rectangle)
btn_rect.grid(row=1, column=3, padx=5, pady=5)

btn_grabcut = Button(frame_bottom, text="Run GrabCut", command=run_grabcut)
btn_grabcut.grid(row=1, column=4, padx=5, pady=5)

btn_save = Button(frame_bottom, text="Save Segmented Image", command=save_image)
btn_save.grid(row=1, column=5, padx=5, pady=5)

canvas.bind("<ButtonPress-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)



def main():
    # Run Tkinter loops
    root.mainloop() 

main()
