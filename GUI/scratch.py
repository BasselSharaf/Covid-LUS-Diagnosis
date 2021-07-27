from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfile
from tkinter.messagebox import showinfo

from matplotlib import pyplot as plt
from tkvideo import tkvideo
import tkinter.font as font
from PIL import ImageTk

from utilities import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ws = Tk()
ws.geometry('+20+30')
ws.title('Covid-19 Diagnosis with Lung Ultra Sound By @Bassel_Sharaf')
style = Style()
myFont1 = font.Font(family='Times New Roman', size=12, weight='bold')
myFont2 = font.Font(family='Times New Roman', size=12)
# Create a photoimage object of the image in the path
image1 = Image.open("covidoo.jpg")
image1 = image1.resize((700, 500), Image.ANTIALIAS)
test = ImageTk.PhotoImage(image1)
my_label = Label(ws, image=test)
my_label.image = test
my_label.pack()


def open_file():
    global file_path
    file_path = askopenfile(mode='r', filetypes=[
        ("all video format", ".mp4"),
        ("all video format", ".flv"),
        ("all video format", ".avi"),
    ])
    print(file_path.name)
    if file_path is not None:
        pass
    player = tkvideo(file_path.name, my_label, loop=1, size=(700, 500))
    player.play()
    upld['state'] = NORMAL
    return file_path


def popup_showinfo(prediction):
    showinfo("Diagnosis", "This patient is diagnosed with " + str(prediction))


def buttonPressed():
    top = Toplevel()
    top.geometry("+725+1")
    prediction, images = getprediction(file_path.name)
    fig, axs = plt.subplots(5, 1)
    fig.set_size_inches(2, 9)

    for i in range(0, 5):
        axs[i].imshow(images[0])
        axs[i].set_title('Frame-' + str(i) + ' ' + prediction[i])
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    canvas = FigureCanvasTkAgg(fig, top)
    canvas.get_tk_widget().pack(side=RIGHT)
    canvas.draw()
    final = most_common(prediction)
    popup_showinfo(final)


adhar = Label(
    ws,
    text='Press the Choose File button to choose your Lung Ultrasound Scan ',
    font=myFont1
)
adhar.pack(side=LEFT)

style.configure("BW.TLabel", background='blue', foreground='white', font=myFont2)
adharbtn = Button(ws, text='Choose File', style="BW.TLabel",
                  command=lambda: open_file())

adharbtn.pack(side=LEFT)

upld = Button(
    ws,
    text='Get Diagnosis',
    style="BW.TLabel",
    command=buttonPressed,
    state=DISABLED
)
upld.pack(side=RIGHT)

ws.mainloop()
