import tkinter as tk
import random
import numberDetectionTools

GRID_SIZE = 28
PIXEL_SIZE = 20 

currentIndex = random.randint(0,len(numberDetectionTools.x_test)-1)
grid_data = numberDetectionTools.x_test[currentIndex]

network = numberDetectionTools.getTrainedNetwork()

def changeData():
    global grid_data
    global currentIndex
    currentIndex = random.randint(0,len(numberDetectionTools.x_test)-1)
    grid_data = numberDetectionTools.x_test[currentIndex]
    generateGrid()
    text.config(state=tk.NORMAL)
    text.delete("1.0",tk.END)
    generateText()
    text.config(state=tk.DISABLED)


def get_color(value):
    gray = int(value * 255)
    return f'#{gray:02x}{gray:02x}{gray:02x}'

def generateGrid():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x1 = col * PIXEL_SIZE
            y1 = row * PIXEL_SIZE
            x2 = x1 + PIXEL_SIZE
            y2 = y1 + PIXEL_SIZE
            color = get_color(grid_data[row][col])
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

def generateText():
    networkResult = network.forward(grid_data.reshape(1, 28*28))
    for i in range(len(networkResult)):
        if(i == numberDetectionTools.y_test[currentIndex]):
            text.insert(tk.END,str(round(networkResult[i],2))+", ","vert")
        else:
            text.insert(tk.END,str(round(networkResult[i],2))+", ","rouge")

window = tk.Tk()
window.title("Grille 28x28 - Noir et Blanc")

canvas = tk.Canvas(window, width=GRID_SIZE * PIXEL_SIZE, height=GRID_SIZE * PIXEL_SIZE)
canvas.pack()
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        x1 = col * PIXEL_SIZE
        y1 = row * PIXEL_SIZE
        x2 = x1 + PIXEL_SIZE
        y2 = y1 + PIXEL_SIZE
        color = get_color(grid_data[row][col])
        canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

text = tk.Text(window,height=5)
text.tag_configure("rouge", foreground="red")
text.tag_configure("vert", foreground="green")
text.config(font=("Courier",22))
text.pack()
generateText()
text.config(state=tk.DISABLED)

regenerateButton = tk.Button(window,text="Regenerate",command=changeData)
regenerateButton.pack()

window.mainloop()