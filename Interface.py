import threading
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import cv2
import os
import glob
import shutil

homepath=os.path.dirname(__file__)
FILE_OUTPUT = homepath+'/LipFeature Extraction/localizedFaces'

#fileHandling
try:
    if os.path.exists(FILE_OUTPUT):
        for dir in os.listdir(FILE_OUTPUT):
            shutil.rmtree(os.path.join(FILE_OUTPUT,dir))
except OSError:
    print ("Error in Refreshing the folder localizedFaces")
   

try:
    if not os.path.exists(FILE_OUTPUT+'/*'):
        print()
    else:
        frameImgs = glob.glob(os.path.join(FILE_OUTPUT+'/*/','*g'))
        for fi in frameImgs:
            os.remove(fi)
except OSError:
    print ('Error: Creating directory')



    


e = threading.Event()
p = None

def video_stream():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    main.imgtk = imgtk
    main.configure(image=imgtk)
    main.after(1, video_stream)
    

def startrecording(e):
    set_text("Recording started")
    main.destroy()
    import detector  
    set_text("Recording Finished")
   

def start_recording_proc():
    global p
    p = threading.Thread(target=startrecording, args=(e,))
    p.start()
       
def findWord():
    if len(os.listdir(FILE_OUTPUT) ) == 0:
        set_text("Record the video first")
    else:
        set_text("Analyzing the Word")
        import LipFeature
        import Textoutput.textoutput as textout
        set_text(LipFeature.User+' : '+'"'+textout.text+'"')
    
            
def restart():
    python = sys.executable
    os.execl(python, python, * sys.argv)
    

def set_text(text):
    ent.delete(0,END)
    #ent.insert(END, text)
    #ent.insert(END,"\n")
    ent.insert(0,text)
    return

def refresh():
    set_text("")


if __name__ == "__main__":
    root = tk.Tk()
    app = Frame(root)
    app.pack()
    root.title('User console')
    #root.geometry('1300x700+0+0')
    root.state('zoomed')
    main = Label(app)
    main.grid()
    


    cap= cv2.VideoCapture(1)
    
    video_stream()

    
    #GUI components
    ent = Entry(root,width=80,bg='black',fg='white')
    ent.pack(side=RIGHT,padx=1,pady=10,fill=Y)
    refreshbutton=tk.Button(root,width=20,height=2,text='REFRESH',command=refresh, bg='blue')
    refreshbutton.pack(side=RIGHT,padx=1, pady=5)
   
    startbutton=tk.Button(root,width=20,height=2,text='START',command=start_recording_proc, bg='blue')
    startbutton.pack( padx=0, pady=10)
    findbutton=tk.Button(root,width=20,height=2,text='FIND WORD',command=findWord, bg='blue')
    findbutton.pack(padx=1, pady=10)
    
    quitbutton = tk.Button(root,width=20,height=2,text="RESTART",command=restart, bg='blue')
    quitbutton.pack(padx=1, pady=10)

    root.mainloop()
