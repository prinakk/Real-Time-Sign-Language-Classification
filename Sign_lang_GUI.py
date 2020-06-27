from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import keras as keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import load_model
import numpy as np
import cv2
import os
os.chdir('../Sign_lang_complete/')
MyWindow = Tk() # Create a window
MyWindow.title("Sign Language Translator") # Change the Title of the GUI
MyWindow.geometry('2000x2000')
MyLabel = Label(text = "Select from the below sign languages", font=("Arial Bold", 20))
ClassficationResultLabel = Label(text = "Translation result : ",fg='red', font=("Arial Bold", 20))

# Create Event Methods attached to the button etc.
def Button1_Clicked():
    Classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']
    messagebox.showinfo("Info", "You have selected ASL translation")
    # Use the File Dialog component to Open the Dialog box to select files
    file = filedialog.askopenfilename(filetypes = (("Nummpy files","*.npy"),("Images files","*.png"),("Images files","*.jpg"),("Video Files","*.mp4"),("all files","*.*")))
    messagebox.showinfo("File Selected", file)
    messagebox.showinfo("Info", "Translate image")
    a=np.load(file)
    a=np.reshape(a,[1,64,64,3])
    model = load_model('ASLmodel.hdf5')
    y_pred = np.argmax(model.predict(a),-1)
    results= Classes[int(y_pred)]
    ClassficationResultLabel.configure(text ="Translation Result:" +results,fg='blue',font=("Arial Bold", 30))  # Update the Label text on the Window
    
    

def Button2_Clicked():
    Classes=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
    messagebox.showinfo("Info", "You have selected ISL translation")
    # Use the File Dialog component to Open the Dialog box to select files
    file = filedialog.askopenfilename(filetypes = (("Nummpy files","*.npy"),("Images files","*.png"),("Images files","*.jpg"),("Video Files","*.mp4"),("all files","*.*")))
    messagebox.showinfo("File Selected", file)
    messagebox.showinfo("Info", "Translate image")
    a=np.load(file)
    a=np.reshape(a,[1,64,64,3])
    model = load_model('ISLmodel.hdf5')
    y_pred = np.argmax(model.predict(a),-1)
    results= Classes[int(y_pred)]
    ClassficationResultLabel.configure(text ="Translation Result:" +results,fg='blue',font=("Arial Bold", 30))

def Button3_Clicked():
    import cv2
    import numpy as np
    from keras.models import load_model
    from skimage.transform import resize, pyramid_reduce
    import PIL
    from PIL import Image

    model = load_model('ISLmodel.hdf5')


    def prediction(pred):
        return(chr(pred+ 65))


    def keras_predict(model, image):
        data = np.asarray( image, dtype="int32" )
    
        pred_probab = model.predict(data)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        return max(pred_probab), pred_class

    def keras_process_image(img):
    
        image_x = 64
        image_y = 64
        img = cv2.resize(img, (1,64,64), interpolation = cv2.INTER_AREA)
  
        return img
 

    def crop_image(image, x, y, width, height):
        return image[y:y + height, x:x + width]

    def main():
        l = []
    
        while True:
        
            cam_capture = cv2.VideoCapture(0)
            _, image_frame = cam_capture.read()  
    # Select ROI
            im2 = crop_image(image_frame, 300,300,300,300)
            image_grayscale = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (15,15), 0)
            im3 = cv2.resize(image_grayscale, (64,64), interpolation = cv2.INTER_AREA)


    
            im4 = np.resize(im3, (64, 64, 3))
            im5 = np.expand_dims(im4, axis=0)
    

            pred_probab, pred_class = keras_predict(model, im5)
    
            curr = prediction(pred_class)
        
            cv2.putText(image_frame, curr, (700, 300), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
            
            
    
 
    # Display cropped image
            cv2.rectangle(image_frame, (300, 300), (600, 600), (255, 255, 00), 3)
            cv2.imshow("frame",image_frame)
        
        
    #cv2.imshow("Image4",resized_img)
            cv2.imshow("Image3",image_grayscale)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


    if __name__ == '__main__':
        main()

    cam_capture.release()
    cv2.destroyAllWindows()


canvas2 = Canvas(MyWindow, width = 300, height = 300)  
img = PhotoImage(file="ISL.gif")  
canvas2.create_image(20, 20, anchor=NW, image=img)
canvas2.grid(column=3, row=2)
canvas1= Canvas(MyWindow, width = 300, height = 300)  
img1 = PhotoImage(file="ASL.gif")  
canvas1.create_image(20, 20, anchor=NW, image=img1)
canvas1.grid(column=2, row=2)
Button1 = Button(text="American Sign Language", command=Button1_Clicked)
Button1.grid(column=2, row=4) # Adding the Open Button
Button2 = Button(text="Indian Sign Language", command=Button2_Clicked)
Button2.grid(column=3, row=4) # Adding the Open Button
ClassficationResultLabel.grid(column=1, row=8) # Adding the label to display classfication result
MyLabel.grid(column=2, row=0) # Adding label for information on the Text Entry box
Button3= Button(text="Real-Time Translation - Beta Version", command=Button3_Clicked)
Button3.grid(column=1, row=9)
MyWindow.mainloop()
