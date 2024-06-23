from tkinter import *
from tkinter import ttk
from PIL import Image
from PIL import ImageTk
import torch
from tkinter import filedialog as tkFileDialog
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics

inf = [
    [[0],['7-Eleven'],['Podium level, Shaw Amenities Building (Core VA)'],['You can buy drinks, snacks and other items here']],
                [[1], ['Anita Chan Lai Ling Building (Core P)'], ['NA'], ['NA']],
                [[2], ['Block X Sports Centre (Block X)'], ['NA'], ['NA']],
                [[3], ['CMA Building (Core C:Wing CD)'], ['NA'], ['NA']],
                [[4], ['Chan Sui Kau and Chan Lam Moon Chun Square'], ['NA'], ['NA']],
                [[5], ['Chan Sui Wai Building (CoreE:Wing EF)'], ['NA'], ['NA']],
                [[6], ['Chan Tai Ho Building (Core F:Wing FJ)'], ['NA'], ['NA']],
                [[7],['Cheung Che Man and Kwok Yuen Ho Bamboo Court(H-cafe)'],['NA'],['NA']],
                [[8], ['Choi Kai Yau Building (Core R:Wing QR)'], ['NA'], ['NA']],
                [[9], ['Chow Yei Ching Building (Core Q:Wing QT)'], ['NA'], ['NA']],
                [[10], ['Chung Sze Yuen Building (Core A:Wing AG)'], ['NA'], ['NA']],
                [[11], ['Communal Building (Core S)'], ['NA'], ['NA']],
                [[12], ['GH Podium Annexe (Core G:Wing GH)'], ['NA'], ['NA']],
                [[13], ['Global Student Hub'], ['NA'], ['NA']],
                [[14], ['Ho lu Kwong Building:Industrial Centre (Block W) '], ['NA'], ['NA']],
                [[15], ['Jockey Club Auditorium'], ['NA'], ['NA']],
                [[16], ['Jockey Club Innovation Tower (Block V)'], ['NA'], ['NA']],
                [[17], ['Kinmay W. Tang Building (Core F:Wing FG)'], ['NA'], ['NA']],
                [[18], ['Lawn (Lawn Cafe)'], ['NA'], ['NA']],
                [[19], ['Lee Shau Kee Building (Block Y)'], ['NA'], ['NA']],
                [[20], ['Li Ka Shing Tower (Block M)'], ['NA'], ['The place where AF, LMS, MM and FB in']],
                [[21], ['Library Cafe'], ['NA'], ['NA']],
                [[22], ['Lui Che Woo Building (Core D:Wing DE)'], ['NA'], ['NA']],
                [[23], ['Mong Man Wai Building (Wing PQ)'], ['NA'], ['It is now the Department of Computing (COMP)']],
                [[24],['Ng Wing Hong Building (Core S:Wing ST)'],['蒙民伟楼'],['NA']],
                [[25], ['Open Gate'], ['Redesigned as the main campus gate to celebrate the 85th anniversary of the university'],
                ['The main campus entrance is a clever blend of classic colonnade design and iconic red brick architecture']],
                [[26],['Pao Yue-Kong Library (Block L)'],['NA'],['NA']],
                [[27], ['Realink Building (Core U)'], ['NA'], ['NA']],
                [[28], ['SUBWAY'], ['NA'], ['NA']],
                [[29], ['Shaw Amenities Building (Block VA)'], ['NA'], ['NA']],
                [[30], ['Shirley Chan Building (Core R)'], ['NA'], ['NA']],
                [[31], ['Stanley Ho Building (Core J:Wing HJ)'], ['NA'], ['NA']],
                [[32],['Stephen Cheong Kam Chen Memorial Plaza:Flag raising platform'],['NA'],['NA']],
                [[33], ['Tang Ping Yuan Building (Core F:Wing CF)'], ['NA'], ['NA']],
                [[34], ['Yip Kit Chuen Building (Core T:Wing TU)'], ['NA'], ['NA']],
                ]

win = Tk()
win.title("Polyu Building Detection")
win.geometry('500x800')
SIZE = 256
train_images,train_labels=np.load('x_train.npy'),np.load('y_train.npy')
test_images,test_labels = np.load('x_test.npy'), np.load('y_test.npy')
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0

#Now, let us use features from convolutional network for RF
feature_extractor=VGG_model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_training = features #This is our X input to RF


#RANDOM FOREST
# model_rf = RandomForestClassifier(random_state = 42, criterion= 'gini',n_estimators = 50)
# model_rf.fit(X_for_training, y_train)

#Send test data through same feature extractor process
X_test_feature = VGG_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

model_rf = torch.load('model.rf.pt')
#Now predict using the trained xgb model. 
prediction_rf = model_rf.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction_rf = le.inverse_transform(prediction_rf)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_rf))
print('Train score' , model_rf.score(X_for_training, y_train))

# prediction_rf = model_rf.predict(X_test_features)
# #Inverse le transform to get original label back. 
# prediction_rf = le.inverse_transform(prediction_rf)


# print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_rf))
# print('Train score' , model_rf.score(X_for_training, y_train))

predicted_label = StringVar()
predicted_label.set("")

label_2 = StringVar()
label_2.set("")

label_3 = StringVar()
label_3.set("")


def select_image():   
    # grab a reference to the image panels
    global panelA, panelB, predicted_label,label_2,label_3
    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename()
    # ensure a file path was selected
    if len(path) > 0:
        img = cv2.imread(path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img / 255.0
        input_img = np.expand_dims(img, axis=0)
        input_img_feature=VGG_model.predict(input_img)
        input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
        prediction = model_rf.predict(input_img_features)[0]
        predicted_label.set(inf[prediction][1][0])
        label_2.set(inf[prediction][2][0])
        label_3.set(inf[prediction][3][0])


        prediction = le.inverse_transform([prediction])  #Reverse the label encoder to original name
        print("The prediction for this image is: ", prediction[0])

        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        image = cv2.resize(image, (500, 400))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 500, 100)
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert the images to PIL format...
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)
            # if the panels are None, initialize them
        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            # # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)
        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged
        # initialize the window toolkit along with the two image panels

    

panelA = None
panelB = None
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI

btn = Button(win, text="Select an image", height = 2, width = 10, command=select_image)
btn.pack(side="bottom", expand="yes", padx="10", pady="10")

label1 = Label(win, text="Introduction", font=("Arial", 16, "bold italic"),anchor="w")
label1.pack(fill=X, padx=10, pady=10)

# 添加分割线
sep = ttk.Separator(win, orient=HORIZONTAL)
sep.pack(fill=X, padx=10, pady=10)

predicted_label_label = Label(win, textvariable=predicted_label,anchor="w")
predicted_label_label.pack(fill=X, padx=10, pady=10)
# predicted_label_label.place(x=530, y=700)
label_2_2 = Label(win, textvariable=label_2,anchor="w")
label_2_2.pack(fill=X, padx=10, pady=10)

label_3_3 = Label(win, textvariable=label_3,anchor="w")
label_3_3.pack(fill=X, padx=10, pady=10)


win.mainloop()