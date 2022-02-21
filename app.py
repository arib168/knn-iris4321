import streamlit as st 
import numpy as np 
from sklearn.datasets import load_iris
from PIL import Image  #python imaging library

st.title("Iris Flower classification!")
image = Image.open('Picture1.jpg')
st.image(image, caption='TYPES OF IRIS FLOWERS')

#Load the dataset 
var = load_iris() 

# split the data into input and output 
x = var.data    #input 
y = var.target  #output 

# import the KNN model 
from sklearn.neighbors import KNeighborsClassifier 
model = KNeighborsClassifier(n_neighbors = 15)

# fit the KNN model 
model.fit(x,y)

# finding max and min values to take user inputs for sliders
xmin = np.min(x,axis = 0)
xmax = np.max(x,axis = 0)

# create 4 sliders for inputs 
sepal_length = st.slider("Sepal Length",float(xmin[0]),float(xmax[0]))
sepal_width  = st.slider("Sepal Width",float(xmin[1]),float(xmax[1]))
petal_length = st.slider("Petal Length",float(xmin[2]),float(xmax[2]))
petal_width  = st.slider("Petal Width",float(xmin[3]),float(xmax[3]))

# predict the output 
y_pred = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

# print the object class 
op = ['Iris-setosa','Iris-versicolor','Iris-virginica']
st.title(op[y_pred[0]])
