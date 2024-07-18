import pandas as pd 
import numpy as np 
import sklearn 
import joblib 
import streamlit as st


model=joblib.load('iris_linear.pkl')
#features=joblib.load() 
# predictiing values 
def predict(new_data): 
    p=model.predict(new_data)
    return p 

# getting user values 
sepal_length=st.number_input("enter speal length")
sepal_width=st.number_input("enter sepal width")
petal_length=st.number_input("enter petal length")
petal_width=st.number_input("enter petal width")
# converting input data into Data Frame 


# calling the function predict 
# Predict button
if st.button("Predict"):
    input_data={
        'sepal_length': [sepal_length], 
        'sepal_width': [sepal_width], 
        'petal_length': [petal_length],
        'petal_width': [petal_width]
        }
input_data=pd.DataFrame(input_data)



    
pre=predict(input_data)
    # Display prediction result
st.write(f"The predicted flower species is: {pre[0]}")

