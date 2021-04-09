import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def classify(num):
    if num == 1:
        return 'Defaulter'
    else:
        return 'Not-Defaulter'
html_temp = """
<div style="background-color:teal ;padding:10px">
<h2 style="color:white;text-align:center;"> Loan Defaulter Prediction App </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.write("""
This app predicts the **whether the person is defaulter or not** !
""")


st.header('User input features')


def user_input_features():
    CreditScore = st.slider('Credit Score',350,850)
    Geography= st.selectbox('Country',('France', 'Spain' ,'Germany'))
    Gender = st.selectbox('Gender', ('Female', 'Male'))
    Age = st.slider('Age', 18,92)
    Tenure = st.slider('Tenure', 0,1)
    Balance = st.slider('Balance', 0,260000)
    NumberofProducts = st.slider('Number of Products', 1,4)
    HasCrcard = st.slider('Has Credit Card', 0,1)
    IsActiveMember = st.selectbox('Is Active Member',(0,1))
    EstimatedSalary = st.slider('Estimated Salary',10,200000)

    data = {'CreditScore':CreditScore,
    'Geography':Geography,
    'Gender':Gender,
    'Age':Age,
    'Tenure':Tenure,
    'Balance':Balance,
    'NumberofProducts':NumberofProducts,
    'HasCrcard':HasCrcard,
    'IsActiveMember':IsActiveMember,
    'EstimatedSalary':EstimatedSalary}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()



objList = input_df.select_dtypes(include = "object").columns

for feat in objList:
    input_df[feat] = le.fit_transform(input_df[feat].astype(str))

df = input_df[:1]

if st.button('Predict'):
        load_clf = pickle.load(open('predictionrfc.pkl','rb'))
        st.subheader('User Input features')
        st.write(df)
        prediction = load_clf.predict(df)
        st.success(classify(load_clf.predict(df)))
