import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from PIL import Image

st.write("""
# this is Stroke prediction  WebApp
 the dataset of This Stroke prediction dataset collected from Kaggle Which  has 5110 rows
and 12 columns )
""")
image = Image.open('image1.jpg')
st.image(image, caption='Stroke Prediction')



st.sidebar.header('User Input Parameters')

def user_input_features():



    selected_gender = st.selectbox("Select your gender ", ["Male", "Female"])
    gender_mapping = { "Male": 1, "Female": 0}
    gender = gender_mapping[selected_gender]



    age=st.sidebar.slider("What is your age?", min_value=0.0, max_value=120.0, step=0.1)


    
    selected_hypertension = st.selectbox("Have you ever had hypertension   ", ["Yes", "No"])
    hypertension_mapping = { "Yes": 1, "No": 0}
    hypertension = hypertension_mapping[selected_hypertension]


    
   
    selected_heart_disease = st.selectbox("Have you ever had heart_disease?   ", ["Yes", "No"])
    heart_disease_mapping = { "Yes": 1, "No": 0}
    heart_disease = heart_disease_mapping[selected_heart_disease]






    selected_ever_married = st.selectbox("Have you ever married?   ", ["Yes", "No"])
    ever_married_mapping = { "Yes": 1, "No": 0}
    ever_married = ever_married_mapping[selected_ever_married]







    

    selected_work_type = st.selectbox("Select your Work_type  ", ["Private", "Self-employed","children","Govt_job","Never_worked"])
    work_type_mapping = { "Private": 2, "Self-employed": 3, "children": 2, "Govt_job": 0, "Never_worked": 1}
    work_type =work_type_mapping[selected_work_type]






    selected_Residence_type = st.selectbox("Select your Residence_type   ", ["Urban", "Rural"])
    Residence_type_mapping = { "Urban": 1, "Rural": 0}
    Residence_type= Residence_type_mapping[selected_Residence_type]




    avg_glucose_level=st.sidebar.slider("What is your avg_glucose_level?", min_value=0.0, max_value=400.0, step=0.001)
    bmi=st.sidebar.slider("What is your bmi?",0.0,100.0, step=0.1)
    

    selected_smoking_status = st.selectbox("Select your smoking_status  ", ["formerly smoked", "never smoked","smokes","Unknown"])
    smoking_status_mapping = { "formerly smoked": 1, "never smoked": 2, "smokes": 3, "Unknown": 0}
    smoking_status =smoking_status_mapping[selected_smoking_status]

    data = {'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
              }
    features = pd.DataFrame(data, index=[0])
    return features

df2 = user_input_features()

st.subheader('User Input parameters')
st.write(df2)
#data frame
df=pd.read_csv("healthcare-dataset-stroke-data.csv")

df['bmi']=df['bmi'].fillna(df['bmi'].mean())

#label encoding
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
df['ever_married']=l.fit_transform(df['ever_married'])
df['Residence_type']=l.fit_transform(df['Residence_type'])
df['work_type']=l.fit_transform(df['work_type'])
df['smoking_status']=l.fit_transform(df['smoking_status'])
df['gender']=l.fit_transform(df['gender'])


#select x and y 
df=df.iloc[:,1:]
x=df.iloc[:,:-1].values
y=df['stroke'].values

#scaling
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
x=s.fit_transform(x)


#SMOE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=25)
x_resampled, y_resampled= smote.fit_resample(x, y)


#Train_Test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_resampled,y_resampled,test_size=0.3)


#model training
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(max_depth=None,min_samples_leaf=1,min_samples_split=2)
rf_model=rf.fit(x_train,y_train)
prediction=rf.predict_proba(df2)*100

st.subheader('Prediction in percent is  :')
st.write('The number behind the 1 is the probabillity of Stroke in percent :',prediction)


