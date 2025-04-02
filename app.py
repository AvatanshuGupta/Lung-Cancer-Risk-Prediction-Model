#Importing required dependencies
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.markdown("""
# 🩺 Welcome to the Lung Cancer Risk Checker!  

This tool helps estimate your risk of lung cancer based on certain lifestyle and health factors.  
Answer the questions below, and our AI model will analyze your inputs.  
            
About Dataset used for model training:\n
The Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases (IQ-OTH/NCCD) lung cancer dataset was collected in the above-mentioned specialist hospitals over a period of three months in fall 2019. 
It includes CT scans of patients diagnosed with lung cancer in different stages, as well as healthy subjects.
IQ-OTH/NCCD slides were marked by oncologists and radiologists in these two centers.
The dataset contains a total of 1190 images representing CT scan slices of 110 cases. These cases are grouped into three classes: normal, benign, and malignant. of these, 40 cases are diagnosed as malignant;
 15 cases diagnosed with benign; and 55 cases classified as normal cases. The CT scans were originally collected in DICOM format. The scanner used is SOMATOM from Siemens. CT protocol includes: 120 kV, slice thickness of 1 mm, with window width ranging from 350 to 1200 HU and window center from 50 to 600 were used for reading. with breath hold at full inspiration.
 All images were de-identified before performing analysis. Written consent was waived by the oversight review board. The study was approved by the institutional review board of participating medical centers. Each scan contains several slices. The number of these slices range from 80 to 200 slices, each of them represents an image of the human chest with different sides and angles. The 110 cases vary in gender, age, educational attainment, area of residence and living status. Some of them are employees of the Iraqi ministries of Transport and Oil, others are farmers and gainers. 
Most of them come from places in the middle region of Iraq, particularly, the provinces of Baghdad, Wasit, Diyala, Salahuddin, and Babylon.
\n
Note:
This dataset was created using the original dataset collected from Mendeley Data Publication, and various data augmentation techniques were applied to solve the class imbalance problem.

*Citation: alyasriy, hamdalla; AL-Huseiny, Muayed (2021), “The IQ-OTHNCCD lung cancer dataset”, Mendeley Data, V2, doi: 10.17632/bhmdr45bh2.2*


⚠️ **Disclaimer:** This tool is for informational purposes only and does not replace medical advice.  
""")




#Reading Data 
dfa=pd.read_csv("data2.csv")


#Handling missing values
#Replacing missing values in WHEEZING with it's mode
mode=dfa['WHEEZING'].mode()[0]
dfa['WHEEZING'] = dfa['WHEEZING'].fillna(value=mode)
#Replacing missing values in AGE with it's mode
age_mode=dfa['AGE'].mode()[0]
dfa['AGE']=dfa['AGE'].fillna(value=age_mode)


#Label Encoding for LUNG_CANCER and GENDER
le = LabelEncoder()

# 1 for Yes and 0 for No
dfa['LUNG_CANCER'] = le.fit_transform(dfa['LUNG_CANCER'])
# 1 for male 0 for female
dfa['GENDER']= le.fit_transform(dfa['GENDER'])

# Creating Dependent and Independent Variables
x=dfa.drop(columns='LUNG_CANCER')
y=dfa['LUNG_CANCER']

# Splitting data for training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=30)

# Instantiating Logistic Regression Model
model=LogisticRegression(max_iter=200)

#Fitting Training and Testing data in the model
model.fit(x_train,y_train)

#calculating Model's accuracy
sc=model.score(x_test,y_test)


#Making Ui for WebApp


#input for age
gender=st.selectbox("gender",['male','female'])

#encoding gender input
if(gender=="male"):
    encoded_gender=1
else:
    encoded_gender=0
# Integer input for age
age = st.number_input("Age", min_value=1, max_value=120, step=1)
    

# Checkbox inputs for other parameters

features={
"smoking":st.checkbox("Smoking"),
"yellow_fingers":st.checkbox("Yellow Fingers"),
"anxiety":st.checkbox("Anxiety"),
"peer_pressure":st.checkbox("Peer Pressure"),
"chronic_disease":st.checkbox("Chronic Disease"),
"fatigue":st.checkbox("Fatigue"),
"allergy":st.checkbox("Allergy"),
"wheezing":st.checkbox("Wheezing"),
"alcohol_consuming":st.checkbox("Alcohol Consuming"),
"coughing":st.checkbox("Coughing"),
"shortness_of_breath":st.checkbox("Shortness of Breath"),
"swallowing_difficulty":st.checkbox("Swallowing Difficulty"),
"chest_pain":st.checkbox("Chest Pain")
}

#encoding features accordingly
for key in features:
    features[key] = 2 if features[key] else 1


#making input list
input_list=[encoded_gender,age]+list(features.values())
features = dfa.columns.tolist()
features.remove('GENDER')
features.remove('AGE')
features.remove('LUNG_CANCER')

#making dataframe
idf = pd.DataFrame([input_list], columns=["GENDER", "AGE"] +features)

#making prediction
res=model.predict(idf)

#button action
if st.button("PREDICT"):
    if res == 0:
        st.markdown(
            """
            <div style="
                padding: 15px; 
                border-radius: 10px; 
                background-color: #D4EDDA; 
                border-left: 6px solid #28A745;
                text-align: center;
                font-size: 22px; 
                font-weight: bold;
                color: #155724;">
                ✅ You are at **low risk** of LUNG CANCER. Stay healthy! 😊
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.image("https://i.pinimg.com/originals/2d/f6/f9/2df6f9919f18d1b19045df49c1e51b7d.png", width=100)  # Healthy icon
    else:
        st.markdown(
            """
            <div style="
                padding: 15px; 
                border-radius: 10px; 
                background-color: #F8D7DA; 
                border-left: 6px solid #DC3545;
                text-align: center;
                font-size: 22px; 
                font-weight: bold;
                color: #721C24;">
                ⚠️ You are at **high risk** of LUNG CANCER. Please consult a doctor. 
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.image("https://cdn4.iconfinder.com/data/icons/men-health-color-filled/300/25043250Untitled-3-512.png", width=100)  # Warning icon
