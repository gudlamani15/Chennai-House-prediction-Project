#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pickle
with open("model_pickle","rb") as f:
    mp = pickle.load(f)
    
def prediction(AREA, INT_SQFT, N_BEDROOM, N_BATHROOM, N_ROOM, SALE_COND,
       PARK_FACIL, UTILITY_AVAIL, STREET, MZZONE,
       BUILDTYPE_Commercial, BUILDTYPE_House, BUILDTYPE_Others):  
    X_test1 =[[AREA, INT_SQFT, N_BEDROOM, N_BATHROOM, N_ROOM, SALE_COND,
       PARK_FACIL, UTILITY_AVAIL, STREET, MZZONE,
       BUILDTYPE_Commercial, BUILDTYPE_House, BUILDTYPE_Others]]
    prediction = mp.predict(X_test1)
    print(prediction)
    return prediction


# In[3]:


import streamlit as st


# In[4]:


def main():
    st.title("Chennai House Prediction")
    Area = st.selectbox("Area:",['Karapakkam', 'Anna Nagar', 'Adyar', 'Velchery', 'Chormpet',
       'KK Nagar', 'T Nagar'])
    Areaencoding = {"Karapakkam":0,"Adyar":1,"Chormpet":2,"Velchery":3,"KK Nagar":4,"Anna Nagar":5,"T Nagar":6}
    AREA = Areaencoding[Area]
    st.success(Area)
    INT_SQFT = st.number_input("Enter Squarefeet of house")
    N_BEDROOM = st.number_input("Enter number of bedroom as a whole number")
    N_BATHROOM = st.number_input("Enter number of bathroom as a whole number")
    N_ROOM = st.number_input("Enter number of room as a whole number")
    salecondition = st.selectbox("Sale_condtion",['Ab Normal', 'Family', 'Partial', 'Adj Land', 'Normal Sale'])
    saleencoding = {"Partial":0,"Family":1,"Ab Normal":2,"Normal Sale":3,"Adj Land":4}
    SALE_COND = saleencoding[salecondition]
    st.success(salecondition)
    status = st.radio("Parking Facility: ", ('Yes', 'No'))
    if (status == 'yes'):
        st.success("yes")
        PARK_FACIL = 1
    else:
        st.success("No")
        PARK_FACIL = 0
        
    utility_avail = {"ELO":0,"NoSeWa":1,"NoSewr ":2,"AllPub":3}
    utility = st.selectbox("Utility Available: ",["ELO","NoSeWa","NoSewr ","AllPub"])
    UTILITY_AVAIL = utility_avail[utility]
    street = {"No Access":0,"Paved":1,"Gravel":2}
    value = st.selectbox("Street: ",["No Access","Paved","Gravel"])
    STREET = street[value]
    Mzzone = {"A":0,"C":1,"I":2,"RH":3,"RL":4,"RM":5}
    zvalue = st.selectbox("Mzzone: ",["A","C","I","RH","RL","RM"])
    MZZONE = Mzzone[zvalue]
    
    BuildType = st.selectbox("Build Type: ",["House","commercial","others"])
    if BuildType == "House":
        BUILDTYPE_House = 1
        BUILDTYPE_Commercial = 0 
        BUILDTYPE_Others = 0
    elif BuildType == "commercial":
        BUILDTYPE_House = 0
        BUILDTYPE_Commercial = 1
        BUILDTYPE_Others = 0
    else:
        BUILDTYPE_House = 0
        BUILDTYPE_Commercial = 0 
        BUILDTYPE_Others = 1
    result =""
    if st.button("Predict"):
        result = prediction(AREA, INT_SQFT, N_BEDROOM, N_BATHROOM, N_ROOM, SALE_COND,
       PARK_FACIL, UTILITY_AVAIL, STREET, MZZONE,
       BUILDTYPE_Commercial, BUILDTYPE_House, BUILDTYPE_Others)
    st.success('The output is {}'.format(result))
        
main()       


# In[ ]:




