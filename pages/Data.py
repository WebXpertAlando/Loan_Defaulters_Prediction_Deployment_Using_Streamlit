import streamlit as st # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="👋",
    layout= 'wide'
)

st.title('View Data used in Modeling')
st.write('Click the buttons to show the dataset')

df = pd.read_csv('data/loans.csv')

#show numeric columns
def show_numerics():
    numeric = df.select_dtypes(include=[np.number]).columns
    for col in numeric:
        df[col] = df[col].apply(lambda x: f"{x:.2f}")
    return df[numeric]

#show categorical columns
def show_categoricals():
    categorical = df.select_dtypes(exclude=[np.number]).columns
    return df[categorical].drop(columns=['LoanID'])

def SysRerun():
    st.experimental_rerun()

col1, col2, col3 = st.columns(3)
with col1:
    All = st.button('Full Dataset',help='Display the full dataset')
with col2:
    Numerics = st.button('Numeric Variables', help='Display only the numeric columns')
with col3:
    Categorical = st.button('Catgeorical Variables', help='Display only Categorical columns')

if Numerics:
    st.dataframe(show_numerics())
    Hide = st.button('Hide', help='Hide the Data')
    if Hide:
        SysRerun()
    
elif Categorical:
    st.dataframe(show_categoricals())
    Hide = st.button('Hide', help='Hide the Data')
    if Hide:
        SysRerun()

elif All:
    st.dataframe(df)
    Hide = st.button('Hide', help='Hide the Data')
    if Hide:
        SysRerun()
else:
    pass