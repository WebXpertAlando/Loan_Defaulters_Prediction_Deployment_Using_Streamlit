import streamlit as st # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
import streamlit_authenticator as stauth # type: ignore


st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="👋",
    layout= 'wide'
)

# -- LOG OUT --



## -- LOAD DATA --
df = pd.read_csv('data/loans.csv')


def numeric(df):
    numeric = df.select_dtypes(include=[np.number])
    return numeric

def categorical(df):
    categorical = df.select_dtypes(exclude=np.number)
    return categorical

    
st.session_state['Button'] = None
st.write('Select a Dashboard to display')

#EDA PLOTS
@st.spinner('Loading',_cache = True)
def Univariate_plots():
    st.title("Univariate Analysis")
    col1, col2, col3, col4 = st.columns(4)
    cols = categorical(df).columns.drop(['LoanID'])
    for i, col in enumerate(cols):
        with [col1,col2,col3,col4][i % 4]:
            data = df[col].value_counts()
            fig = px.pie(values= data,
                        names=data.index,
                        title= (f"{col} Distribution"),
                        color_discrete_map = {'German Shephard': 'rgb(255,255,0)'}
                        )
            st.plotly_chart(fig)
                
    col4,col5,col6 = st.columns(3)
    with col4:
        fig = px.box(df['Age'], 
                    orientation='h',
                    title='Distribution of Age',
                    hover_data=None
                    )
        fig.update_yaxes(visible=False, showticklabels=False)
        st.plotly_chart(fig)

    with col5:
        fig = px.box(df['LoanAmount'], 
                    orientation='h',
                    title='Distribution of Loan Amount',
                    hover_data=None
                    )
        fig.update_yaxes(visible=False, showticklabels=False)
        st.plotly_chart(fig)
        
    with col6:
        fig = px.box(df['Income'], 
                    orientation='h',
                    title='Distribution of Income',
                    hover_data= None
                    )
        fig.update_yaxes(visible=False, showticklabels=False)
        st.plotly_chart(fig)

            
@st.spinner('Loading',_cache = True)                
def Bivariate_plots():
    st.title('Bivariate Analysis')
    col1,col2,col3 = st.columns(3)
    cols = categorical(df).columns.drop(['LoanID'])
    for i, col in enumerate(cols):
        with [col1, col2, col3][i % 3]:
            data = df.groupby('Default')[col].value_counts().reset_index()
            fig = px.bar(data,
                        x=data[col],
                        y=data['count'],
                        color = data['Default'],
                        title= (f"Default vs {col}"),
                        color_discrete_map = {'German Shephard': 'rgb(255,255,0)'}
                        )
            st.plotly_chart(fig)
            
    numbers = numeric(df)
    for i, col in enumerate(numbers.columns):
        data = df.groupby('Default')[col].value_counts().reset_index()
        fig = px.bar(data,
                    x=data[col],
                    y=data['count'],
                    color=data['Default'],
                    title= (f"Default vs {col}"),
                    color_discrete_map = {'German Shephard': 'rgb(255,255,0)'}
                    )
        st.plotly_chart(fig)
    return


def Multivariate_plots():
    st.title('Multivariate Analysis')
    

def KPI_plots():
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        fig = st.metric(value=df['Default'].sum(), label='Customers Who have Defaulted')

    
colA, colB = st.columns(2)
with colA:
    EDA = st.button('EDA Dashboard', use_container_width=True)
    if EDA:
        st.session_state['Button'] = 'EDA'
with colB:
    KPI = st.button('KPI Dashboard', use_container_width=True)
    if KPI:
        st.session_state['Button'] = 'KPI'
        
        

if st.session_state['Button'] == 'EDA':
    Univariate_plots()
    Bivariate_plots()

elif st.session_state['Button'] == 'KPI':
    KPI_plots()

  
