import os
import streamlit as st
import joblib
import pandas as pd
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


st.set_page_config(
    page_title="Customer Loan Prediction App",
    page_icon="👋",
    layout= 'wide'
)
   

st.title('Select a Model and Predict Customer Churn 🔮')


@st.cache_resource(show_spinner='Loading prediction Models')
def load_adaboost():
    pipeline = joblib.load('./models/adaboost.joblib')
    return pipeline


@st.cache_resource(show_spinner='Loading prediction Models')
def load_logisticreg():
    pipeline = joblib.load('./models/logisticregression.joblib')
    return pipeline


@st.cache_resource(show_spinner='Loading Prediction Models')
def load_randomforest():
    pipeline = joblib.load('./models/randomforest.joblib')
    return pipeline

@st.cache_resource(show_spinner='Loading Prediction Models')
def load_decision_tree():
    pipeline = joblib.load('./models/decision_tree.joblib')
    return pipeline

@st.cache_resource(show_spinner ='Loading Prediction Model')
def load_catboost():
    pipeline = joblib.load('./models/catboost.joblib')
    return pipeline

@st.cache_resource(show_spinner='Loading Prdiction Models')
def loadXGBoost():
    pipeline = joblib.load('./models/xgboost.joblib')
    return pipeline

@st.cache_resource(show_spinner='Loading Prection Model')
def loadLightGBM():
    pipeline = joblib.load('./models/lightgbm.joblib')
    return pipeline


encoder = joblib.load('./models/encoder.joblib')

## Initialize the session state variables
if 'pipeline' not in st.session_state:
    st.session_state['pipeline'] = None

if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = None

if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = False        

#create buttons to select models
def selected_model():
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        AB = st.button('Adaboost')
        if AB:
            st.session_state['selected_model'] = 'AdaBoost'
            st.session_state['pipeline'] = load_adaboost()
            

    with col2:
        LR = st.button('Logistic Regression')
        if LR:
            st.session_state['selected_model'] = 'Logistic_Regression'
            st.session_state['pipeline'] = load_logisticreg()
            

    with col3:
        RF = st.button('Random Forest')
        if RF:
            st.session_state['selected_model'] = 'Random_Forest'
            st.session_state['pipeline'] = load_randomforest()  
            
    with col4:
        DT = st.button('Decision Tree')   
        if DT:
            st.session_state['selected_model'] = 'Decision_Tree'   
            st.session_state['pipeline'] = load_decision_tree()
            
    with col5:
        CB = st.button('CatBoost')
        if CB:
            st.session_state['selected_model'] = 'CatBoost'
            st.session_state['pipeline'] = load_catboost()
            
    with col6:
        XGB = st.button('XGBoost')
        if XGB:
            st.session_state['selected_model'] = 'XGBoost'
            st.session_state['pipeline'] = loadXGBoost()
            
    with col7:
        LTGBM = st.button('LightGBM')
        if LTGBM:
            st.session_state['selected_model'] = 'LightGBM'
            st.session_state['pipeline'] = loadLightGBM()
            
        
    

def make_prediction(features:pd.DataFrame, model):
    # Make prediction using the selected model
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    return prediction, probability


def predict():         
    
    selected_model()

    if st.session_state['selected_model'] == None:
        st.warning('No Model Selected, Please Select a Model to continue')
    else:
        st.success(f'You have selected : {st.session_state["selected_model"]} Classifier Model')
    
    # Collecting user input features into a list
    with st.form('input features'):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write('### Demographics')
            # Add input fields and store values in input_features dictionary
            Gender = st.radio('Gender', options=['Male', 'Female'], key='Gender', horizontal =True)
            MaritalStatus = st.radio('MaritalStatus', options=['Married', 'Divorved', 'Single'], key='maritalstatus', horizontal=True)
            HasDependents = st.radio('HasDependents', options=['Yes', 'No'], key='HasDependents', horizontal=True)
            Education = st.radio('Education', options=['Bachelor','Masters','PhD', 'High School'], key='Education', horizontal=True)

        with col2:
            st.write('### Basic  Requirements')
            # Input fields for subscription-related features
            EmploymentType = st.radio('EmploymentType', options=['Full-time', 'Unemployed', 'Self-employed'], key='EmployementType', horizontal=True)             
            MultipleLines = st.radio('MultipleLines', options=['Yes', 'No'], key='MultipleLines', horizontal=True) 
            InternetService = st.radio('InternetService', options=['Fiber optic', 'DSL', 'No'], key='InternetService') 
            OnlineSecurity = st.radio('OnlineSecurity', options=['Yes', 'No'], key='OnlineSecurity', horizontal=True) 
            
        with col3:
            st.write('### Other Requirements')
            OnlineBackup = st.radio('OnlineBackup', options=['Yes', 'No'], key='OnlineBackup', horizontal=True) 
            DeviceProtection = st.radio('DeviceProtection', options=['Yes', 'No'], key='DeviceProtection', horizontal=True) 
            TechSupport = st.radio('TechSupport', options=['Yes', 'No'], key='TechSupport', horizontal=True) 
            StreamingTV = st.radio('StreamingTV', options=['Yes', 'No'], key='StreamingTV', horizontal=True) 
            StreamingMovies = st.radio('StreamingMovies', options=['Yes', 'No'], key='StreamingMovies', horizontal=True)
            
        with col4:
            st.write('### Billing')
            # Input fields for payment-related features
            Contract = st.radio('Contract', options=['Month-to-month', 'Two year', 'One year'], key='Contract') 
            PaperlessBilling = st.radio('PaperlessBilling', options=['Yes', 'No'], key='PaperlessBilling', horizontal=True) 
            PaymentMethod = st.radio('PaymentMethod', options=['Electronic check', 'Credit card (automatic)', 'Mailed check', 'Bank transfer (automatic)'], key='PaymentMethod') 
            
        col5,col6,col7,col8,col9,col10,col11,col12,col13,col14 = st.columns(10)
        
        with col5:
            Age = st.slider('Age', min_value=1, max_value=60, step=1, key='Age') 
        
        with col6:
            Income = st.slider('Income', min_value=1, max_value=100000, key='Income')
        
        with col7:
            LoanAmount = st.slider('LoanAmount', min_value=1, max_value=50000)

        with col8:
            CreditScore = st.slider('CreditScore', min_value=1, max_value=1000)

        with col9:
            MonthsEmployed = st.slider('MonthsEployed', min_value=1, max_value=200)

        with col10:
            NumCreditLines = st.slider('NumCreditLines', min_value=1, max_value=1000)
        
        with col11:
            InterestRate = st.slider('InterestRate',min_value=1, max_value=500)
            

         
            
            if st.form_submit_button('Submit'):
                st.session_state['button_clicked'] = True
                         
            input_features = pd.DataFrame({
                    'age': [Age],
                    'income': [Income],
                    'loanamount': [LoanAmount],
                    'creditscore': [CreditScore],
                    'monthsemployed': [MonthsEmployed],
                    'numcreditlines': [NumCreditLines],
                    'interestrate': [InterestRate],
                    'loanterm': [LoanTerm],
                    'dtiratio': [DTIRatio],
                    'education': [Education],
                    'employmenttype': [EmployementType],
                    'maritalstatus': [MaritalStatus],
                    'hasmortgage': [HasMortgage],
                    'hasdependants': [HasDependents],
                    'loanpurpose': [LoanPurpose],
                    'hascosigner': [HasCoSigner],
                    
                })
            pred_features = input_features.astype('object')
            
            #check whether pipeline is selected
            if st.session_state['pipeline'] is not None:
                # Make prediction using the selected model
                prediction, probabilities = make_prediction(pred_features, st.session_state['pipeline'])
            else:
                st.error('No Model Selected, Please Select a Model above to continue')
                sys.exit()

            if prediction == 0:
                prediction = 'Not Defaulting'
            else:
                prediction = 'Default'
            
            probability = pd.DataFrame(probabilities).T
            stay_proba = round(probability[0][0]*100, ndigits=2)
            default_proba = round(probability[0][1]*100, ndigits=2)
            
            # Display prediction results
            st.markdown("### **Prediction Results**")
            if prediction == 'Not Defaulting':
                st.success(f"The customer has a {stay_proba}% probability of Not Defaulting")
            else:
                st.error(f"The customer is likely to default with a {default_proba}% probability")
            
            # Add the prediction, churn probability, and model used to the input_features
            input_features['prediction'] = [prediction]
            input_features['defaulting probability %'] = [default_proba]
            input_features['remain probability %'] = [stay_proba]
            input_features['model_used'] = [st.session_state['selected_model']]
            
            # Check if the history.csv file exists
            if os.path.exists('data/history.csv'):
                # If it exists, read the existing data and append the input features
                history_data = pd.read_csv('data/history.csv')
                history_data = pd.concat([history_data, input_features], ignore_index=True)
            else:
                # If it doesn't exist, create a new dataframe with the input features
                history_data = input_features
            
            history_data.to_csv('data/history.csv', index=False)

                       
# Call the data function directly
if __name__ == '__main__':
    predict()
    if st.session_state['button_clicked']:
        response = st.radio('Do you want to try another model?', options=['No','Yes'])
        if response == 'Yes':
            st.session_state['button_clicked'] = False
            st.session_state['selected_model'] = None
            st.session_state['pipeline'] = None
            st.experimental_rerun()