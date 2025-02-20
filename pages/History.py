import streamlit as st
import pandas as pd
import os
import time

st.set_page_config(
    page_title="Prediction History",
    page_icon="👋",
    layout= 'wide'
)

st.title('Prediction History')

def load_history():
    if os.path.exists('data/history.csv'):
        history_df = pd.read_csv('data/history.csv')
    else:
        history_df = pd.DataFrame()  # Create an empty DataFrame if the file doesn't exist
    return history_df

def clear_history():
    if os.path.exists('data/history.csv'):
        with st.spinner('Cleaning past history...'):
            os.remove('data/history.csv')
            time.sleep(3)
        st.success('History Cleared Succesfully')
        time.sleep(3)  # Add a delay
        st.experimental_rerun()
    else:
        st.warning("History is already empty.")
        time.sleep(2)  # Add a 4-second delay
        st.experimental_rerun()
        

def main(): 
    # Load history data
    history_df = load_history()
    
    # Display history data if available
    if not history_df.empty:
        st.write(history_df)
    else:
        st.error("History is empty.")
    
    # Add a clear button
    if st.button('Clear History'):
        clear_history()
        time.sleep(2)  # Add a 4-second delay
        st.experimental_rerun()

if __name__ == '__main__':
    main()