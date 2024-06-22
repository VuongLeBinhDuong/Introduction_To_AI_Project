import streamlit as st
import pickle
import streamlit.web.cli as stcli
import time

lr_model=pickle.load(open('lr_model.pkl','rb'))
#rfc_model=pickle.load(open('rfc_model.pkl','rb'))
xgb_model=pickle.load(open('xgb_model.pkl','rb'))
tabnet_model = pickle.load(open('tabnet_model.pkl','rb'))
catboost_model = pickle.load(open('catboost_model.pkl','rb'))

st.set_page_config(
    page_title="Predict Probability",
)

import pandas as pd

def preprocess_input(loan_amnt, int_rate, emp_length, annual_inc, dti, pub_rec_bankruptcies,
                     chargeoff_within_12_mths, month_cr_history, term, house_ownership, purpose, app_type):
    data = {
        'loan_amnt': [loan_amnt],
        'int_rate': [int_rate],
        'emp_length': [emp_length],
        'annual_inc': [annual_inc],
        'dti': [dti],
        'pub_rec_bankruptcies': [pub_rec_bankruptcies],
        'chargeoff_within_12_mths': [chargeoff_within_12_mths],
        "month_cr_history": [month_cr_history],
        'term_36_months': [1 if term == '36 months' else 0],
        'term_60_months': [1 if term == '60 months' else 0],
        'home_ownership_ANY': [1 if house_ownership == 'Any' else 0],
        'home_ownership_MORTGAGE': [1 if house_ownership == 'Mortgage' else 0],
        'home_ownership_NONE': [1 if house_ownership == 'None' else 0],
        'home_ownership_OTHER': [1 if house_ownership == 'Other' else 0],
        'home_ownership_OWN': [1 if house_ownership == 'Own' else 0],
        'home_ownership_RENT': [1 if house_ownership == 'Rent' else 0],
        'purpose_car': [1 if purpose == 'Car' else 0],
        'purpose_credit_card': [1 if purpose == 'Credit Card' else 0],
        'purpose_debt_consolidation': [1 if purpose == 'Debt Consolation' else 0],
        'purpose_educational': [1 if purpose == 'Educational' else 0],
        'purpose_home_improvement': [1 if purpose == 'Home Improvement' else 0],
        'purpose_house': [1 if purpose == 'House' else 0],
        'purpose_major_purchase': [1 if purpose == 'Major Purchase' else 0],
        'purpose_medical': [1 if purpose == 'Medical' else 0],
        'purpose_moving': [1 if purpose == 'Moving' else 0],
        'purpose_other': [1 if purpose == 'Other' else 0],
        'purpose_renewable_energy': [1 if purpose == 'Renewable Energy' else 0],
        'purpose_small_business': [1 if purpose == 'Small Business' else 0],
        'purpose_vacation': [1 if purpose == 'Vacation' else 0],
        'purpose_wedding': [1 if purpose == 'Wedding' else 0],
        'app_type_Individual': [1 if app_type == 'Individual' else 0],
        'app_type_Joint_App': [1 if app_type == 'Joint App' else 0]
    }
    df = pd.DataFrame(data)
    return df

def submit(prediction):
    output='{0:.{1}f}'.format(prediction[0][1]*100, 2)
    return output

def main():
    st.title("Loan Status Expectation")
    st.header("Predict the probability of successfully repay")
    activities=['Logistic Regression','Random Forest Classifier', 'XGBoost', "Tabnet", "CatBoost"]
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    loan_amnt=st.number_input('Loan amount: ',value=None, step=1.,format="%.2f")
    int_rate=st.number_input('Interest rate: ', value=None,step=1.,format="%.2f")
    emp_length=st.number_input('Employment length: ', value=None,step=1.,format="%.2f")
    annual_inc=st.number_input('Annual increase: ', value=None,step=1.,format="%.2f")
    dti=st.number_input('Debt-to-income ratio: ', value=None,step=1.,format="%.2f")
    pub_rec_bankruptcies=st.number_input('Number of public record bankruptcies: ', value=None,step=1.,format="%.2f")
    chargeoff_within_12_mths=st.number_input('Charge-offs within 12 months:', value=None,step=1.,format="%.2f")
    month_cr_history = st.number_input('Month Credit History: ', value=None,step=1.,format="%.2f")
    term = st.selectbox(
    'Term: ',
    ('36 months', '60 months'),
    index=None)
    house_ownership = st.selectbox(
    'House Ownership: ',
    ('Rent', 'Own', 'Mortgage','Any', 'None', 'Other'),
    index=None)
    purpose = st.selectbox(
    'Purpose: ',
    ('Business', 'Car', 'Credit Card', 'Debt Consolation', 'Educational', 'Home Improvement', 'House', 'Major Purchase', 'Medical', 'Moving', 'Renewable Energy', 'Vacation', 'Wedding', 'Other'),
    index=None)
    app_type = st.selectbox(
    'Application Type: ',
    ('Individual', 'Joint App'),
    index=None)

    inputs = preprocess_input(loan_amnt, int_rate, emp_length, annual_inc, dti, pub_rec_bankruptcies,
                                  chargeoff_within_12_mths, month_cr_history, term, house_ownership, purpose, app_type)
    if st.button('Submit'):
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()
        if option=='Logistic Regression':
            if (float(submit(lr_model.predict_proba(inputs))) < 70):
                st.success("The probability that the person can not pay loan: " + submit(lr_model.predict_proba(inputs)) + "%")
            else:
                st.error("The probability that the person can not pay loan: " + submit(lr_model.predict_proba(inputs)) + "%")
        # elif option=='Random Forest Classifier':
            # if (float(submit(rfc_model.predict_proba(inputs))) < 70):
            #     st.success("The probability that the person can not pay loan: " + submit(rfc_model.predict_proba(inputs)) + "%")
            # else:
            #     st.error("The probability that the person can not pay loan: " + submit(rfc_model.predict_proba(inputs)) + "%")
        elif option=='Tabnet':
            if (float(submit(tabnet_model.predict_proba(inputs.values))) < 70):
                st.success("The probability that the person can not pay loan: " + submit(tabnet_model.predict_proba(inputs.values)) + "%")
            else:
                st.error("The probability that the person can not pay loan: " + submit(tabnet_model.predict_proba(inputs.values)) + "%")
        elif option=='CatBoost':
            if (float(submit(catboost_model.predict_proba(inputs.astype(int, errors='ignore')))) < 70):
                st.success("The probability that the person can not pay loan: " + submit(catboost_model.predict_proba(inputs.astype(int, errors='ignore'))) + "%")
            else:
                st.error("The probability that the person can not pay loan: " + submit(catboost_model.predict_proba(inputs.astype(int, errors='ignore'))) + "%")   
        else:
            if (float(submit(xgb_model.predict_proba(inputs))) < 70):
                st.success("The probability that the person can not pay loan: " + submit(xgb_model.predict_proba(inputs)) + "%")
            else:
                st.error("The probability that the person can not pay loan: " + submit(xgb_model.predict_proba(inputs)) + "%")
    

if __name__=='__main__':
    main()