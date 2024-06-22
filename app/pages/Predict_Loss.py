import streamlit as st
import pickle
import streamlit.web.cli as stcli
import pandas as pd
import numpy as np
import seaborn as sns
import re
import time
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Predict Loss",
)

def result():
    results_df = pd.read_parquet("data_results.parquet")
    X_test = pd.read_parquet("X_test.parquet")
    results_df["loan_amnt"] = X_test["loan_amnt"].copy().reset_index(drop=True)
    results_df["lgd"] = 1

    results_df["lr_expected_lost"] = results_df["lr_pred_proba"] * results_df["lgd"] * results_df["loan_amnt"]
    results_df["rfc_expected_lost"] = results_df["rfc_pred_proba"] * results_df["lgd"] * results_df["loan_amnt"]
    results_df["xgb_expected_lost"] = results_df["xgb_pred_proba"] * results_df["lgd"] * results_df["loan_amnt"]
    results_df["tabnet_expected_lost"] = results_df["tabnet_pred_proba"] * results_df["lgd"] * results_df["loan_amnt"]
    results_df["catboost_expected_lost"] = results_df["catboost_pred_proba"] * results_df["lgd"] * results_df["loan_amnt"]
    results_df["total_loss"] = results_df["loan_status"] * results_df["lgd"] * results_df["loan_amnt"]

    sum_loan_amount = results_df["loan_amnt"].sum()
    sum_total_loss = results_df["total_loss"].sum()
    sum_lr_expected_loss = results_df["lr_expected_lost"].sum()
    sum_rfc_expected_loss = results_df["rfc_expected_lost"].sum()
    sum_xgb_expected_loss = results_df["xgb_expected_lost"].sum()
    sum_tabnet_expected_loss = results_df["tabnet_expected_lost"].sum()
    sum_catboost_expected_loss = results_df["catboost_expected_lost"].sum()

    st.success(f"Total Loan Amount: ${sum_loan_amount:,.2f}")
    st.success(f"Total Loss : ${sum_total_loss:,.2f}")
    st.success(f"Expected Loss By LR: ${sum_lr_expected_loss:,.2f}")
    st.success(f"Expected Loss By RFC: ${sum_rfc_expected_loss:,.2f}")
    st.success(f"Expected Loss By XGB: ${sum_xgb_expected_loss:,.2f}")
    st.success(f"Expected Loss By Tabnet: ${sum_tabnet_expected_loss:,.2f}")
    st.success(f"Expected Loss By Catboost: ${sum_catboost_expected_loss:,.2f}")

    # loan_status = results_df["loan_status"].value_counts()
    # fig1, ax1 = plt.subplots()
    # ax1.pie(loan_status, labels = loan_status.index, autopct="%1.1f%%", startangle = 90)
    # ax1.axis("equal")
    # st.write("""##### Loan Status""")
    # st.pyplot(fig1)

    # lr_pred = results_df["lr_pred"].value_counts()
    # fig2, ax2 = plt.subplots()
    # ax2.pie(lr_pred, labels = lr_pred.index, autopct="%1.1f%%", startangle = 90)
    # ax2.axis("equal")
    # st.write("""##### Logistic Regression""")
    # st.pyplot(fig2)

    # rfc_pred = results_df["rfc_pred"].value_counts()
    # fig3, ax3 = plt.subplots()
    # ax3.pie(rfc_pred, labels = rfc_pred.index, autopct="%1.1f%%", startangle = 90)
    # ax3.axis("equal")
    # st.write("""##### Random Forest Classifier""")
    # st.pyplot(fig3)

    # xgb_pred = results_df["xgb_pred"].value_counts()
    # fig4, ax4 = plt.subplots()
    # ax4.pie(xgb_pred, labels = xgb_pred.index, autopct="%1.1f%%", startangle = 90)
    # ax4.axis("equal")
    # st.write("""##### XGBoost""")
    # st.pyplot(fig4)

def main():
    st.title("Loan Status Expectation")
    st.header("Predict the expected loss")

    st.write("##")

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

    if st.button('Submit'):
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()
        result()
    
if __name__=='__main__':
    main()