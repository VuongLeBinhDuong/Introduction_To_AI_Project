import streamlit as st
import pickle
import streamlit.web.cli as stcli
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score

st.set_page_config(
    page_title="Model Assessment",
)

def my_classification_report(y_test, y_proba_preds, labels={0:"Non-Default", 1:"Default"}, threshold=0.7):
    
    y_preds = np.where(y_proba_preds >= threshold, 1, 0)
    
    report_df = pd.DataFrame()
    for i in labels:
        precision = precision_score(y_test, y_preds, pos_label=i)*100
        recall = recall_score(y_test, y_preds, pos_label=i)*100
        f1 = f1_score(y_test, y_preds, pos_label=i)*100
        
        tmp_df = pd.DataFrame([f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}"], columns=[labels[i]])
        report_df = pd.concat([report_df, tmp_df], axis=1)
        
        report_df = report_df.rename(index={0:"Precision", 1:"Recall", 2:"F1_score"})
        
        accuracy = accuracy_score(y_test, y_preds)*100
        return [f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}", f"{accuracy:.2f}"]

def plot_roc_curve(y_test, data=None, col_names={}, colors={}):
    plt.figure(figsize=(12,10))
    
    ns_proba = np.ones(len(y_test)) * 0.5
    ns_fpr, ns_sensitivity, ns_thresholds = roc_curve(y_test, ns_proba)
    ns_auc = roc_auc_score(y_test, ns_proba)
    plt.plot(ns_fpr, ns_sensitivity, linestyle='-.', color='steelblue', label="No Skill")
    
    plt.text(0.9, 0.1, f"AUC = {ns_auc:.3f}", ha="right", fontsize=16, weight="bold", color="steelblue")
    plt.fill_between(ns_fpr, ns_sensitivity, facecolor="LightSteelBlue", alpha=0.6)
    
    y_auc = 0.2
    for col in col_names:
        y_proba = data[col]
        fpr, sensitivity, thresholds = roc_curve(y_test, y_proba)
        model_auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, sensitivity, color=colors[col], label=col_names[col])
        plt.text(0.9, y_auc, f"AUC = {model_auc:.3f}", ha="right", fontsize=16, weight="bold", color=colors[col])
        plt.fill_between(fpr, sensitivity, facecolor="#FFD580", alpha=0.4)
        y_auc += 0.1
        
    plt.title(f"ROC Chart")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Sensitivity")
    plt.legend(fontsize=14)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def main():
    st.title("Loan Status Expectation")
    st.header("Model Assessment")
    results_df = pd.read_parquet("data_results.parquet")
    lr = my_classification_report(results_df['loan_status'], results_df['lr_pred_proba'])
    rfc = my_classification_report(results_df['loan_status'], results_df['rfc_pred_proba'])
    xgb = my_classification_report(results_df['loan_status'], results_df['xgb_pred_proba'])
    tabnet = my_classification_report(results_df['loan_status'], results_df['tabnet_pred_proba'])
    catboost = my_classification_report(results_df['loan_status'], results_df['catboost_pred_proba'])
    st.write("#####")
    st.subheader("Logistic Regression")
    st.write("Precision: " + str(lr[0]) + "%")
    st.write("Recall: " + str(lr[1]) + "%")
    st.write("F1 Score: " + str(lr[2]) + "%")
    st.write("Accuracy: " + str(lr[3]) + "%")
    st.subheader("Random Forest Classifier")
    st.write("Precision: " + str(rfc[0]) + "%")
    st.write("Recall: " + str(rfc[1]) + "%")
    st.write("F1 Score: " + str(rfc[2]) + "%")
    st.write("Accuracy: " + str(rfc[3]) + "%")
    st.subheader("XGBoost")
    st.write("Precision: " + str(xgb[0]) + "%")
    st.write("Recall: " + str(xgb[1]) + "%")
    st.write("F1 Score: " + str(xgb[2]) + "%")
    st.write("Accuracy: " + str(xgb[3]) + "%")
    st.subheader("Tabnet")
    st.write("Precision: " + str(tabnet[0]) + "%")
    st.write("Recall: " + str(tabnet[1]) + "%")
    st.write("F1 Score: " + str(tabnet[2]) + "%")
    st.write("Accuracy: " + str(tabnet[3]) + "%")
    st.subheader("Catboost")
    st.write("Precision: " + str(catboost[0]) + "%")
    st.write("Recall: " + str(catboost[1]) + "%")
    st.write("F1 Score: " + str(catboost[2]) + "%")
    st.write("Accuracy: " + str(catboost[3]) + "%")
    
    dict_names = {
        "lr_pred_proba": "Logistic Regression",
        "rfc_pred_proba": "Random Forest Classifier",
        "xgb_pred_proba": "XGBoost Classifier",
        "tabnet_pred_proba": "Tabnet Classifier",
        "catboost_pred_proba": "Catboost Classifier"
    }

    colors = {
        "lr_pred_proba": "#FF8C00",
        "rfc_pred_proba": "#9400D3",
        "xgb_pred_proba": "#2E8B57",
        "tabnet_pred_proba": "#FF5733",
        "catboost_pred_proba": "#1E90FF"
    }
    st.write("######")
    st.subheader("ROC Chart")
    plot_roc_curve(results_df["loan_status"], data=results_df, col_names=dict_names, colors=colors)

if __name__=='__main__':
    main()
