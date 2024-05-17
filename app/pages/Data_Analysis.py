import streamlit as st
import pickle
import streamlit.web.cli as stcli
import time
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Data Analysis",
)

def main():
    st.title("Loan Status Expectation")
    st.header("Analyzing data")

    data = pd.read_parquet("essential_df.parquet")

    st.write("###")

    homeownership = data["home_ownership"].value_counts()
    st.write("""##### Number of home ownership data""")
    st.bar_chart(homeownership)

    st.write("""##### Number of purpose data""")
    purpose = data["purpose"].value_counts()
    st.bar_chart(purpose)

    term = data["term"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(term, labels = term.index, autopct="%1.1f%%", startangle = 90)
    ax1.axis("equal")
    st.write("""##### Number of term data""")
    st.pyplot(fig1)

    type = data["application_type"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(type, labels = type.index, autopct="%1.1f%%", startangle = 90)
    ax2.axis("equal")
    st.write("""##### Number of application type data""")
    st.pyplot(fig2)

if __name__=='__main__':
    main()