import streamlit as st

# Set up the page configuration
st.set_page_config(page_title="Loan Status Information", layout="wide")

# Creating a title
st.title('Loan Status Information')

st.snow()

# Introduction text

st.markdown("""
Welcome to the Loan Status Information Page! Here you can find all you need to know 
about how to apply for a loan, the criteria for approval, and tips for ensuring your 
loan application is successful.

#### How to Apply:
- **Step 1:** Gather required documents, including proof of income, identity, and residence.
- **Step 2:** Complete the application form on our secure application portal.
- **Step 3:** Submit your application and await preliminary approval.

#### Loan Approval Criteria:
- **Credit Score:** A minimum score of 620 is generally required.
- **Income Verification:** Your income must be verified through recent pay stubs or tax returns.
- **Debt-to-Income Ratio:** Must be under 45% to ensure you can repay the loan comfortably.

#### Tips for a Successful Application:
- **Check Your Credit Report:** Ensure there are no errors that could negatively affect your score.
- **Calculate Your Needs:** Only borrow what you need and can realistically repay.
- **Understand the Terms:** Read the terms and conditions carefully before submitting your application.

For more detailed information, please contact our support team or visit our FAQ section.
""")

# Optional: Adding an image or visual
# st.image('path_to_image.jpg', caption='Visual guide to Loan Application Process')

# Contact information
st.sidebar.header("Contact Information")
st.sidebar.write("For more assistance, reach out via:")
st.sidebar.write("📞 Phone: +1 234 567 8900")
st.sidebar.write("📧 Email: help@loaninfo.com")
st.sidebar.write("🌐 Website: www.loaninfo.com")


