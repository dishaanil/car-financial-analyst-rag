"""
Home page - overview, data coverage, and example questions.
"""

import streamlit as st

st.title("Automotive Annual Report Analyst")
st.caption("A tool for extracting and comparing financial data across automotive sector annual reports.")

st.divider()

# About

st.subheader("About")
st.markdown(
    """
    Analysts working with automotive sector annual reports spend a lot of time manually
    pulling key financial metrics like revenue, EBITDA and growth figures, then comparing
    them across companies. This tool automates that process.

    Ask a question in plain English and get answers drawn directly from the source reports,
    with page-level references so you can verify every figure.
    """
)

st.divider()

# Data coverage

st.subheader("Available Data")
st.markdown("The following annual reports have been indexed:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**BMW**")
    st.markdown("2021 Group Report")
    st.markdown("2022 BMW Finance N.V.")
    st.markdown("2023 BMW Finance N.V.")

with col2:
    st.markdown("**Ford**")
    st.markdown("2021 Annual Report")
    st.markdown("2022 Annual Report")
    st.markdown("2023 Annual Report")

with col3:
    st.markdown("**Tesla**")
    st.markdown("2022 Annual Report")
    st.markdown("2023 Annual Report")

st.divider()

# Example questions

st.subheader("What You Can Ask")
st.markdown("Some examples to get you started:")

st.markdown(
    """
    - What was Tesla's total revenue in 2023?
    - Compare Ford and Tesla net income in 2022
    - Give me a 3-year revenue summary for all three companies
    - How did BMW's revenue grow between 2021 and 2023?
    - Which Tesla vehicles are currently in development?
    - How did Ford's operating income change between 2021 and 2023?
    """
)

st.divider()

# CTA

st.markdown("Ready to explore the data?")
if st.button("Start chatting", type="primary", use_container_width=False):
    st.switch_page("pages/chat.py")
