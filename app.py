"""
Car Financial RAG - Streamlit entry point.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(
    page_title="Car Financial RAG",
    layout="centered",
)

pg = st.navigation([
    st.Page("pages/home.py", title="Home"),
    st.Page("pages/chat.py", title="Chat"),
])
pg.run()
