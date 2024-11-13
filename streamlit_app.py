import streamlit as st
# import sqlite3
# import pysqlite3
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from openai import OpenAI
# from st_pages import Page, show_pages, add_page_title

import random  
import hmac  
  
# """  
# This file contains the common components used in the Streamlit App.  
# This includes the sidebar, the title, the footer, and the password check.  
# """  

# Show title and description.
st.title("Welcome")
st.write(
    "Welcome to the Home page! This is the CPF Policy Explainer. "
    "The objective of this app is to help you plan your retirement based on CPF policies. "
    "Please use the sidebar to navigate between the various relevant topics you are interest in."
)

# Disclaimer
with st.expander("Disclaimer"):
    st.write("""

    IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

    Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

    Always consult with qualified professionals for accurate and personalized advice.

    """) 
