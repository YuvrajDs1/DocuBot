import streamlit as st
import requests

st.title('DocuBot 🤖')

uploaded_file = st.file_uploader("Enter your file ", type=['pdf', 'docx'])
