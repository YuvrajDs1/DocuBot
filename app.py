import streamlit as st
import requests

st.title('DocuBot 🤖')

uploaded_file = st.file_uploader("Upload a PDF or Word File", type=['pdf', 'docx'])
question = st.text_input('Enter your Query')

if uploaded_file and question:
    files = {
        "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
    }

    response = requests.post(
        f"http://127.0.0.1:8000/ask/?question={question}",
        files=files,
    )

    if response.status_code == 200:
        res = response.json()
        st.write("**Answer:**", res["answer"])
        if "sources" in res:
            with st.expander("Sources"):
                for i, src in enumerate(res["sources"], 1):
                    st.write(f"Source {i}:", src)
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
