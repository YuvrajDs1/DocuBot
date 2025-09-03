import streamlit as st
import requests
import time
import os  

BACKEND_URL = os.getenv("BACKEND_URL", "https://docubot-6v6r.onrender.com")

st.set_page_config(
    page_title="DocuBot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .question-section {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">DocuBot 🤖</h1>', unsafe_allow_html=True)
st.markdown("**Upload a document and ask questions about its content!**")

col1, col2 = st.columns([2, 1])

with col1:

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or Word file",
        type=['pdf', 'docx'],
        help="Supported formats: PDF (.pdf) and Word documents (.docx)",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="question-section">', unsafe_allow_html=True)
    st.subheader("❓ Ask Question")
    question = st.text_area(
        'Enter your question about the document:',
        placeholder="What would you like to know about the document?",
        height=100,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("ℹ️ How to use:")
    st.markdown("""
    1. **Upload** a PDF or DOCX file
    2. **Ask** a question about the content
    3. **Get** AI-powered answers with sources
    
    **Tips:**
    - Be specific in your questions
    - Try different phrasings if needed
    - Check sources to verify information
    """)

if uploaded_file and question:
    if st.button("🚀 Get Answer", type="primary", use_container_width=True):
        with st.spinner('🔍 Processing your document and analyzing the question...'):
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                data = {"question": question}
                
                start_time = time.time()
                response = requests.post(
                    f"{BACKEND_URL}/ask/",
                    files=files,
                    data=data,
                    timeout=120
                )
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    res = response.json()
                    st.success(f"✅ Answer generated in {processing_time:.2f} seconds!")
                    st.markdown("### 🎯 Answer:")
                    answer_container = st.container()
                    with answer_container:
                        st.markdown(f"**{res['answer']}**")
                    
                    if "sources" in res and res["sources"]:
                        st.markdown("### 📚 Sources:")
                        with st.expander(f"View {len(res['sources'])} source(s)", expanded=False):
                            for i, src in enumerate(res["sources"], 1):
                                st.markdown(f"**📄 Source {i}:**")
                                st.text_area(
                                    f"Content from source {i}",
                                    value=src,
                                    height=120,
                                    key=f"source_{i}_{hash(src[:50])}",
                                    label_visibility="collapsed"
                                )
                                if i < len(res["sources"]):
                                    st.divider()
                    else:
                        st.info("ℹ️ No specific sources found for this answer.")
                        
                elif response.status_code == 400:
                    st.error("❌ Bad request: Please check your question and try again.")
                elif response.status_code == 500:
                    st.error("❌ Server error: There was an issue processing your request.")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Error**: Cannot connect to the backend server.")
                st.info(f"Make sure your backend is running and accessible at {BACKEND_URL}")
            except requests.exceptions.Timeout:
                st.error("❌ **Timeout Error**: The request took too long to process. Try a smaller document.")
            except Exception as e:
                st.error(f"❌ **Unexpected Error**: {str(e)}")

elif uploaded_file and not question:
    st.info("❓ Please enter a question about your document.")
elif question and not uploaded_file:
    st.info("📄 Please upload a document first.")
else:
    st.info("👆 Upload a document and enter a question to get started!")

with st.sidebar:
    st.markdown("## 🛠️ Controls")
    
    if st.button("🔍 Check Server Status"):
        try:
            health_response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ Server is running!")
                st.json(health_response.json())
            else:
                st.error("❌ Server not responding properly")
        except:
            st.error("❌ Server is not running")
    
    if st.button("🧹 Clean Server Files"):
        try:
            cleanup_response = requests.delete(f"{BACKEND_URL}/cleanup/", timeout=10)
            if cleanup_response.status_code == 200:
                st.success("✅ Server files cleaned up!")
            else:
                st.error("❌ Failed to clean up server files")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to server")
        except Exception as e:
            st.error(f"❌ Cleanup failed: {str(e)}")
    
    st.markdown("---")
    st.markdown("## 📋 Supported Files")
    st.markdown("""
    - **PDF files** (.pdf)
    - **Word documents** (.docx)
    """)
    
    st.markdown("## ⚡ Performance Tips")
    st.markdown("""
    - Smaller files process faster
    - Clear, specific questions work best
    - Wait for processing to complete
    """)
