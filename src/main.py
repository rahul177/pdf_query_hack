import streamlit as st
from auth import UserManager, SessionManager, CacheManager
from indexing import PDFIndexer
from retrieval import QueryProcessor
from response import ResponseGenerator
from evaluation import ModelEvaluator
import os
from utils import *
from pymongo import MongoClient
from config import mongo_config
import fitz
from PIL import Image
import uuid
import pandas as pd
import re
from langchain_core.documents import Document
from sklearn.metrics import precision_score, recall_score, f1_score

if config.langfuse_public_key and config.langfuse_secret_key:
    langfuse = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host
    )
else:
    langfuse = None

os.makedirs(config.upload_folder, exist_ok=True)

def get_mongo_client():
    return MongoClient(mongo_config.mongo_uri)

# Authentication Pages
def show_login_page():
    st.title("PDF Analyzer - Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            authenticated, message = UserManager.authenticate_user(username, password)
            if authenticated:
                session_id = SessionManager.create_session(username)
                st.session_state.session_id = session_id
                st.session_state.username = username
                st.rerun()
            else:
                st.error(message)
    
    if st.button("Don't have an account? Register here"):
        st.session_state.show_register = True
        st.rerun()

def show_register_page():
    st.title("PDF Analyzer - Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = UserManager.create_user(username, password, email)
                if success:
                    st.success(message)
                    st.session_state.show_register = False
                    st.rerun()
                else:
                    st.error(message)
    
    if st.button("Already have an account? Login here"):
        st.session_state.show_register = False
        st.rerun()

def show_settings_page():
    st.title("User Settings")
    
    # Cache Management
    st.subheader("Cache Management")
    if st.button("Clear My Cache"):
        CacheManager.clear_user_cache(st.session_state.username)
        st.success("Cache cleared successfully!")
    
    # Session Management
    st.subheader("Session Management")
    if st.button("Logout from all devices"):
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            sessions = db[mongo_config.mongo_sessions_collection]
            sessions.delete_many({"username": st.session_state.username})
        st.success("Logged out from all devices")



def main_application():
    st.set_page_config(page_title="PDF Analyzer", page_icon="📄", layout="wide")
    
    # Add a logout button to the sidebar
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.username}**")
        if st.button("Logout"):
            SessionManager.delete_session(st.session_state.session_id)
            del st.session_state.session_id
            del st.session_state.username
            st.rerun()
           
        
        if st.button("Settings"):
            st.session_state.show_settings = True
            st.rerun()
            
    if st.session_state.get("show_settings", False):
        show_settings_page()
        if st.button("Back to Application"):
            st.session_state.show_settings = False
            st.rerun()
        return
    
    # PDF Viewer Function - Add this new function inside main_application
    def show_pdf_viewer():
        """PDF viewer that doesn't trigger full reruns"""
        st.subheader("📄 PDF Viewer")
        current_page = st.session_state.current_page
        highlight_text = st.session_state.highlight_map.get(current_page, None)
        
        # Use cached PDF document
        doc = fitz.open(st.session_state.pdf_path)
        page = doc.load_page(current_page - 1)
        
        if highlight_text:
            instances = page.search_for(highlight_text)
            for inst in instances:
                page.add_highlight_annot(inst)
        
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        st.image(img, caption=f"Page {current_page}", use_column_width=True)
        
        # Navigation controls
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            if st.button("⏮ First"):
                st.session_state.current_page = 1
        with col2:
            if st.button("◀ Previous") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
        with col3:
            if st.button("Next ▶") and st.session_state.current_page < len(doc):
                st.session_state.current_page += 1
        with col4:
            if st.button("Last ⏭"):
                st.session_state.current_page = len(doc)
        
        st.markdown(f"**Page {current_page} of {len(doc)}**")
        
        if st.button("Back to Results"):
            st.session_state.show_pdf_viewer = False

    # Page change callback - Add this new function inside main_application
    def change_page(page_num):
        """Callback function for page navigation"""
        st.session_state.current_page = page_num
        st.session_state.show_pdf_viewer = True

    # Check if we should show PDF viewer first
    if st.session_state.get("show_pdf_viewer", False):
        show_pdf_viewer()
        st.stop()  # This prevents the rest of the app from running

    # Rest of your existing application code
    tab1, tab2 = st.tabs(["📄 PDF Analyzer", "📊 Retriever Accuracy Check"])
    
    with tab1:
        st.title("📄 PDF Analyzer")
        st.markdown("Upload a PDF document and ask questions about its content")

        # Initialize session state with user-specific cache
        if "pdf_processed" not in st.session_state:
            st.session_state.pdf_processed = False
        if "query_processor" not in st.session_state:
            st.session_state.query_processor = QueryProcessor()
        if "response_gen" not in st.session_state:
            st.session_state.response_gen = ResponseGenerator()
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        if "show_pdf_viewer" not in st.session_state:
            st.session_state.show_pdf_viewer = False
        if "highlight_map" not in st.session_state:
            st.session_state.highlight_map = {}

        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file and not st.session_state.pdf_processed:
            trace = start_trace("pdf_upload", user_id=st.session_state.username)
            try:
                if trace:
                    trace.update(input={"filename": uploaded_file.name})
                file_path = os.path.join(config.upload_folder, uploaded_file.name)
                
                progress_bar = st.progress(0)
                with open(file_path, "wb") as f:
                    for chunk in uploaded_file:
                        f.write(chunk)
                        progress_bar.progress(min(100, f.tell() * 100 // uploaded_file.size))
                
                st.session_state.pdf_path = file_path
                progress_bar.empty()
                if trace:
                    trace.update(
                        output={"status": "processed"},
                        metadata={"file": uploaded_file.name}
                    )
            except Exception as e:
                if trace:
                    trace.update(
                        output={"status": "failed", "error": str(e)},
                        level="ERROR"
                    )
                raise

            with st.spinner("Processing PDF..."):
                try:
                    indexer = PDFIndexer()
                    elements = indexer.parse_pdf(file_path)
                    documents = indexer.process_elements(elements)
                    collection_name = f"{config.qdrant_collection_prefix}{str(uuid.uuid4())[:8]}"
                    index_result = indexer.index_documents(documents, collection_name)

                    st.session_state.documents = documents
                    st.session_state.collection_name = collection_name
                    st.session_state.qdrant_client = index_result["qdrant_client"]
                    st.session_state.embedder = index_result["embedder"]
                    st.session_state.bm25_retriever = index_result["bm25_retriever"]
                    st.session_state.pdf_processed = True
                    st.success("PDF processed successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error during PDF processing: {e}")
                    return

        # Question box
        question = st.text_input("Ask a question about the document:")
        if question:
            trace = start_trace("question_answered", 
                          user_id=st.session_state.username,
                          metadata={"pdf": st.session_state.get("pdf_path", "unknown")})
            try:
                with st.spinner("Analyzing your question..."):
                    try:
                        if trace:
                            trace.update(input={"question": question})
                        queries_with_weights, translated_query = st.session_state.query_processor.process_query(question)
                        documents, query_info = st.session_state.query_processor.retrieve_documents(
                            queries_with_weights=queries_with_weights,
                            collection_name=st.session_state.collection_name,
                            top_k=10
                        )
                        response = st.session_state.response_gen.generate_response(question, documents)

                        st.subheader("Answer")
                        st.markdown(response)

                        st.subheader("Query Analysis")
                        num_variations = len(query_info)
                        if num_variations > 3:
                            st.success("🔍 Found multiple perspectives to analyze your question!")
                        else:
                            st.info("💡 Analyzing your question from different angles...")
                        
                        st.markdown(f"""
                        **Translated Query:**  
                        *"{translated_query}"*  
                        *We refined your question to better match the document content*
                        """)
                        
                        st.markdown("**Query Variations and Weights:**")
                        query_data = []
                        for i, q in enumerate(query_info):
                            query_data.append([
                                q["query"],
                                q["query_type"],
                                f"{q['weights']['keyword']:.2f}",
                                f"{q['weights']['vector']:.2f}",
                                f"{q['weights']['knowledge_graph']:.2f}"
                            ])
                        
                        st.table(pd.DataFrame(
                            query_data,
                            columns=["Query", "Type", "Keyword Weight", "Vector Weight", "KG Weight"]
                        ))

                        # Create highlight map
                        highlight_map = {}
                        for doc in documents:
                            page = doc.metadata.get("page")
                            if page and page not in highlight_map:
                                highlight_map[page] = doc.page_content[:150]
                        st.session_state.highlight_map = highlight_map

                        st.subheader("Top Results — Click to View Page")

                        for i, doc in enumerate(documents[:3]):
                            page_num = doc.metadata.get("page", "N/A")
                            if page_num != "N/A":
                                col1, col2 = st.columns([1, 9])
                                with col1:
                                    st.button(
                                        f"📄 Page {page_num}",
                                        key=f"page_btn_{i}",
                                        on_click=change_page,
                                        args=(page_num,)
                                    )
                                with col2:
                                    st.markdown(f"**{doc.metadata.get('retriever', 'Retriever')}** — Score: `{doc.metadata.get('score', 0):.3f}`")
                                    st.markdown(doc.page_content[:300] + "...")
                    except Exception as e:
                        st.error(f"Error during question analysis: {e}")
                if trace:
                    trace.update(
                            output=documents,
                            metadata={"result_count": len(documents)}
                    )    
            except Exception as e:
                if trace:
                    trace.update(
                        output={"status": "failed", "error": str(e)},
                        level="ERROR"
                    )
                raise         

    with tab2:
        st.title("📊 Model Evaluation")
        st.markdown("Evaluate the model's performance using sample questions")
        
        if "evaluator" not in st.session_state:
            st.session_state.evaluator = ModelEvaluator()
        
        if st.button("Run Comprehensive Evaluation"):
            with st.spinner("Running evaluation (this may take a minute)..."):
                # Ensure we have the required components
                if "query_processor" not in st.session_state:
                    st.session_state.query_processor = QueryProcessor()
                if "response_gen" not in st.session_state:
                    st.session_state.response_gen = ResponseGenerator()
                
                # Run evaluation
                metrics = st.session_state.evaluator.evaluate_model(
                    st.session_state.query_processor,
                    st.session_state.response_gen
                )
                
                # Display results
                st.subheader("Evaluation Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Precision", metrics["precision"])
                with col2:
                    st.metric("Recall", metrics["recall"])
                with col3:
                    st.metric("F1 Score", metrics["f1"])
                with col4:
                    st.metric("MRR", metrics["mrr"])
                
                st.subheader("Detailed Results")
                
                for i, result in enumerate(metrics["results"]):
                    with st.expander(f"Question {i+1}: {result['question']}"):
                        st.markdown(f"**Correct Answer:** {result['correct_answer']}")
                        st.markdown(f"**Model's Direct Answer:** {result['generated_answers'][0]}")
                        st.markdown("**Full Response:**")
                        st.markdown(result["full_response"])
                        
                        # Check if answer was correct
                        correct_normalized = st.session_state.evaluator._normalize_text(result["correct_answer"])
                        found = any(
                            correct_normalized in st.session_state.evaluator._normalize_text(ans) or 
                            st.session_state.evaluator._normalize_text(ans) in correct_normalized
                            for ans in result["generated_answers"]
                        )
                        
                        if found:
                            st.success("✅ Correct answer found in response")
                        else:
                            st.error("❌ Correct answer not found in response")
                
                st.subheader("Sample Contexts Used")
                st.json([{"page": item["page"], "context": item["context"]} for item in st.session_state.evaluator.sample_data])

# Main function to control page flow
def main():
    # Check for existing session
    if "session_id" in st.session_state:
        valid, username = SessionManager.validate_session(st.session_state.session_id)
        if valid:
            st.session_state.username = username
            main_application()
        else:
            del st.session_state.session_id
            st.session_state.show_register = False
            show_login_page()
    else:
        if st.session_state.get("show_register", False):
            show_register_page()
        else:
            show_login_page()

if __name__ == "__main__":
    main()