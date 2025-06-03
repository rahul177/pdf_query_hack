
#GROQ_API_KEY = "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"


import os
import pandas as pd
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
import streamlit as st
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import convert_to_dict
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.callbacks import CallbackManager
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import CrossEncoder
from neo4j import GraphDatabase
from pymongo import MongoClient
from datetime import datetime, timedelta
import hashlib 
import secrets
import string
import bcrypt
import warnings
import time
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

load_dotenv()

# Configuration
class Config(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_prefix: str = "pdf_chat_"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "0CQAyD1VJx7G4mNNhwAU3IG5TjpdlzVUaokoKHJ_cWk"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    openai_api_key: str = ""
    groq_api_key: str = ""
    upload_folder: str = "uploaded_pdfs"
    chat_model: str = "gpt-4o"  # Default to GPT-4o for most tasks

config = Config(
    langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# MongoDB Configuration
class MongoDBConfig(BaseModel):
    mongo_uri: str = "mongodb://root:example@localhost:27017/?authSource=admin"
    mongo_db: str = "pdf_chat_auth"
    mongo_users_collection: str = "users"
    mongo_sessions_collection: str = "sessions"
    mongo_cache_collection: str = "user_cache"

mongo_config = MongoDBConfig()

# Initialize observability
if config.langfuse_public_key and config.langfuse_secret_key:
    langfuse = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host
    )
else:
    langfuse = None

# Create upload folder if not exists
os.makedirs(config.upload_folder, exist_ok=True)

def get_mongo_client():
    return MongoClient(mongo_config.mongo_uri)

def init_mongo_collections():
    with get_mongo_client() as client:
        db = client[mongo_config.mongo_db]
        # Create collections if they don't exist
        if mongo_config.mongo_users_collection not in db.list_collection_names():
            db.create_collection(mongo_config.mongo_users_collection)
        if mongo_config.mongo_sessions_collection not in db.list_collection_names():
            db.create_collection(mongo_config.mongo_sessions_collection)
        if mongo_config.mongo_cache_collection not in db.list_collection_names():
            db.create_collection(mongo_config.mongo_cache_collection)

        # Create indexes
        db[mongo_config.mongo_users_collection].create_index("username", unique=True)
        db[mongo_config.mongo_sessions_collection].create_index("session_id", unique=True)
        db[mongo_config.mongo_sessions_collection].create_index("expires_at", expireAfterSeconds=0)

init_mongo_collections()

def show_pdf(file_path, page_num, highlight_text=None):
    doc = fitz.open(file_path)
    page = doc.load_page(page_num - 1)

    if highlight_text:
        instances = page.search_for(highlight_text)
        for inst in instances:
            page.add_highlight_annot(inst)

    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    st.image(img, caption=f"Page {page_num}", use_column_width=True)

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

class UserManager:
    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

    @staticmethod
    def create_user(username: str, password: str, email: str) -> Tuple[bool, str]:
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            users = db[mongo_config.mongo_users_collection]
            
            if users.find_one({"username": username}):
                return False, "Username already exists"
            
            hashed_pw = UserManager.hash_password(password)
            user_data = {
                "username": username,
                "password": hashed_pw,
                "email": email,
                "created_at": datetime.utcnow(),
                "last_login": None
            }
            
            try:
                users.insert_one(user_data)
                return True, "User created successfully"
            except Exception as e:
                return False, f"Error creating user: {str(e)}"

    @staticmethod
    def authenticate_user(username: str, password: str) -> Tuple[bool, str]:
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            users = db[mongo_config.mongo_users_collection]
            
            user = users.find_one({"username": username})
            if not user:
                return False, "User not found"
            
            if not UserManager.verify_password(password, user["password"]):
                return False, "Incorrect password"
            
            # Update last login
            users.update_one(
                {"_id": user["_id"]},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            
            return True, "Authentication successful"
        
# Session Management
class SessionManager:
    @staticmethod
    def create_session(username: str, expires_hours: int = 24) -> str:
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            sessions = db[mongo_config.mongo_sessions_collection]
            
            sessions.insert_one({
                "session_id": session_id,
                "username": username,
                "created_at": datetime.utcnow(),
                "expires_at": expires_at
            })
        
        return session_id

    @staticmethod
    def validate_session(session_id: str) -> Tuple[bool, Optional[str]]:
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            sessions = db[mongo_config.mongo_sessions_collection]
            
            session = sessions.find_one({"session_id": session_id})
            if not session:
                return False, None
            
            if session["expires_at"] < datetime.utcnow():
                sessions.delete_one({"session_id": session_id})
                return False, None
            
            return True, session["username"]

    @staticmethod
    def delete_session(session_id: str):
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            sessions = db[mongo_config.mongo_sessions_collection]
            sessions.delete_one({"session_id": session_id})

# Cache Management
class CacheManager:
    @staticmethod
    def get_user_cache(username: str) -> Dict[str, Any]:
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            cache = db[mongo_config.mongo_cache_collection]
            
            user_cache = cache.find_one({"username": username})
            return user_cache["cache_data"] if user_cache else {}

    @staticmethod
    def update_user_cache(username: str, cache_data: Dict[str, Any]):
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            cache = db[mongo_config.mongo_cache_collection]
            
            cache.update_one(
                {"username": username},
                {"$set": {"cache_data": cache_data}},
                upsert=True
            )

    @staticmethod
    def clear_user_cache(username: str):
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            cache = db[mongo_config.mongo_cache_collection]
            cache.delete_one({"username": username})

# ----------------------------
# Indexing Components
# ----------------------------

class ElementSummarizer:
    def __init__(self):
        self.table_llm = ChatGroq(model="llama3-70b-8192", temperature=0, api_key=config.groq_api_key)
        self.image_llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=config.openai_api_key)
        
        # Improved prompts
        self.table_prompt_template = ChatPromptTemplate.from_template(
            """You are an expert at analyzing and summarizing tabular data. 
            Carefully examine the following table content and create a comprehensive summary that:
            
            1. Identifies all key numerical data points and their relationships
            2. Explains the structure and organization of the table
            3. Highlights any trends, comparisons, or important values
            4. Preserves all critical information while being concise
            
            Table Content:
            {table_content}
            
            Summary Guidelines:
            - Begin with the table's purpose or main theme
            - Describe row and column structure
            - Highlight key data points with their exact values
            - Note any important relationships or patterns
            - Keep the summary under 200 words
            
            Table Summary:"""
        )
        
        self.image_prompt_template = ChatPromptTemplate.from_template(
            """You are an expert at analyzing visual content. Examine the following image description and OCR text, 
            then create a detailed summary that:
            
            1. Describes the visual elements and their arrangement
            2. Extracts and interprets any text found in the image
            3. Explains the overall purpose or message of the image
            4. Connects the image to potential document context
            
            Image Context:
            {image_content}
            
            Summary Guidelines:
            - Begin with the image's likely purpose
            - Describe visual elements (charts, diagrams, photos, etc.)
            - Interpret any text or labels
            - Note colors, shapes, and spatial relationships
            - Keep the summary under 150 words
            
            Image Summary:"""
        )
        
        self.table_chain = self.table_prompt_template | self.table_llm | StrOutputParser()
        self.image_chain = self.image_prompt_template | self.image_llm | StrOutputParser()
    
    def summarize(self, element: Dict[str, Any]) -> Dict[str, Any]:
        if element["type"] == "table":
            trace = langfuse.trace(name="table_summarization") if langfuse else None
            generation = trace.generation(
                name="summarize_table",
                input={"element_content": element.get("metadata", {}).get("table_as_html", "")[:1000]},
            ) if trace else None
            
            content = "\n".join(["|".join(row) for row in element["metadata"].get("table_as_html", [])])
            summary = self.table_chain.invoke({"table_content": content})
            
            if generation:
                generation.end(output=summary)
            
            return {
                **element,
                "summary": summary,
                "original_content": content[:5000]
            }
            
        elif element["type"] == "image":
            trace = langfuse.trace(name="image_summarization") if langfuse else None
            generation = trace.generation(
                name="summarize_image",
                input={"element_content": element.get("text", "")[:1000]},
            ) if trace else None

            content = element.get("text", "") + "\n".join(element.get("metadata", {}).get("legend", []))
            summary = self.image_chain.invoke({"image_content": content})
            
            if generation:
                generation.end(output=summary)
            
            return {
                **element,
                "summary": summary,
                "original_content": content[:5000]
            }
        return element

class PDFIndexer:
    def __init__(self):
        self.summarizer = ElementSummarizer()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
    
    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Parse PDF while preserving structure using unstructured.io and PyMuPDF"""
        # Parse structured elements
        elements = partition_pdf(
            pdf_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
            include_metadata=True,
            include_page_breaks=True,
            languages=["eng"],
            extract_images_in_pdf=True,
        )
        element_dicts = convert_to_dict(elements)
        
        # Extract images separately with PyMuPDF for better quality
        doc = fitz.open(pdf_path)
        image_elements = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            images = page.get_images(full=True)
            
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save as temporary image file
                image_path = f"temp_image_{page_num}_{img_index}.png"
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # Use pytesseract for OCR
                try:
                    text = pytesseract.image_to_string(Image.open(image_path))
                except Exception as e:
                    print(f"OCR failed for image: {e}")
                    text = ""
                
                image_elements.append({
                    "type": "image",
                    "page_number": page_num + 1,
                    "image_index": img_index,
                    "image_path": image_path,
                    "text": text,
                    "metadata": {
                        "bbox": img[1:5],
                        "rotation": page.rotation,
                        "dpi": base_image.get("dpi", 72)
                    }
                })
                
                # Clean up temporary image
                try:
                    os.remove(image_path)
                except:
                    pass
        
        # Combine structured elements and images
        return element_dicts + image_elements
    
    def process_elements(self, elements: List[Dict[str, Any]]) -> List[Document]:
        """Convert parsed elements into structured documents with chunking"""
        # Summarize tables and images
        summarized_elements = [self.summarizer.summarize(el) for el in elements]
        
        # Create initial documents with context tracking
        documents = []
        current_context = {
            "page": None,
            "section": None,
            "heading": None,
            "figure": None,
            "table": None
        }
        
        for element in summarized_elements:
            # Update context tracking
            if element['metadata'].get("page_number"):
                current_context["page"] = element['metadata']["page_number"]
            
            if element["type"] == "Title":
                current_context["heading"] = element["text"]
            
            if element["type"] in ["Figure", "Image"]:
                current_context["figure"] = element.get("summary", element.get("text", ""))
            
            if element["type"] == "Table":
                current_context["table"] = element.get("summary", element.get("text", ""))
            
            # Create document with metadata
            metadata = {
                **current_context,
                "element_type": element["type"],
                "source": os.path.basename(st.session_state.get("pdf_path")),
            }
            
            content = ""
            if "summary" in element:
                content = f"[{element['type'].upper()} SUMMARY]: {element['summary']}"
            else:
                content = element.get("text", "")
            
            if content.strip():
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
        
        # Split large text chunks while keeping other elements intact
        split_documents = []
        for doc in documents:
            if doc.metadata["element_type"] == "NarrativeText":
                split_documents.extend(self.text_splitter.split_documents([doc]))
            else:
                split_documents.append(doc)
        
        return split_documents
    
    def index_documents(self, documents: List[Document], collection_name: str) -> Dict[str, Any]:
        """Index documents in Qdrant and Neo4j"""
        # Initialize clients
        qdrant_client = QdrantClient(config.qdrant_url)
        neo4j_driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
        
        # Create Qdrant collection with OpenAI embeddings
        embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=config.openai_api_key)
        embeddings = embedder.embed_documents([doc.page_content for doc in documents])
        
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=len(embeddings[0]),
                distance=models.Distance.COSINE
            )
        )
        
        # Prepare points for Qdrant
        points = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            points.append(models.PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
            ))
        
        # Upload to Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = len(documents)
        
        # Create Neo4j knowledge graph
        self._create_knowledge_graph(neo4j_driver, documents)
        
        return {
            "qdrant_client": qdrant_client,
            "embedder": embedder,
            "bm25_retriever": bm25_retriever,
            "collection_name": collection_name
        }
    
    def _create_knowledge_graph(self, driver: GraphDatabase.driver, documents: List[Document]):
        """Create knowledge graph in Neo4j"""
        with driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create fulltext index
            session.run("""
            CREATE FULLTEXT INDEX chunkIndex IF NOT EXISTS 
            FOR (n:Chunk) ON EACH [n.text]
            """)
            
            # Create document node
            doc_name = os.path.basename(st.session_state.get("pdf_path"))
            session.run(
                "MERGE (d:Document {name: $doc_name})",
                doc_name=doc_name
            )
            
            # Process all chunks
            for i, doc in enumerate(documents):
                # Create chunk node
                session.run("""
                MERGE (c:Chunk {id: $id})
                SET c.text = $text,
                    c.type = $type,
                    c.page = $page
                WITH c
                MATCH (d:Document {name: $doc_name})
                MERGE (d)-[:CONTAINS]->(c)
                """, {
                    "id": f"chunk_{i}",
                    "text": doc.page_content[:500],
                    "type": doc.metadata.get("element_type", "unknown"),
                    "page": doc.metadata.get("page", 0),
                    "doc_name": doc_name
                })
                
                # Create relationships based on metadata
                if doc.metadata.get("heading"):
                    session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (h:Heading {text: $heading})
                    MERGE (c)-[:UNDER_HEADING]->(h)
                    """, {
                        "chunk_id": f"chunk_{i}",
                        "heading": doc.metadata["heading"]
                    })
                
                if doc.metadata.get("figure"):
                    session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (f:Figure {description: $figure})
                    MERGE (c)-[:REFERENCES_FIGURE]->(f)
                    """, {
                        "chunk_id": f"chunk_{i}",
                        "figure": doc.metadata["figure"]
                    })
                
                if doc.metadata.get("table"):
                    session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (t:Table {description: $table})
                    MERGE (c)-[:REFERENCES_TABLE]->(t)
                    """, {
                        "chunk_id": f"chunk_{i}",
                        "table": doc.metadata["table"]
                    })
            
            # Create sequential relationships
            session.run("""
            MATCH (c:Chunk)
            WITH c ORDER BY c.page, c.id
            WITH COLLECT(c) AS chunks
            UNWIND RANGE(0, SIZE(chunks) - 2) AS i
            WITH chunks[i] AS c1, chunks[i + 1] AS c2
            MERGE (c1)-[:NEXT]->(c2)
            """)

# ----------------------------
# Retrieval Components
# ----------------------------

class QueryProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=config.openai_api_key)
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=config.openai_api_key)
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_prompts(self):
        self.query_translation_prompt = ChatPromptTemplate.from_template(
            """As an expert query translator, analyze the user's question and transform it into the most effective search terms.
            Consider:
            1. Technical synonyms and domain-specific terminology
            2. Potential alternative phrasings
            3. Underlying intent behind the question
            4. Key entities and relationships
            
            Original Query: {query}
            
            Guidelines:
            - Preserve the original meaning while optimizing for search
            - Include relevant technical terms
            - Keep it concise (1-2 sentences max)
            
            Translated Query:"""
        )
        
        self.query_expansion_prompt = ChatPromptTemplate.from_template(
            """Generate 3 high-quality query variations that might help retrieve relevant documents.
            Each variation should approach the question from a different perspective while maintaining relevance.
            
            Original Query: {query}
            
            Guidelines for Variations:
            1. Technical Perspective: Use domain-specific terminology
            2. Conceptual Perspective: Focus on underlying concepts
            3. Contextual Perspective: Include related context
            
            Query Variations (as a numbered list):
            1. [Technical variation]
            2. [Conceptual variation]
            3. [Contextual variation]"""
        )
        
        self.query_router_prompt = ChatPromptTemplate.from_template(
            """Analyze the query and determine the optimal search strategy weights based on:
            - Query type (factual, conceptual, relational)
            - Expected answer format
            - Likely document structure
            
            Respond with JSON format containing:
            - query_type classification
            - weights for keyword, vector, and knowledge_graph approaches
            
            Query: {query}
            
            Example Response:
            {{
                "query_type": "factual",
                "weights": {{
                    "keyword": 0.7,
                    "vector": 0.2,
                    "knowledge_graph": 0.1
                }}
            }}
            
            Your Analysis:"""
        )
        
        self.rerank_prompt = ChatPromptTemplate.from_template(
            """Re-rank these documents based on their relevance to the query.
            Consider both semantic relevance and factual accuracy.
            
            Query: {query}
            
            Documents:
            {documents}
            
            Return the re-ranked document IDs in order of relevance, most relevant first.
            Only return the IDs as a comma-separated list."""
        )
    
    def process_query(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """Process query through translation, expansion, and routing"""
        # Translate query
        translated = self._translate_query(query)
        
        # Expand query
        expanded = self._expand_query(translated)
        
        # Route each query variation
        queries_with_weights = []
        for q in expanded:
            routing = self._route_query(q)
            queries_with_weights.append({
                "query": q,
                "weights": routing["weights"],
                "query_type": routing["query_type"]
            })
        
        return queries_with_weights, translated
    
    def retrieve_documents(
        self,
        queries_with_weights: List[Dict[str, Any]],
        collection_name: str,
        top_k: int = 10
    ) -> List[Document]:
        """Perform hybrid retrieval across all search methods"""
        qdrant_client = QdrantClient(config.qdrant_url)
        neo4j_driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
        
        all_results = []
        
        for query_info in queries_with_weights:
            query = query_info["query"]
            weights = query_info["weights"]
            
            # Keyword search
            if weights["keyword"] > 0:
                keyword_results = self._keyword_search(
                    qdrant_client, query, collection_name,
                    int(top_k * weights["keyword"])
                )
                all_results.extend(keyword_results)
            
            # Vector search
            if weights["vector"] > 0:
                vector_results = self._vector_search(
                    qdrant_client, query, collection_name,
                    int(top_k * weights["vector"])
                )
                all_results.extend(vector_results)
            
            # Knowledge graph search
            if weights["knowledge_graph"] > 0:
                kg_results = self._kg_search(
                    neo4j_driver, query,
                    int(top_k * weights["knowledge_graph"])
                )
                all_results.extend(kg_results)
        
        # Remove duplicates and rerank
        unique_results = self._deduplicate_and_rerank(queries_with_weights[0]["query"], all_results, top_k)
        
        return unique_results, queries_with_weights
    
    def _translate_query(self, query: str) -> str:
        chain = self.query_translation_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def _expand_query(self, query: str) -> List[str]:
        chain = self.query_expansion_prompt | self.llm | StrOutputParser()
        expansions = chain.invoke({"query": query})
        # Parse the numbered list response
        variations = [line.split(". ", 1)[1] for line in expansions.split("\n") if ". " in line]
        return [query] + variations[:3]  # Return original + up to 3 variations
    
    def _route_query(self, query: str) -> Dict[str, float]:
        chain = self.query_router_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query})
        try:
            import json
            return json.loads(result)
        except:
            # Default weights if parsing fails
            return {
                "query_type": "conceptual",
                "weights": {
                    "keyword": 0.3,
                    "vector": 0.5,
                    "knowledge_graph": 0.2
                }
            }
    
    def _keyword_search(self, client: QdrantClient, query: str, collection_name: str, k: int) -> List[Document]:
        try:
            results = client.search(
                collection_name=collection_name,
                query_vector=[0]*1536,  # Dummy vector for text search
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="text",
                            match=models.MatchText(text=query))
                    ]
                ),
                limit=k,
                with_payload=True
            )
            return [
                Document(
                    page_content=hit.payload["text"],
                    metadata={
                        **hit.payload["metadata"],
                        "score": 1.0 - (i/len(results)),  # Approximate score based on position
                        "retriever": "BM25"
                    }
                )
                for i, hit in enumerate(results)
            ]
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []
        
    def _vector_search(self, client: QdrantClient, query: str, collection_name: str, k: int) -> List[Document]:
        query_embedding = self.embedder.embed_query(query)
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(
                        value=os.path.basename(st.session_state.get("pdf_path")))
                )]
            ),
            limit=k,
            with_payload=True
        )
        return [
            Document(
                page_content=hit.payload["text"],
                metadata={
                    **hit.payload["metadata"],
                    "score": hit.score,
                    "retriever": "Vector"
                }
            )
            for hit in results
        ]
    
    def _kg_search(self, driver: GraphDatabase.driver, query: str, k: int) -> List[Document]:
        with driver.session() as session:
            results = session.run("""
            CALL db.index.fulltext.queryNodes("chunkIndex", $query) 
            YIELD node, score
            MATCH (node)-[:REFERENCES*0..1]->(related)
            RETURN node, related, score
            ORDER BY score DESC
            LIMIT $k
            """, {"query": query, "k": k})
            
            documents = []
            for record in results:
                labels = list(record["node"].labels)
                node_type = labels[0] if labels else "unknown"
                
                documents.append(Document(
                    page_content=record["node"]["text"],
                    metadata={
                        "source": "knowledge_graph",
                        "type": node_type,
                        "score": float(record["score"]),
                        "retriever": "Knowledge Graph",
                        "page": record["node"].get("page", "N/A")
                    }
                ))
            
            return documents
    
    def _deduplicate_and_rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int
    ) -> List[Document]:
        # Remove duplicates by content
        unique_docs = {}
        for doc in documents:
            content_key = doc.page_content[:200]  # First 200 chars as key
            if content_key not in unique_docs or \
               (doc.metadata.get("score", 0) > unique_docs[content_key].metadata.get("score", 0)):
                unique_docs[content_key] = doc
        
        # Rerank with LLM
        if not unique_docs:
            return []
        
        # Format documents for reranking
        doc_list = list(unique_docs.values())
        doc_texts = "\n".join([
            f"ID: {i}\nContent: {doc.page_content[:500]}\nScore: {doc.metadata.get('score', 0):.3f}\n"
            for i, doc in enumerate(doc_list)
        ])
        
        # Get reranked order from LLM
        rerank_chain = self.rerank_prompt | self.llm | StrOutputParser()
        reranked_order = rerank_chain.invoke({
            "query": query,
            "documents": doc_texts
        })
        
        try:
            # Parse the reranked order
            ranked_ids = [int(id.strip()) for id in reranked_order.split(",")]
            # Return documents in the new order
            return [doc_list[i] for i in ranked_ids[:top_k]]
        except:
            # Fallback to original order if parsing fails
            return doc_list[:top_k]

# ----------------------------
# Response Generation
# ----------------------------

class ResponseGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=config.openai_api_key)
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert document analyst. Provide a comprehensive answer to the question using the provided context.
            
            **Response Guidelines:**
            1. Start with a clear, concise answer in bold
            2. Include page/section references for all key points
            3. Highlight numerical data and important findings
            4. Use bullet points for complex information
            5. If uncertain, say "The document suggests..." instead of asserting
            
            **Question:** {question}
            
            **Document Context:**
            {context}
            
            **Answer Structure:**
            - 🎯 **Direct Answer**: [Clear answer]
            - 📚 **Supporting Evidence**: 
              • [Point 1 with reference]
              • [Point 2 with reference]
            - 🔍 **Additional Insights**: [Optional related information]
            
            **Answer:**"""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def generate_response(self, question: str, context: List[Document]) -> str:
        formatted_context = []
        for doc in context:
            source_info = []
            if "page" in doc.metadata:
                source_info.append(f"page {doc.metadata['page']}")
            if "heading" in doc.metadata:
                source_info.append(f"under '{doc.metadata['heading']}'")
            if "figure" in doc.metadata:
                source_info.append(f"referencing figure: {doc.metadata['figure']}")
            if "table" in doc.metadata:
                source_info.append(f"referencing table: {doc.metadata['table']}")
            
            source = " | ".join(source_info)
            formatted_context.append(f"CONTENT: {doc.page_content}\nSOURCE: {source}\n")
        
        return self.chain.invoke({
            "question": question,
            "context": "\n\n".join(formatted_context)
        })
    
class RetrieverEvaluator:
    def __init__(self):
        self.metrics = {
            "precision": None,
            "recall": None,
            "f1": None
        }
        self.sample_data = None
    
    def load_sample_data(self, file_path: str = None):
        """Load sample Q&A pairs for evaluation from CSV or use default"""
        if file_path and os.path.exists(file_path):
            # Load from CSV if provided
            self.sample_data = pd.read_csv(file_path)
        else:
            # Default sample data
            self.sample_data = pd.DataFrame({
                "questions": [
                    "What is the capital of France?",
                    "Who wrote 'Romeo and Juliet'?",
                    "What is the chemical symbol for gold?"
                ],
                "answers": [
                    "Paris",
                    "William Shakespeare",
                    "Au"
                ],
                "contexts": [
                    "France is a country in Europe. Its capital is Paris.",
                    "William Shakespeare was an English playwright who wrote 'Romeo and Juliet'.",
                    "Gold is a chemical element with the symbol Au and atomic number 79."
                ]
            })
        return self.sample_data
    
    def evaluate_retriever(self, file_path: str = None):
        """Evaluate retriever performance on sample data"""
        df = self.load_sample_data(file_path)
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Simple string matching evaluation (replace with actual retriever if needed)
        for _, row in df.iterrows():
            # Simulate retrieval by checking if answer is in context
            correct_answer_found = row["answers"].lower() in row["contexts"].lower()
            
            if correct_answer_found:
                true_positives += 1
            else:
                false_negatives += 1
            
            # Count false positives (simplified)
            false_positives += 0  # Adjust based on your needs
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.metrics = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
        
        return self.metrics

# Main Application (rest of your existing PDF analyzer code)
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
    
    # Rest of your existing PDF analyzer code
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
            file_path = os.path.join(config.upload_folder, uploaded_file.name)
            
            # Show upload progress
            progress_bar = st.progress(0)
            with open(file_path, "wb") as f:
                for chunk in uploaded_file:
                    f.write(chunk)
                    progress_bar.progress(min(100, f.tell() * 100 // uploaded_file.size))
            
            st.session_state.pdf_path = file_path
            progress_bar.empty()

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
            with st.spinner("Analyzing your question..."):
                try:
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
                
                    # Engaging message based on query complexity
                    num_variations = len(query_info)
                    if num_variations > 3:
                        st.success("🔍 Found multiple perspectives to analyze your question!")
                    else:
                        st.info("💡 Analyzing your question from different angles...")
                    
                    # Translated query with explanation
                    st.markdown(f"""
                    **Translated Query:**  
                    *"{translated_query}"*  
                    *We refined your question to better match the document content*
                    """)
                    
                    # Query variations table
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
                                if st.button(f"📄 Page {page_num}", key=f"page_button_{i}"):
                                    st.session_state.current_page = page_num
                                    st.session_state.show_pdf_viewer = True
                                    st.rerun()
                            with col2:
                                st.markdown(f"**{doc.metadata.get('retriever', 'Retriever')}** — Score: `{doc.metadata.get('score', 0):.3f}`")
                                st.markdown(doc.page_content[:300] + "...")
                except Exception as e:
                    st.error(f"Error during question analysis: {e}")

        # Display PDF viewer without reprocessing
        if st.session_state.show_pdf_viewer:
            st.subheader("📄 PDF Viewer")
            current_page = st.session_state.current_page
            highlight_text = st.session_state.highlight_map.get(current_page, None)
            
            # Immediately show the page without reprocessing
            show_pdf(st.session_state.pdf_path, current_page, highlight_text=highlight_text)
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"**Page {current_page}**")
                if st.button("Back to Results"):
                    st.session_state.show_pdf_viewer = False
                    st.rerun()

    with tab2:
        st.title("📊 Retriever Accuracy Check")
        st.markdown("Evaluate the retriever's performance using sample questions")
        
        if "evaluator" not in st.session_state:
            st.session_state.evaluator = RetrieverEvaluator()
        
        # Add file uploader for CSV
        eval_file = st.file_uploader("Upload evaluation data (CSV)", type=["csv"])
        
        if st.button("Run Evaluation"):
            with st.spinner("Running evaluation..."):
                # Use temporary file if uploaded, otherwise use default data
                file_path = None
                if eval_file:
                    file_path = os.path.join(config.upload_folder, "temp_eval.csv")
                    with open(file_path, "wb") as f:
                        f.write(eval_file.getbuffer())
                
                metrics = st.session_state.evaluator.evaluate_retriever(file_path)
                
                st.subheader("Evaluation Metrics")
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precision", metrics["precision"])
                with col2:
                    st.metric("Recall", metrics["recall"])
                with col3:
                    st.metric("F1 Score", metrics["f1"])
                
                # Display confusion matrix
                st.subheader("Confusion Matrix")
                confusion_data = {
                    "Actual Positive": [metrics["true_positives"], metrics["false_negatives"]],
                    "Actual Negative": [metrics["false_positives"], 0]
                }
                st.table(pd.DataFrame(
                    confusion_data,
                    index=["Predicted Positive", "Predicted Negative"]
                ))
                
                # Display sample questions
                st.subheader("Sample Questions Used")
                st.dataframe(st.session_state.evaluator.sample_data)
                
                # Clean up temporary file
                if eval_file and os.path.exists(file_path):
                    os.remove(file_path)

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