from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever
from langchain_community.graphs import Neo4jGraph
from qdrant_client import QdrantClient
from qdrant_client.http import models
from neo4j import GraphDatabase
from config import config
import fitz
import pytesseract
from PIL import Image
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import *
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import convert_to_dict
import streamlit as st

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
            trace = start_trace.langfuse.trace(name="table_summarization") if start_trace.langfuse else None
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
            trace = start_trace.langfuse.trace(name="image_summarization") if start_trace.langfuse else None
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
        trace = start_trace("pdf_parsing", metadata={"file": pdf_path})
        try:
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
            
                if trace:
                    trace.event(
                        name="pdf_parsed",
                        metadata={"page_count": len(doc), "elements": len(element_dicts + image_elements)}
                    )
                # Combine structured elements and images
                return element_dicts + image_elements
        except Exception as e:
            if trace:
                trace.event(
                    name="parse_failed",
                    metadata={"error": str(e)},
                    level="ERROR"
                )
            raise
    
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
        trace = start_trace("document_indexing", metadata={"doc_count": len(documents)})
        try:
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
            if trace:
                trace.event(
                    name="indexing_complete",
                    metadata={
                        "collection": collection_name,
                        "vector_count": len(points),
                        "graph_nodes": len(documents)
                    }
                )
            
            return {
                "qdrant_client": qdrant_client,
                "embedder": embedder,
                "bm25_retriever": bm25_retriever,
                "collection_name": collection_name
            }
        except Exception as e:
            if trace:
                trace.event(
                    name="indexing_failed",
                    metadata={"error": str(e)},
                    level="ERROR"
                )
            raise
    
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