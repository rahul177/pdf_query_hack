from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from config import config
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from utils import *
from qdrant_client.http import models
import os
import streamlit as st

if config.langfuse_public_key and config.langfuse_secret_key:
    langfuse = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host
    )
else:
    langfuse = None

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
        trace = start_trace("query_processing", input={"query": query})
        try:
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
            if trace:
                trace.event(
                    name="query_processed",
                    metadata={
                        "variations": len(queries_with_weights),
                        "translated_query": translated
                    }
                )
            return queries_with_weights, translated
        except Exception as e:
            if trace:
                trace.event(
                    name="query_processing_failed",
                    metadata={"error": str(e)},
                    level="ERROR"
                )
            raise

    def retrieve_documents(
        self,
        queries_with_weights: List[Dict[str, Any]],
        collection_name: str,
        top_k: int = 10
    ) -> List[Document]:
        """Perform hybrid retrieval across all search methods"""
        trace = start_trace("document_retrieval", metadata={
            "collection": collection_name,
            "top_k": top_k
        })
        try:
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
            if trace:
                trace.event(
                    name="retrieval_complete",
                    metadata={
                        "unique_results": len(unique_results),
                        "retrieval_methods": list(set(d.metadata.get("retriever") for d in unique_results))
                    }
                )
            return unique_results, queries_with_weights
            
        except Exception as e:
            if trace:
                trace.event(
                    name="retrieval_failed",
                    metadata={"error": str(e)},
                    level="ERROR"
                )
            raise
    
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