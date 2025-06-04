from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from config import config
from utils import *
from typing import List


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
        trace = start_trace("response_generation", input={"question": question})
        try:
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
                if trace:
                    trace.update(
                        output=context,
                        metadata={
                            "context_chunks": len(context),
                        }
                    )
            
            return self.chain.invoke({
                "question": question,
                "context": "\n\n".join(formatted_context)
            })
        except Exception as e:
            if trace:
                trace.update(
                    output={"status": "failed", "error": str(e)},
                    level="ERROR"
                )
            raise
    