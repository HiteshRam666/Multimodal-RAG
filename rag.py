from langchain_core.documents import Document
import fitz 
from transformers import CLIPProcessor, CLIPModel 
from PIL import Image 
import numpy as np
from langchain.chat_models import init_chat_model 
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.metrics.pairwise import cosine_similarity
import base64 
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
from time import time

import os 
import torch    
from dotenv import load_dotenv 

# Logging 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class MultiModalRAG:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", llm_model: str = "gpt-4o-mini", chunk_size: int = 500, chunk_overlap: int = 100):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        # Initializing model 
        logger.info(f"Loading CLIP model: {model_name}")
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name) 
        self.clip_model.eval()

        # Initialize LLM
        self.llm = init_chat_model(model=llm_model)

        # Text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size, 
            chunk_overlap = chunk_overlap 
        )

        # Storage
        self.all_docs: List[Document] = []
        self.all_embeddings: List[np.ndarray] = []
        self.image_data_store: Dict[str, str] = {}
        self.vector_store: Optional[FAISS] = None

    def image_embed(self, image_path) -> np.ndarray:
        """Embed images using CLIP with better error handling"""
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path 
            
            inputs = self.clip_processor(images=image, return_tensors="pt") 
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs) 
                features = features / features.norm(dim = -1, keepdim=True)
                return features.squeeze().numpy()  
        except Exception as e:
            logger.error(f"Error embedding image: {str(e)}")
            return None

    def text_embed(self, text: str) -> np.ndarray:
        """Embed text using CLIP with better preprocessing"""
        try:
            # Clean and truncate text
            text = text.strip() 
            if len(text) > 1000:
                text = text[:1000] 

            inputs = self.clip_processor(
                text=text, 
                return_tensors="pt", 
                padding = True,
                truncation = True, 
                max_length = 77 
            ) 
            with torch.no_grad():
                features = self.clip_model.get_text_features(**inputs) 
                features = features / features.norm(dim = -1, keepdim=True) 
                return features.squeeze().numpy() 
        except Exception as e:
            logger.error(f"Error encoding text: {e}") 
            return None

    def process_pdf(self, pdf_path: str, document_id: str):
        """Process PDF with improved error handling and logging, and return a DocumentInfo-like dict"""
        start_time = time()
        logger.info(f"Processing PDF: {pdf_path}")

        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}") 

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        text_chunks = 0
        image_count = 0

        try:
            for i, page in enumerate(doc):
                logger.info(f"Processing page {i+1}/{total_pages}")

                # Process text
                text = page.get_text()
                if text.strip():
                    chunks_before = len(self.all_docs)
                    self._process_page_text(text, i) 
                    text_chunks += len(self.all_docs) - chunks_before 
                
                # Process images
                docs_before = len(self.all_docs) 
                self._process_page_images(page, doc, i)
                image_count += len(self.all_docs) - docs_before
        finally:
            doc.close()
        
        # Create vector store
        if self.all_embeddings:
            self._create_vector_store()
            logger.info(f"Created vector store with {len(self.all_docs)} documents")
        else:
            logger.warning("No embeddings created - check your PDF content")
        
        processing_time = time() - start_time

        return {
            "filename": Path(pdf_path).name,
            "pages": total_pages,
            "text_chunks": text_chunks,
            "images": image_count,
            "processing_time": round(processing_time, 2),
            "document_id": document_id
        }

    def _process_page_text(self, text: str, page_num: int) -> None:
        """Process text from a page"""
        temp_doc = Document(page_content=text, metadata = {'page': page_num, 'type': 'text'})
        text_chunks = self.text_splitter.split_documents([temp_doc])

        for chunk_idx, chunk in enumerate(text_chunks):
            embedding = self.text_embed(chunk.page_content) 
            if embedding is not None:
                # Add chunk index to metadata
                chunk.metadata["chunk_id"] = f"page_{page_num}_chunk_{chunk_idx}"
                self.all_embeddings.append(embedding) 
                self.all_docs.append(chunk)
    
    def _process_page_images(self, page, doc, page_num: int) -> None:
        """Process images from a page"""
        for img_index, img in enumerate(page.get_images(full = True)):
            try:
                xref = img[0] 
                base_image = doc.extract_image(xref=xref) 
                image_bytes = base_image.get("image") 
                if image_bytes is None:
                    continue

                # Skip very small images (likely artifacts)
                if len(image_bytes) < 1000:
                    continue

                # Convert to PIL Image 
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Skip very small images by dimensions
                if pil_image.size[0] < 50 or pil_image.size[1] < 50:
                    continue

                image_id = f"page_{page_num}_img_{img_index}"

                # Store image as base64
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG") 
                img_base64 = base64.b64encode(buffered.getvalue()).decode() 
                self.image_data_store[image_id] = img_base64 

                # Embed Image 
                embedding = self.image_embed(pil_image) 
                if embedding is not None:
                    self.all_embeddings.append(embedding) 

                    image_doc = Document(
                        page_content=f"[Image: {image_id}]", 
                        metadata={
                            "page": page_num, 
                            "type": "image", 
                            "image_id": image_id,
                            "image_size": pil_image.size
                        }
                    )
                    self.all_docs.append(image_doc)
            except Exception as e:
                logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                continue
    
    def _create_vector_store(self) -> None:
        """Create FAISS vector store"""
        try:
            # Pair up texts with their embeddings
            text_embeddings = [
                (doc.page_content, emb.tolist()) 
                for doc, emb in zip(self.all_docs, self.all_embeddings)
            ]

            # Create FAISS index
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=self.text_embed,  # pass the embedding function
                metadatas=[doc.metadata for doc in self.all_docs]
            )
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def retrieve_multimodal(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve using CLIP embeddings with query expansion"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Process a PDF first.")
        
        # Embed query 
        query_embedding = self.text_embed(query)
        if query_embedding is None:
            return []

        # Search in vector store 
        results = self.vector_store.similarity_search_by_vector(
            query_embedding, 
            k = k 
        )

        return results 
    
    def create_multimodal_message(self, query: str, retrieved_docs: List[Document]) -> HumanMessage:
        """Create enhanced multimodal message"""
        content = []

        # System instruction
        content.append(
            {
                'type': 'text', 
                'text': f"You are a helpful AI assistant analyzing documents. Answer the following question based on the provided context.\n\nQuestion: {query}\n\nContext:\n"  
            }
        )

        # Separate documents by type
        text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
        image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]

        # Add text context with better formatting
        if text_docs:
            text_context = "\n\n".join([
                f"[Page {doc.metadata['page']}: {doc.page_content}]" for doc in text_docs
            ])

            content.append({
                'type': 'text', 
                'text': f"Text content:\n{text_context}\n"
            })

        # Add images with context
        for doc in image_docs:
            image_id = doc.metadata.get('image_id')
            if image_id and image_id in self.image_data_store:
                content.append({
                    "type": "text",
                    "text": f"\nRelevant Image from Page {doc.metadata['page']}:"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self.image_data_store[image_id]}"
                    }
                })

        # Prompt instruction
        content.append({
            'type': 'text', 
            'text': "\n\nInstructions:\n1. Analyze both the text content and any images provided\n2. Answer the question comprehensively\n3. If images contain relevant information, describe what you see\n4. Cite which page(s) your answer comes from\n5. If the context doesn't contain enough information, say so clearly"
        })

        return HumanMessage(content = content)
    
    def query(self, question: str, k: int = 3, verbose: bool = False) -> str:
        """Main query method with optional verbose output"""
        try:
            # Retrieve relevant documents
            context_docs = self.retrieve_multimodal(query=question, k=k) 

            if verbose:
                print(f"\nRetrieved {len(context_docs)} documents:")
                for doc in context_docs:
                    doc_type = doc.metadata.get("type", "unknown")
                    page = doc.metadata.get("page", "?")
                    if doc_type == "text":
                        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        print(f"  - Text from page {page}: {preview}")
                    else:
                        print(f"  - Image from page {page}")

            # Create multimodal message
            message = self.create_multimodal_message(question, context_docs) 

            # Get response from LLM
            response = self.llm.invoke([message]) 

            return response.content
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return f"Sorry, I encountered an error while processing your query: {str(e)}"

if __name__ == "__main__":
    rag_system = MultiModalRAG() 

    pdf_path = r"C:\\Users\\hites\\OneDrive\\Desktop\\Multi-modal RAG\\data\\BERT_Slides.pdf"
    doc_info = rag_system.process_pdf(pdf_path, document_id="test_doc")
    print(f"Processed PDF: {doc_info}")

    queries = [
        "How to inference a language model?",
        "What is the architecture of BERT?",
        "Show me any diagrams about attention mechanisms",
        "What are the key components shown in the images?"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('-'*60)
        answer = rag_system.query(query, k=3, verbose=True)
        print(f"\nAnswer: {answer}")