from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Summarize:
    """
    A comprehensive document summarization class that supports multiple file formats
    and provides medical-focused text summarization capabilities.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loaders = {
            "txt": TextLoader,
            "pdf": PyPDFLoader,
            "csv": CSVLoader
        }
        self.summary = ""
        self.summarizer = None
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Initializing summarizer for file: {os.path.basename(file_path)}")
        
    def getLoader(self) -> List[Any]:
        """Load document content based on file extension."""
        file_extension = os.path.splitext(self.file_path)[1][1:].lower()
        
        if file_extension not in self.loaders:
            supported_formats = ", ".join(self.loaders.keys())
            raise ValueError(f"Unsupported file extension: {file_extension}. Supported formats: {supported_formats}")
            
        logger.info(f"Loading document with {file_extension.upper()} loader")
        loader = self.loaders[file_extension]
        load = loader(self.file_path)
        data = load.load()
        
        logger.info(f"Loaded {len(data)} document chunks")
        return data
        
    def callRag(self) -> str:
        """Main method to generate document summary using RAG approach."""
        try:
            logger.info("Starting document summarization process")
            data = self.getLoader()
            splits = self.splitText(data)
            self.sum_splits(splits)
            
            # Clean up summary
            self.summary = self.summary.strip()
            if not self.summary:
                self.summary = "Unable to generate summary from the provided document."
                
            logger.info(f"Summary generated successfully ({len(self.summary)} characters)")
            return self.summary
            
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return f"Error generating summary: {str(e)}"
        
    def sum_splits(self, splits: List[Any]) -> None:
        """Generate summaries for document splits."""
        if not self.summarizer:
            logger.info("Initializing summarization model")
            self.summarizer = pipeline("summarization", model="Falconsai/text_summarization")
        
        logger.info(f"Processing {len(splits)} document splits")
        
        for i, split in enumerate(splits):
            try:
                # Ensure minimum content length for summarization
                content = split.page_content.strip()
                if len(content) < 50:  # Skip very short chunks
                    continue
                    
                summary_result = self.summarizer(
                    content, 
                    max_length=150, 
                    min_length=30, 
                    do_sample=False
                )
                
                for result in summary_result:
                    summary_text = result["summary_text"].strip()
                    if summary_text:
                        self.summary += " " + summary_text
                        
                logger.debug(f"Processed split {i+1}/{len(splits)}")
                
            except Exception as e:
                logger.warning(f"Error processing split {i+1}: {str(e)}")
                continue
        
    def splitText(self, data: List[Any]) -> List[Any]:
        """Split documents into manageable chunks for processing."""
        logger.info("Splitting documents into chunks")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased for better context
            chunk_overlap=200,  # More overlap for better continuity
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        all_splits = text_splitter.split_documents(data)
        logger.info(f"Created {len(all_splits)} text chunks")
        
        return all_splits
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get statistics about the generated summary."""
        return {
            "summary_length": len(self.summary),
            "word_count": len(self.summary.split()) if self.summary else 0,
            "file_processed": os.path.basename(self.file_path),
            "file_size": os.path.getsize(self.file_path)
        }