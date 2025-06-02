from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
import os

class Summarize:
    def __init__(self, file_path):
        self.file_path = file_path
        self.loaders = {
            "txt": TextLoader,
            "pdf": PyPDFLoader,
            "csv": CSVLoader
        }
        self.summary = ""
        
    def getLoader(self):
        file_extension = os.path.splitext(self.file_path)[1][1:].lower()
        if file_extension not in self.loaders:
            raise ValueError(f"Unsupported file extension: {file_extension}")
            
        loader = self.loaders[file_extension]
        load = loader(self.file_path)
        data = load.load()
        return data
        
    def callRag(self):
        data = self.getLoader()
        splits = self.splitText(data)
        self.sum_splits(splits)
        return self.summary
        
    def sum_splits(self, splits):
        summarizer = pipeline("summarization", model="Falconsai/text_summarization")
        for split in splits:
            sum = summarizer(split.page_content, max_length=5, min_length=1, do_sample=False)
            for s in sum:
                self.summary += " " + s["summary_text"]
        
    def splitText(self, data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_splits = text_splitter.split_documents(data)
        return all_splits