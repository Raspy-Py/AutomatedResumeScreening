import re
import PyPDF2
import pdfplumber
import unicodedata


class PDFReader:
    def __init__(self, backend: str = 'pypdf2'):
        if backend == 'pypdf2':
            self.backend = self._read_pypdf2
        elif backend == 'pdfplumber':
            self.backend = self._read_pdfplumber
        else:
            raise ValueError("Backend not supported")
        
    def read(self, file_path: str) -> str:
        return self.backend(file_path)
    
    def _read_pypdf2(self, file_name):
        text = ""
        with open(file_name, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        return text

    def _read_pdfplumber(self, file_name):
        text = ""
        with pdfplumber.open(file_name) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
                
        return text  

class TextPreprocessor:
    def __init__(self):
       pass


    def normalize(self, text: str) -> str:
        """
        Normalize text by performing the following operations:
        - Convert to lowercase
        - Remove extra whitespace
        - Handle unicode characters
        - Remove multiple spaces
        - Remove email addresses (optional)
        - Remove URLs (optional)
        - Remove phone numbers (optional)
        
        Args:
            text (str): Input text to normalize
            
        Returns:
            str: Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Handle unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    