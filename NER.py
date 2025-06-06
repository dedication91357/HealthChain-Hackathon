import fitz  # PyMuPDF+
from transformers import pipeline
import hashlib
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class NER:
    """Named Entity Recognition class for medical document processing."""
    
    def __init__(self, file):
        self.uploaded_file = file
        self.pipe = None
        self.entity_cache = {}
        logger.info("NER instance initialized")

    # Function to extract text from PDF
    def extract_text_from_pdf(self, pdf_file):
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    
    def run_NER(self):
        # Initialize the NER model
        self.pipe = pipeline("token-classification", model="blaze999/clinical-ner", aggregation_strategy='simple')
        # Streamlit app
        if self.uploaded_file is not None:
            # Extract text from the uploaded PDF
            pdf_text = self.extract_text_from_pdf(self.uploaded_file)

            st.write("Extracted Text:")
            st.write(pdf_text)

            # Run the NER model on the extracted text
            ner_output = self.ner_model(pdf_text)

            highlighted_text = self.highlight_text(pdf_text, ner_output)

            st.write("Text with Highlighted Entities:")
            st.markdown(highlighted_text, unsafe_allow_html=True)

    # Function to run NER model
    def ner_model(self,text):
        result = self.pipe(text)
        return result

    def generate_color_for_entity(self, entity_type):
        """Generate a consistent color for each entity type with caching."""
        if entity_type in self.entity_cache:
            return self.entity_cache[entity_type]
            
        # Hash the entity type to get a consistent value
        hash_value = int(hashlib.md5(entity_type.encode()).hexdigest(), 16)
        # Generate a more vibrant color with better contrast
        hue = (hash_value % 360)
        saturation = 70 + (hash_value % 30)  # 70-100% saturation
        lightness = 80 + (hash_value % 15)   # 80-95% lightness
        
        # Convert HSL to hex (simplified)
        color = f"hsl({hue}, {saturation}%, {lightness}%)"
        
        # Cache the color for performance
        self.entity_cache[entity_type] = color
        logger.debug(f"Generated color {color} for entity type: {entity_type}")
        
        return color

    # Function to highlight entities in the text
    def highlight_text(self,text, ner_output):
        highlighted_text = ""
        last_idx = 0

        # Sort entities by their start index
        ner_output = sorted(ner_output, key=lambda x: x['start'])

        # Assign colors to each entity type
        entity_colors = {}
        for entity in ner_output:
            entity_type = entity['entity_group']
            if entity_type not in entity_colors:
                entity_colors[entity_type] = self.generate_color_for_entity(entity_type)

        css = "<style>"
        for entity_type, color in entity_colors.items():
            css += f"""
            .{entity_type} {{
                background-color: {color};
                padding: 2px 6px;
                border-radius: 4px;
                display: inline-block;
                margin: 2px;
            }}
            .{entity_type} .label {{
                font-size: 10px;
                color: #fff;
                background-color: rgba(0, 0, 0, 0.6);
                border-radius: 3px;
                padding: 0 3px;
                margin-left: 5px;
            }}
            """
        css += "</style>"

        for entity in ner_output:
            start = entity['start']
            end = entity['end']
            entity_text = entity['word']
            entity_type = entity['entity_group']

            # Add text before the entity
            highlighted_text += text[last_idx:start]
            # Highlight the entity
            highlighted_text += f'<span class="{entity_type}">{entity_text}<span class="label">{entity_type}</span></span>'
            last_idx = end

        # Add the remaining text
        highlighted_text += text[last_idx:]

        return css + highlighted_text
