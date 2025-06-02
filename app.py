from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from summarizer import Summarize
import fitz  # PyMuPDF
from transformers import pipeline
import hashlib
import json
import os
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def call_node_function():
    try:
        result = subprocess.run(
            ['node', 'sdk.js'],  # Path to the Node.js script
            capture_output=True,
            text=True
        )
        
        # Print detailed output for debugging
        print(f"Arguments: {result.args}")
        print(f"Return Code: {result.returncode}")
        print(f"Standard Output:\n{result.stdout}")
        print(f"Standard Error:\n{result.stderr}")

        # Raise an exception if the process exited with a non-zero status
        if result.returncode != 0:
            raise Exception(f"Node.js script exited with error code {result.returncode}")
        
        # Process the output
        output_lines = result.stdout.strip().split('\n')
        return output_lines
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

class NER:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pipe = None

    # Function to extract text from PDF
    def extract_text_from_pdf(self):
        doc = fitz.open(self.file_path)
    
        text = ""
        # Iterate over each page and extract text
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    
    def run_NER(self):
        # Initialize the NER model
        self.pipe = pipeline("token-classification", model="blaze999/clinical-ner", aggregation_strategy='simple')
        
        # Extract text from the PDF
        pdf_text = self.extract_text_from_pdf()

        # Run the NER model on the extracted text
        ner_output = self.pipe(pdf_text)

        highlighted_text = self.highlight_text(pdf_text, ner_output)
        
        return highlighted_text

    # Function to generate a consistent light color for each entity type
    def generate_color_for_entity(self, entity_type):
        # Hash the entity type to get a consistent value
        hash_value = int(hashlib.md5(entity_type.encode()).hexdigest(), 16)
        # Generate a light color
        color_value = (hash_value % 0xFFFFFF) + 0x808080
        color = "#{:06x}".format(color_value & 0xFFFFFF)
        return color

    # Function to highlight entities in the text
    def highlight_text(self, text, ner_output):
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was submitted
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If the user doesn't select a file, browser submits an empty file
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save the file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the file
        try:
            # Call Node.js function (blockchain-related)
            node_result = call_node_function()
            
            # Generate summary
            summarizer = Summarize(file_path)
            summary_text = summarizer.callRag()
            
            # Run NER
            ner = NER(file_path)
            highlighted_text = ner.run_NER()
            
            # Return the processed results
            return render_template(
                'results.html',
                summary=summary_text,
                highlighted_text=highlighted_text,
                filename=filename
            )
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)