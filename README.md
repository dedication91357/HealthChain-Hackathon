# HealthChain Decoder

A secure blockchain-powered platform that decodes and simplifies medical records. HealthChain Decoder uses advanced blockchain technology for document security while employing AI models to translate complex medical terminology into accessible language. By securely storing health records on a distributed ledger and making them understandable to patients, the application bridges the gap between medical professionals and individuals seeking to better understand their healthcare.

## Demo

A demonstration video (demo.mp4) is available in the repository that shows the application in action. This video walks through the full functionality of the system including document upload, processing, and the blockchain integration. We recommend watching the demo to get a better understanding of how the application works.


https://github.com/user-attachments/assets/a1240b19-8ce5-4572-b516-2a0ef5e17ee1


## The Inspiration

Health records are challenging to understand and often include a lot of foreign jargon. When it comes to your healthcare, having transparency on what medications and procedures you are being prescribed without the compromise of security is key. We solve this problem by leveraging natural language modeling techniques that provide users with greater insight into their medical documents. Specifically by employing a named entity recognition model to identify keywords within a patient's health report and an informed summarization model that concisely explains your care pathway.

## Features
- PDF document upload and processing
- Medical entity extraction using NER
- Document summarization
- Blockchain integration for secure document storage
- Modern, responsive user interface


## Usage
1. Start the Flask application:
   ```
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```
3. Upload a PDF medical document on the home page.
4. View the extracted entities and document summary on the results page.

## Technical Details

### NER (Named Entity Recognition)
- Uses the `blaze999/clinical-ner` model from Hugging Face for medical entity recognition
- Entities are highlighted with different colors based on their type
- Supports medical-specific entity types like PROBLEM, TEST, TREATMENT, etc.

### Summarization
- Uses the `Falconsai/text_summarization` model for document summarization
- Processes documents in chunks to handle long medical texts efficiently
- Condenses complex medical information into concise summaries

### Blockchain Integration
- Integrates with a blockchain network via a Node.js script
- Securely stores document hashes and metadata on the blockchain
- Provides transaction IDs for future verification

## Technologies
- Python
- Flask
- NodeJS
- HuggingFace
- LangChain
- RAG Pipelines
- Web3 technologies
- PyMuPDF
- Transformers

## Important Note
`sdk.js` contains private information and is not pushed to this repository. To duplicate our results, you must develop and attach your own NodeJS SDK file to connect the file uploading to a Web3 storage database.
