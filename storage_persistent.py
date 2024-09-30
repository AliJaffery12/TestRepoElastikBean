from PIL import Image
import easyocr

# Load the image
image_path = 'C:\\Users\\alija\\Desktop\\Extraction_and_keyword_Matching\\Research_Paper_Extraction_And_keywords_match\\Flask_App\\keywords\\IMG_20240619_115351.jpg'
img = Image.open(image_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en', 'de'])

# Perform OCR on the image
results = reader.readtext(image_path, detail=0)

# Print extracted text from the image
extracted_text = ' '.join(results)
print("Extracted Text: ", extracted_text)

# Look for CAS number pattern
import re
cas_pattern = r'\d{2,7}-\d{2}-\d'
cas_number = re.findall(cas_pattern, extracted_text)
print("CAS Number: ", cas_number if cas_number else "No CAS Number found")


from flask import Flask, request, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import openai
import re
import json
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import os
from sqlalchemy.exc import SQLAlchemyError
from flask import abort
import easyocr # For OCR
from PIL import Image  # For opening image files
import requests  # For PubChem API
import cv2
import numpy
import requests  

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pdfs.db'  # You can change this to your preferred database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['KEYWORDS_FOLDER'] = 'keywords/'
db = SQLAlchemy(app)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in .env file")

openai.api_key = openai_api_key


ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

class PDF(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_keywords_from_pdf(file_path):
    keywords = []
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        start_extracting = False
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            lines = page_text.split('\n')
            for line in lines:
                line = line.strip()
                if line == "Analytical Chemistry and":
                    start_extracting = True
                    continue
                if start_extracting and line and not line[0].isdigit() and ":" not in line:
                    words = line.split()
                    if 1 <= len(words) <= 8 and all(len(word) > 1 for word in words):
                        if not any(word.lower() in ['and', 'or', 'the', 'of', 'including'] for word in words):
                            if not line.endswith(('and', 'or', 'the', 'of')):
                                keywords.append(line)
    return keywords

def get_latest_keywords_file():
    keywords_dir = os.path.join(app.root_path, app.config['KEYWORDS_FOLDER'])
    pdf_files = [f for f in os.listdir(keywords_dir) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError("No keyword PDF files found in the keywords folder.")
    latest_file = max(pdf_files, key=lambda f: os.path.getmtime(os.path.join(keywords_dir, f)))
    return os.path.join(keywords_dir, latest_file)

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Apply dilation to strengthen the text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Save the preprocessed image
    preprocessed_path = image_path.rsplit('.', 1)[0] + '_preprocessed.png'
    cv2.imwrite(preprocessed_path, dilated)
    
    return preprocessed_path

def extract_cas_number(image_path):
    # Initialize the EasyOCR reader with the desired languages (e.g., English)
    reader = easyocr.Reader(['en', 'de'])
    
    # Perform OCR on the image
    results = reader.readtext(image_path, detail=0)
    
    # Join the extracted text into a single string
    extracted_text = ' '.join(results)
    
    print(f"Extracted text: {extracted_text}")  # Debug: Print extracted text
    
    # Improved regular expression for CAS number
    cas_patterns = [
        r'\d{2,7}-\d{2}-\d',
        r'\b([1-9]\d{1,6}-\d{2}-\d)\b',
        r'\b(?:CAS[-\s]?(?:No\.?|Number)?[:.]?|CAS[-\s]?Nr\.?[:.]?)?\s*([1-9]\d{1,6}[-\s]?\d{2}[-\s]?\d)\b',
        r'\b([1-9]\d{1,6}[-\s]?\d{2}[-\s]?\d)(?:\s*CAS)?\b'
    ]
    
    for i, pattern in enumerate(cas_patterns):
        print(f"Trying pattern {i + 1}: {pattern}")  # Debug: Print current pattern
        matches = re.findall(pattern, extracted_text, re.IGNORECASE)
        if matches:
            for match in matches:
                # Remove hyphens and spaces from the extracted CAS number
                normalized_cas = re.sub(r'[-\s]', '', match)
                print(f"Extracted CAS Number: {normalized_cas}")
                return normalized_cas
    
    print("No CAS number found in the image")
    return None

def get_chemical_data(cas_number):
    base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/'
    
    # Log the request URL
    print(f"Requesting PubChem API for normalized CAS number: {cas_number}")
    
    # Send request to PubChem API using the CAS number as cid and ask for specific properties
    response = requests.get(f'{base_url}/{cas_number}/property/MolecularWeight,MolecularFormula,Title/JSON')
    
    if response.status_code == 200:
        try:
            data = response.json()
            # Extract information from the JSON response
            properties = data.get('PropertyTable', {}).get('Properties', [])
            
            if not properties:
                return {'error': 'No compound information available in PubChem data'}
            
            # Extract the required properties from the first item
            compound_data = properties[0]
            molecular_weight = compound_data.get('MolecularWeight', "Not available")
            molecularFormula = compound_data.get('MolecularFormula', "Not available")
            title = compound_data.get('Title', "Not available")
            
            print(f"CAS Number: {cas_number}, Molecular Weight: {molecular_weight}, MolecularFormula: {molecularFormula}, Title: {title}")
            return {
                'cas_number': cas_number,
                'Molecular_weight': molecular_weight,
                'MolecularFormula': molecularFormula,
                'Title': title,
            }
        except (IndexError, KeyError, json.JSONDecodeError) as e:
            return {'error': f'Failed to parse PubChem data: {str(e)}'}
    else:
        # Log the error message returned by PubChem
        return {'error': f'PubChem API issue: {response.status_code} - {response.text}'}




@app.route('/UploadChemicalImage', methods=['GET', 'POST'])
def upload_chemical_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['KEYWORDS_FOLDER'], filename)  # Temporarily save image
            file.save(image_path)

            try:
                # Extract CAS number
                cas_number = extract_cas_number(image_path)
                if cas_number:
                    # Get chemical data from PubChem
                    chemical_data = get_chemical_data(cas_number)
                    return jsonify(chemical_data)
                else:
                    return jsonify({'error': 'No CAS number found in the image'}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    return render_template('upload_image.html')

def match_keywords_using_tfidf(text, keywords):
    corpus = [text] + keywords
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    keyword_similarity_pairs = [(keywords[i], cosine_similarities[i]) for i in range(len(keywords))]
    keyword_similarity_pairs = sorted(keyword_similarity_pairs, key=lambda x: x[1], reverse=True)
    seen_keywords = set()
    unique_keywords = []
    for keyword, similarity in keyword_similarity_pairs:
        if keyword not in seen_keywords:
            seen_keywords.add(keyword)
            unique_keywords.append((keyword, similarity))
        if len(unique_keywords) >= 10:
            break
    return unique_keywords

def extract_text_from_pdf_bytes(pdf_bytes):
    pdf_file = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    text = ""
    for page_num in range(min(2, doc.page_count)):
        text += doc[page_num].get_text()
    clean_text = re.sub(r"(DOI:.*|http[s]?://\S+|\bDownloaded\b.*|Science, \d{4}|Permission.*|[Cc]opyright.*)", "", text)
    return clean_text

def is_image_based_pdf_bytes(pdf_bytes):
    pdf_file = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    image_based_pages = 0
    for page_num in range(min(2, doc.page_count)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        text = page.get_text()
        if image_list and len(text) < 50:
            image_based_pages += 1
    return image_based_pages > 0

def create_standard_prompt(text):
    prompt = f"""
    I have the following research article text:
    
    {text}

    Please extract the following information in JSON format:
    - Authors
    - Abstract
    - Institutions
    - Corresponding authors with emails (if any)

    Return the result in this JSON structure:
    {{
        "authors": [],
        "abstract": "",
        "institutions": [],
        "corresponding_authors": []
    }}
    """
    return prompt

def create_prompt_for_image(text):
    prompt = f"""
    The following research article contains scanned images or has limited extractable text. Please do the following:
    
    1. Describe the image content in detail.
    2. Try to identify any visible text, especially the names of authors, abstract, Corresponding authors with emails (if any) and any institutions.
    3. Use best-guess reasoning based on the image's content to provide relevant information.
    
    Return the result only in valid JSON format, nothing else:
    {{
        "authors": [],
        "abstract": "",
        "institutions": [],
        "corresponding_authors": []
    }}
    """
    return prompt

def extract_info_via_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0
        )
        if 'choices' not in response or not response['choices']:
            raise ValueError("No valid response from OpenAI API")
        extracted_info = response['choices'][0]['message']['content'].strip()
        json_start = extracted_info.find('{')
        json_end = extracted_info.rfind('}')
        clean_json = extracted_info[json_start:json_end + 1]
        return clean_json
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        raise ValueError(f"OpenAI API Error: {e}")

def process_pdf_with_openai_and_keywords(pdf_bytes):
    if is_image_based_pdf_bytes(pdf_bytes):
        print("Detected image-based PDF. Attempting to describe images and extract text.")
        extracted_text = extract_text_from_pdf_bytes(pdf_bytes)
        prompt = create_prompt_for_image(extracted_text)
    else:
        extracted_text = extract_text_from_pdf_bytes(pdf_bytes)
        if not extracted_text:
            raise ValueError("No text extracted from the PDF. The PDF might be empty or corrupted.")
        prompt = create_standard_prompt(extracted_text)

    extracted_info_json = extract_info_via_openai(prompt)
    extracted_info_dict = json.loads(extracted_info_json)

    keyword_list_path = get_latest_keywords_file()
    keywords = extract_keywords_from_pdf(keyword_list_path)
    matched_keywords = match_keywords_using_tfidf(extracted_info_dict.get('abstract', ''), keywords)
    extracted_info_dict['matched_keywords'] = matched_keywords

    return extracted_info_dict

@app.route('/db_info', methods=['GET'])
def get_db_info():
    try:
        # Get total number of PDFs
        total_pdfs = PDF.query.count()

        # Get information about the last 5 uploaded PDFs
        recent_pdfs = PDF.query.order_by(PDF.id.desc()).limit(5).all()
        recent_pdf_info = [
            {
                'id': pdf.id,
                'filename': pdf.filename,
                'size': len(pdf.data)  # Size in bytes
            } for pdf in recent_pdfs
        ]

        return jsonify({
            'total_pdfs': total_pdfs,
            'recent_uploads': recent_pdf_info
        })
    except SQLAlchemyError as e:
        return jsonify({'error': 'Database error occurred'}), 500
    

@app.route('/UploadResearchPaper', methods=['GET', 'POST'])
def upload_ResearchPaperfile():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pdf_bytes = file.read()

            # Save PDF to database
            new_pdf = PDF(filename=filename, data=pdf_bytes)
            db.session.add(new_pdf)
            db.session.commit()

            try:
                info_json = process_pdf_with_openai_and_keywords(pdf_bytes)
                return jsonify(info_json)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    return render_template('upload.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)