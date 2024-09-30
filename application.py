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

application = Flask(__name__)

application.config['KEYWORDS_FOLDER'] = 'keywords/'


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in .env file")

openai.api_key = openai_api_key


@application.route('/')
def home():
    response_html = f"<p>Connecting the deployed endpoint to moodle for test response</p>"
    return response_html

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def get_latest_keywords_file():
    keywords_dir = os.path.join(application.root_path, application.config['KEYWORDS_FOLDER'])
    pdf_files = [f for f in os.listdir(keywords_dir) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError("No keyword PDF files found in the keywords folder.")
    latest_file = max(pdf_files, key=lambda f: os.path.getmtime(os.path.join(keywords_dir, f)))
    return os.path.join(keywords_dir, latest_file)


def preprocess_image(image_path, max_size=600):
    # Read the image
    img = cv2.imread(image_path)
    
    # Resize the image while maintaining aspect ratio
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # applicationly thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return thresh

def extract_cas_number(image_path):
    preprocessed_image = preprocess_image(image_path)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(preprocessed_image)
    
    extracted_text = ' '.join([result[1] for result in results])
    print(f"Extracted text: {extracted_text}")
    
    # Look for CAS number indicators
    cas_indicators = r'(?:CAS|CAS-Nr|CAS Nr|CAS Number)[.:]?\s*'
    
    # Look for the CAS number format (digits-digits-digit) after the indicator
    cas_pattern = r'(\d{1,7}-\d{2}-\d)'
    
    full_pattern = cas_indicators + cas_pattern
    
    match = re.search(full_pattern, extracted_text, re.IGNORECASE)
    if match:
        cas_number = match.group(1)
        print(f"Extracted CAS Number: {cas_number}")
        return cas_number
    
    # If not found with indicator, try to find any occurrence of the CAS number format
    match = re.search(cas_pattern, extracted_text)
    if match:
        cas_number = match.group(1)
        print(f"Extracted CAS Number (without indicator): {cas_number}")
        return cas_number
    
    print("No valid CAS number found in the image. Please upload a clearer picture.")
    return None

def validate_cas_number(cas_number):
    # Remove any non-digit characters
    digits = re.sub(r'\D', '', cas_number)
    
    if len(digits) < 5:
        return False
    
    # Calculate the check digit
    total = sum(int(digit) * (i + 1) for i, digit in enumerate(digits[:-1][::-1]))
    check_digit = total % 10
    
    return check_digit == int(digits[-1])

def get_chemical_data(cas_number):
    cas_number_no_hyphens = cas_number.replace('-', '')
    
    base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
    
    
    cid_url = f'{base_url}/compound/name/{cas_number}/cids/JSON'
    cid_response = requests.get(cid_url)
    
    if cid_response.status_code != 200:
        return {'error': f'Failed to fetch CID: {cid_response.status_code} - {cid_response.text}'}
    
    cid_data = cid_response.json()
    cid = cid_data.get('IdentifierList', {}).get('CID', [])
    
    if not cid:
        return {'error': 'CID not found for the given CAS number'}
    
    cid = cid[0]
    
    
    props = ["MolecularFormula", "MolecularWeight", "IUPACName", "ExactMass"]
    props_str = ",".join(props)
    
    property_url = f'{base_url}/compound/cid/{cid}/property/{props_str}/JSON'
    property_response = requests.get(property_url)
    
    if property_response.status_code != 200:
        return {'error': f'Failed to fetch properties: {property_response.status_code} - {property_response.text}'}
    
    property_data = property_response.json()
    
    
    exp_property_url = f'{base_url}_view/data/compound/{cid}/JSON'
    exp_property_response = requests.get(exp_property_url)
    
    if exp_property_response.status_code != 200:
        return {'error': f'Failed to fetch experimental properties: {exp_property_response.status_code} - {exp_property_response.text}'}
    
    exp_property_data = exp_property_response.json()
    
    
    compound_properties = property_data.get('PropertyTable', {}).get('Properties', [{}])[0]
    
    def extract_experimental_data(data, property_name):
        sections = data.get('Record', {}).get('Section', [])
        for section in sections:
            if section.get('TOCHeading') == 'Chemical and Physical Properties':
                subsections = section.get('Section', [])
                for subsection in subsections:
                    if subsection.get('TOCHeading') == 'Experimental Properties':
                        properties = subsection.get('Section', [])
                        for prop in properties:
                            if prop.get('TOCHeading') == property_name:
                                return prop.get('Information', [{}])[0].get('Value', {}).get('StringWithMarkup', [{}])[0].get('String', 'N/A')
        return 'N/A'
    
    combined_properties = {
        'cas_number': cas_number,
        'molecular_formula': compound_properties.get('MolecularFormula', 'N/A'),
        'molecular_weight': compound_properties.get('MolecularWeight', 'N/A'),
        'iupac_name': compound_properties.get('IUPACName', 'N/A'),
        'exact_mass': compound_properties.get('ExactMass', 'N/A'),
        'density': extract_experimental_data(exp_property_data, 'Density')
    }
    
    return combined_properties




@application.route('/UploadChemicalImage', methods=['GET', 'POST'])
def upload_chemical_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(application.config['KEYWORDS_FOLDER'], filename)
            file.save(image_path)

            try:
                cas_number = extract_cas_number(image_path)
                if cas_number:
                    if validate_cas_number(cas_number):
                        chemical_data = get_chemical_data(cas_number)
                        return jsonify(chemical_data)
                    else:
                        return jsonify({'error': 'Extracted CAS number is not valid'}), 400
                else:
                    return jsonify({'error': 'No CAS number found in the image'}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            finally:
                os.remove(image_path)
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    return render_template('upload_image.html')


if __name__ == '__main__':
    
    application.run(debug=True)