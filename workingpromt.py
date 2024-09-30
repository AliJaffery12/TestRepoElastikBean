from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import openai
import re
from dotenv import load_dotenv
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['STATIC_FOLDER'] = 'static/'

load_dotenv()
openai_api_key = ""
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in .env file")


openai.api_key = openai_api_key

ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    
    # Read the first 2-3 pages
    for page_num in range(min(2, doc.page_count)):
        text += doc[page_num].get_text()

    # Perform basic cleanup to remove excess metadata
    clean_text = re.sub(r"(DOI:.*|http[s]?://\S+|\bDownloaded\b.*|Science, \d{4}|Permission.*|[Cc]opyright.*)", "", text)
    
    return clean_text

# Function to detect if a PDF is image-based
def is_image_based_pdf(file_path):
    doc = fitz.open(file_path)
    image_based_pages = 0
    
    # Check the first 2 pages to detect if they are primarily images
    for page_num in range(min(2, doc.page_count)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        text = page.get_text()

        
        if image_list and len(text) < 50:
            image_based_pages += 1
    
    return image_based_pages > 0

# Standard prompt for regular PDFs
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

# Special prompt for image-based PDFs
def create_prompt_for_image(text):
    prompt = f"""
    The following research article contains scanned images or has limited extractable text. Please do the following:
    
    1. Describe the image content in detail.
    2. Try to identify any visible text, especially the names of authors, abstract,Corresponding authors with emails (if any) and any institutions.
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
            max_tokens=1000,
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


def process_pdf_with_openai(file_path):
    # Detect if the PDF is image-based
    if is_image_based_pdf(file_path):
        print("Detected image-based PDF. Attempting to describe images and extract text.")
        extracted_text = extract_text_from_pdf(file_path)[:4000]
        prompt = create_prompt_for_image(extracted_text)
    else:
        # For normal text-based PDFs
        extracted_text = extract_text_from_pdf(file_path)[:4000]
        print("Extracted text:", extracted_text)
        if not extracted_text:
            raise ValueError("No text extracted from the PDF. The PDF might be empty or corrupted.")
        
        prompt = create_standard_prompt(extracted_text)
    

    extracted_info_json = extract_info_via_openai(prompt)


    try:
        extracted_info_dict = json.loads(extracted_info_json)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"OpenAI API returned: {extracted_info_json}")
        
        return extracted_info_json

   
    return extracted_info_dict

@app.route('/uploadpaper', methods=['GET', 'POST'])
def uploadpaper():
    if request.method == 'POST':
        return redirect(url_for('upload_ResearchPaperfile'))
    return render_template('upload.html')

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
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                
                info_json = process_pdf_with_openai(file_path)
                # os.remove(file_path)  
                return jsonify(info_json)  # Return formatted JSON response
            except Exception as e:
                # os.remove(file_path)  # Remove the file if an error occurs
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    return render_template('upload.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
    app.run(debug=True)
