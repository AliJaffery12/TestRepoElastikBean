from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import openai
import re
import json
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

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

    return text

# Extract keywords from the PDF keyword list
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

def match_keywords_using_tfidf(text, keywords):
   
    corpus = [text] + keywords


    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Pair keywords with their similarity scores
    keyword_similarity_pairs = [(keywords[i], cosine_similarities[i]) for i in range(len(keywords))]
    
    
    keyword_similarity_pairs = sorted(keyword_similarity_pairs, key=lambda x: x[1], reverse=True)
    return keyword_similarity_pairs[:10]

# Function to create a prompt and send it to OpenAI for processing
def extract_info_via_openai(text):
    # Create the prompt
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

   
    response = openai.ChatCompletion.create(
        model="gpt-4",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0
    )

    
    extracted_info = response['choices'][0]['message']['content'].strip()

    return extracted_info

def process_pdf_with_openai_and_keywords(file_path, keyword_list_path):
   
    extracted_text = extract_text_from_pdf(file_path)

  
    extracted_info_json = extract_info_via_openai(extracted_text)

   
    extracted_info_dict = json.loads(extracted_info_json)

    
    keywords = extract_keywords_from_pdf(keyword_list_path)

   
    matched_keywords = match_keywords_using_tfidf(extracted_text, keywords)

   
    extracted_info_dict['matched_keywords'] = matched_keywords
    return extracted_info_dict

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
               
                keyword_list_path = 'C:\\Users\\alija\\Desktop\\Postingboost solution\\Thief_Catching_App\\keywords_bycateg.pdf'

                
                info_json = process_pdf_with_openai_and_keywords(file_path, keyword_list_path)

               
                # os.remove(file_path)
                return jsonify(info_json)  
            except Exception as e:
               
                # os.remove(file_path)
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    return render_template('upload.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
    app.run(debug=True)
