from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import fitz  # PyMuPDF
import spacy
import re

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


nlp = spacy.load("en_core_web_sm")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_author_name(name):
    return re.sub(r'[\*†]', '', name).strip()


def split_authors(author_line):
   
    parts = re.split(r',|\band\b', author_line)
    return [clean_author_name(part) for part in parts if part.strip()]

def extract_info_from_pdf(file_path):
    doc = fitz.Document(file_path)
    text = ""
    
    
    for page_num in range(min(2, doc.page_count)):
        text += doc[page_num].get_text()

    
    lines = text.split("\n")

    authors = []
    institutions = []
    current_institutions = []
    author_institution_map = {}  
    found_abstract = False
    abstract = ""
    corresponding_authors = []  
    current_author = None
    collecting_institutions = True  
    title_section_passed = False  

   
    institution_keywords = ['university', 'institute', 'department', 'school', 'faculty', 'college']
    non_author_keywords = ['vocabulary', 'language', 'study', 'paper', 'research', 'assessment', 'tokenization']

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if the line is likely the title (e.g., contains words that aren't names)
        if not title_section_passed and any(word.lower() in line.lower() for word in non_author_keywords):
            continue  
        else:
            title_section_passed = True  

        # Stop collecting institutions once we reach the Abstract section
        if re.match(r'^\s*Abstract', line, re.IGNORECASE):
            found_abstract = True
            collecting_institutions = False  
            continue
      
        if found_abstract:
            if re.match(r'^\s*(Introduction|Keywords|References)', line, re.IGNORECASE):
                found_abstract = False  
            else:
                abstract += line + " "
            continue

        if re.match(r"^[A-Z][a-z]+\s[A-Z][a-z]+|^[A-Z]\.\s[A-Z][a-z]+", line) and not any(keyword.lower() in line.lower() for keyword in institution_keywords):
            
            author_list = split_authors(line)
            authors.extend(author_list)  
            current_author = author_list[0]  
            collecting_institutions = True  # Allow institution collection again for these authors
            continue
        
        # If current_author is set and we're collecting institutions, add institutions related to this author
        if current_author and collecting_institutions and any(keyword.lower() in line.lower() for keyword in institution_keywords):
            current_institutions.append(line)
            institutions.append(line)  
           
            for author in author_list:
                author_institution_map[author] = current_institutions

        # Check for the corresponding author based on email, collect all emails
        email_matches = re.findall(r'(\S+@\S+)', line)
        if email_matches:
            corresponding_authors.extend(email_matches)

    authors = list(set(authors))  # Remove duplicates
    institutions = list(set(institutions))  
    corresponding_authors = list(set(corresponding_authors))  

    return {
        'authors': authors,
        'institutions': institutions,
        'abstract': abstract.strip(),
        'corresponding_authors': corresponding_authors or ["Not found"]
    }

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
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
                info = extract_info_from_pdf(file_path)
                os.remove(file_path)  
                return jsonify(info)
            except Exception as e:
                os.remove(file_path)  
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    return render_template('upload.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)








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
    
   
    for page_num in range(min(2, doc.page_count)):
        text += doc[page_num].get_text()

    return text


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


def process_pdf_with_openai(file_path):
   
    extracted_text = extract_text_from_pdf(file_path)

    
    extracted_info_json = extract_info_via_openai(extracted_text)

    
    extracted_info_dict = json.loads(extracted_info_json)

    #
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
                # os.remove(file_path)  # Optionally remove the file after processing
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
    
   
    for page_num in range(min(2, doc.page_count)):
        text += doc[page_num].get_text()

    
    clean_text = re.sub(r"(DOI:.*|http[s]?://\S+|\bDownloaded\b.*|Science, \d{4}|Permission.*|[Cc]opyright.*)", "", text)
    
    return clean_text


def is_image_based_pdf(file_path):
    doc = fitz.open(file_path)
    image_based_pages = 0
    
    # Check the first 2 pages to detect if they are primarily images
    for page_num in range(min(2, doc.page_count)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        text = page.get_text()

        # Consider the page image-based if there are images and very little text
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
            model="gpt-4o", 
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


def process_pdf_with_openai(file_path):
   
    if is_image_based_pdf(file_path):
        print("Detected image-based PDF. Attempting to describe images and extract text.")
        extracted_text = extract_text_from_pdf(file_path)
        prompt = create_prompt_for_image(extracted_text)
    else:
       
        extracted_text = extract_text_from_pdf(file_path)
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
                # os.remove(file_path)  # Optionally remove the file after processing
                return jsonify(info_json)  
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

# Extract keywords from the keywords_bycateg.pdf file
def extract_keywords_from_pdf(pdf_path):
    keywords = set()
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        # Split the text into lines and process each line
        for line in text.split('\n'):
            # Remove any leading/trailing whitespace and convert to lowercase
            keyword = line.strip().lower()
            # Add non-empty keywords to the set
            if keyword and not keyword.isdigit() and len(keyword) > 1:
                keywords.add(keyword)
    print(keywords)
    return list(keywords)

# Load chemistry keywords from the PDF
chemistry_keywords = extract_keywords_from_pdf('C:\\Users\\alija\\Desktop\\Postingboost solution\\Thief_Catching_App\\keywords_bycateg.pdf')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(min(2, doc.page_count)):
        text += doc[page_num].get_text()
    clean_text = re.sub(r"(DOI:.*|http[s]?://\S+|\bDownloaded\b.*|Science, \d{4}|Permission.*|[Cc]opyright.*)", "", text)
    return clean_text

def is_image_based_pdf(file_path):
    doc = fitz.open(file_path)
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

def match_keywords(abstract):
    prompt = f"""
    Given the following abstract from a chemistry research paper:

    {abstract}

    Please identify the 10 most relevant keywords from the following list that best match the content of the abstract. Return only a JSON array of the selected keywords, nothing else.

    Keyword list: {', '.join(chemistry_keywords)}
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0
        )
        if 'choices' not in response or not response['choices']:
            raise ValueError("No valid response from OpenAI API")
        keywords = json.loads(response['choices'][0]['message']['content'].strip())
        return keywords
    except Exception as e:
        print(f"OpenAI API Error in keyword matching: {e}")
        raise ValueError(f"OpenAI API Error in keyword matching: {e}")

def process_pdf_with_openai(file_path):
    if is_image_based_pdf(file_path):
        print("Detected image-based PDF. Attempting to describe images and extract text.")
        extracted_text = extract_text_from_pdf(file_path)
        prompt = create_prompt_for_image(extracted_text)
    else:
        extracted_text = extract_text_from_pdf(file_path)
        print("Extracted text:", extracted_text)
        if not extracted_text:
            raise ValueError("No text extracted from the PDF. The PDF might be empty or corrupted.")
        prompt = create_standard_prompt(extracted_text)
    
    extracted_info_json = extract_info_via_openai(prompt)
    
    try:
        extracted_info_dict = json.loads(extracted_info_json)
        matched_keywords = match_keywords(extracted_info_dict['abstract'])
        extracted_info_dict['matched_keywords'] = matched_keywords
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
                return jsonify(info_json)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    return render_template('upload.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
    app.run(debug=True)
