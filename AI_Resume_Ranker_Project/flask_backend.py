from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import tempfile
import pickle
import numpy as np
from pathlib import Path
import io
import base64
from docx import Document
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import hashlib
import secrets
from cryptography.fernet import Fernet
from datetime import datetime, timedelta
import jwt
import concurrent.futures
import threading
import time
from functools import wraps
warnings.filterwarnings('ignore')

# Import advanced analytics
from advanced_analytics import AdvancedAnalytics

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data safely"""
    try:
        nltk.data.find('tokenizers/punkt')
    except (LookupError, Exception):
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass

    try:
        nltk.data.find('corpora/stopwords')
    except (LookupError, Exception):
        try:
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass

    try:
        nltk.data.find('corpora/wordnet')
    except (LookupError, Exception):
        try:
            nltk.download('wordnet', quiet=True)
        except Exception:
            pass

# Download NLTK data
download_nltk_data()

app = Flask(__name__)
CORS(app)

# Security configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Performance configuration for large file processing
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1GB max file size
app.config['UPLOAD_TIMEOUT'] = 600  # 10 minutes timeout for uploads
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Encryption key for sensitive data
ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY', Fernet.generate_key())
cipher_suite = Fernet(ENCRYPTION_KEY)

# Access control - simple session-based for demo
active_sessions = {}

def generate_session_token():
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

def encrypt_sensitive_data(data):
    """Encrypt sensitive personal information"""
    if isinstance(data, str):
        data = data.encode()
    return cipher_suite.encrypt(data)

def decrypt_sensitive_data(encrypted_data):
    """Decrypt sensitive personal information"""
    return cipher_suite.decrypt(encrypted_data).decode()

def hash_file_content(content):
    """Create a hash of file content for integrity checking"""
    return hashlib.sha256(content).hexdigest()

def require_auth(f):
    """Decorator to require authentication for sensitive endpoints"""
    def decorated_function(*args, **kwargs):
        session_token = request.headers.get('X-Session-Token')
        if not session_token or session_token not in active_sessions:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def decrypt_resume_data(resume_data):
    """Decrypt sensitive data in resume for display"""
    decrypted_data = resume_data.copy()
    
    try:
        # Decrypt name
        if resume_data.get('name'):
            try:
                encrypted_name = base64.b64decode(resume_data['name'])
                decrypted_name = decrypt_sensitive_data(encrypted_name)
                decrypted_data['name'] = decrypted_name
            except Exception as e:
                print(f"Error decrypting name: {e}")
                decrypted_data['name'] = 'Unknown'
        
        # Decrypt contact info
        if resume_data.get('contact', {}).get('email'):
            try:
                encrypted_email = base64.b64decode(resume_data['contact']['email'])
                decrypted_email = decrypt_sensitive_data(encrypted_email)
                decrypted_data['contact']['email'] = decrypted_email
            except Exception as e:
                print(f"Error decrypting email: {e}")
                decrypted_data['contact']['email'] = ''
        
        if resume_data.get('contact', {}).get('phone'):
            try:
                encrypted_phone = base64.b64decode(resume_data['contact']['phone'])
                decrypted_phone = decrypt_sensitive_data(encrypted_phone)
                decrypted_data['contact']['phone'] = decrypted_phone
            except Exception as e:
                print(f"Error decrypting phone: {e}")
                decrypted_data['contact']['phone'] = ''
                
    except Exception as e:
        print(f"Error in decrypt_resume_data: {e}")
        # Return original data if decryption fails
        return resume_data
    
    return decrypted_data

# Global variables for the AI model
ai_ranker = None
processed_resumes = []
current_results = []
hiring_data = []
advanced_analytics = AdvancedAnalytics()

class AIResumeRanker:
    def __init__(self):
        """Initialize the AI Resume Ranker with necessary components"""
        try:
            self.stemmer = PorterStemmer()
        except Exception:
            self.stemmer = None
            
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None
            
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()
            
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,  # Keep at 2000 as requested
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.kmeans_model = None
        self.processed_resumes = []
        self.tfidf_matrix = None
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except (OSError, Exception):
            print("Warning: spaCy English model not found. Some features may be limited.")
            self.nlp = None
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF files - Optimized PyPDF2 to read all pages with high quality"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                # Read all pages for better quality, but limit total text length for performance
                max_text_length = 10000  # Limit to 10k characters for performance
                for i in range(len(pdf_reader.pages)):
                    try:
                        page_text = pdf_reader.pages[i].extract_text()
                        if page_text and page_text.strip():  # Only add non-empty text
                            text += page_text + "\n"
                            # Stop if we have enough text for analysis
                            if len(text) > max_text_length:
                                break
                    except Exception as page_error:
                        print(f"Error reading page {i+1}: {page_error}")
                        continue
            return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_path):
        """Extract text from DOCX files - Enhanced to read all content with performance optimization"""
        try:
            doc = Document(docx_path)
            text = ""
            max_text_length = 10000  # Limit to 10k characters for performance
            # Read all paragraphs for better quality
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():  # Only add non-empty paragraphs
                    text += paragraph.text + "\n"
                    # Stop if we have enough text for analysis
                    if len(text) > max_text_length:
                        break
            return text
        except Exception as e:
            print(f"Error reading DOCX {docx_path}: {e}")
            return ""
    
    def extract_name(self, text):
        """Extract name from resume text using NER and enhanced patterns"""
        if self.nlp:
            doc = self.nlp(text[:1500])  # Process first 1500 chars for better accuracy
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if persons:
                # Clean name and keep only first two parts
                name = persons[0].strip()
                name = re.sub(r'[0-9]+', '', name)  # Remove numbers
                name_parts = name.split()[:2]  # Keep only first two parts
                return " ".join(name_parts)
        
        # Enhanced pattern matching
        lines = text.split('\n')[:15]  # Check first 15 lines
        
        # Common name patterns
        name_patterns = [
            r'^[A-Z][a-z]+ [A-Z][a-z]+$',  # First Last
            r'^[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+$',  # First M. Last
            r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+$'  # First Middle Last
        ]
        
        for line in lines:
            line = line.strip()
            # Skip lines that are clearly not names
            if any(keyword in line.lower() for keyword in [
                'resume', 'cv', 'email', 'phone', 'address', 'linkedin', 'github',
                'experience', 'education', 'skills', 'objective', 'summary',
                'profile', 'contact', 'personal', 'information'
            ]):
                continue
            
            # Check if line matches name patterns
            for pattern in name_patterns:
                if re.match(pattern, line):
                    return line
            
            # Fallback: simple two-word check
            words = line.split()
            if len(words) == 2 and all(word[0].isupper() for word in words):
                if not any(char.isdigit() for char in line):
                    return line
        
        # Try to find name in the very first line (common resume format)
        first_line = lines[0].strip() if lines else ""
        if first_line and len(first_line.split()) <= 3:
            words = first_line.split()
            if all(word[0].isupper() for word in words) and not any(char.isdigit() for char in first_line):
                return first_line
        
        return "Unknown"
    
    def extract_skills(self, text):
        """Extract skills from resume text - Enhanced version"""
        skills = set()
        text_lower = text.lower()
        
        # Comprehensive technical skills list
        technical_skills = [
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift',
            'kotlin', 'scala', 'r', 'matlab', 'perl', 'shell', 'bash', 'powershell',
            
            # Web Technologies
            'react', 'angular', 'vue', 'vue.js', 'ember', 'backbone', 'jquery', 'html', 'html5', 'css', 'css3',
            'bootstrap', 'tailwind', 'sass', 'scss', 'less', 'stylus', 'webpack', 'gulp', 'grunt',
            
            # Backend Frameworks
            'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'spring boot', 'laravel', 'symfony',
            'rails', 'asp.net', 'dotnet', 'gin', 'fiber', 'echo', 'koa', 'hapi',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'oracle', 'sql server', 'sqlite', 'redis', 'elasticsearch',
            'cassandra', 'dynamodb', 'firebase', 'supabase', 'couchdb', 'neo4j', 'influxdb',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'git', 'gitlab',
            'github', 'bitbucket', 'terraform', 'ansible', 'chef', 'puppet', 'vagrant', 'circleci',
            'travis ci', 'github actions', 'azure devops', 'jira', 'confluence',
            
            # AI/ML/Data Science
            'machine learning', 'deep learning', 'ai', 'artificial intelligence', 'nlp', 'natural language processing',
            'computer vision', 'data science', 'data analysis', 'tensorflow', 'pytorch', 'keras',
            'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'jupyter',
            'opencv', 'spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'tableau', 'power bi',
            
            # Mobile Development
            'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic', 'cordova', 'phonegap',
            
            # Testing
            'jest', 'mocha', 'chai', 'cypress', 'selenium', 'pytest', 'junit', 'testng', 'rspec',
            
            # Other Technologies
            'linux', 'unix', 'windows', 'macos', 'nginx', 'apache', 'tomcat', 'iis',
            'microservices', 'rest api', 'graphql', 'soap', 'json', 'xml', 'yaml',
            'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd'
        ]
        
        # Extract skills using multiple methods
        for skill in technical_skills:
            if skill in text_lower:
                skills.add(skill)
        
        # Also look for skills in specific sections
        skill_sections = [
            'skills:', 'technical skills:', 'technologies:', 'programming languages:',
            'expertise:', 'competencies:', 'tools:', 'software:', 'languages:'
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(section in line_lower for section in skill_sections):
                # Look at the next few lines for skills
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not next_line.lower().startswith(('experience', 'education', 'work')):
                        # Split by common separators
                        potential_skills = re.split(r'[,;|•\-\n]', next_line)
                        for potential_skill in potential_skills:
                            skill_clean = potential_skill.strip().lower()
                            if len(skill_clean) > 2 and len(skill_clean) < 30:
                                # Check if it matches any known skill
                                for known_skill in technical_skills:
                                    if known_skill in skill_clean or skill_clean in known_skill:
                                        skills.add(known_skill)
                                        break
        
        return list(skills)[:25]  # Increased limit to 25 skills
    
    def extract_experience(self, text):
        """Extract work experience from resume text - Ultra Enhanced version"""
        text_lower = text.lower()
        years = 0
        companies = []
        
        # Comprehensive patterns for experience extraction
        experience_patterns = [
            # Direct experience patterns
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:experience|exp)[:\s]*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:work|professional)',
            r'(?:work|professional)[:\s]*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:in\s*)?(?:industry|field)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:total|overall)',
            r'(?:total|overall)[:\s]*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:relevant|related)',
            r'(?:relevant|related)[:\s]*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:hands-on|practical)',
            r'(?:hands-on|practical)[:\s]*(\d+)\s*(?:years?|yrs?)',
            
            # Additional patterns for different formats
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:development|programming|coding)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:in\s*)?(?:software|web|mobile|data)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:with|using)\s*(?:python|java|javascript)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:background|history)',
            r'(?:over|more\s*than|at\s*least)\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:expertise|knowledge)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:in\s*)?(?:it|technology|tech)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:career|professional\s*life)',
            
            # Arabic patterns
            r'(\d+)\s*(?:سنة|سنوات|عام|أعوام)\s*(?:من\s*)?(?:الخبرة|التجربة)',
            r'(?:خبرة|تجربة)[:\s]*(\d+)\s*(?:سنة|سنوات|عام|أعوام)',
            r'(\d+)\s*(?:سنة|سنوات|عام|أعوام)\s*(?:في\s*)?(?:العمل|المجال)',
            
            # Numeric patterns
            r'(\d+)\s*(?:y\.o\.e|yoe)',  # Years of Experience abbreviation
            r'(\d+)\s*(?:yrs\s*exp|years\s*exp)',  # Abbreviated forms
        ]
        
        # Extract years from all patterns
        all_years = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text_lower)
            all_years.extend([int(match) for match in matches])
        
        # Look for experience in specific sections
        experience_sections = [
            'experience:', 'work experience:', 'professional experience:', 'career:',
            'employment history:', 'work history:', 'professional background:',
            'الخبرة:', 'التجربة المهنية:', 'الخبرة العملية:'
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(section in line_lower for section in experience_sections):
                # Look at the next few lines for experience information
                for j in range(i+1, min(i+15, len(lines))):
                    next_line = lines[j].strip()
                    if next_line:
                        # Extract years from this line
                        for pattern in experience_patterns:
                            matches = re.findall(pattern, next_line.lower())
                            all_years.extend([int(match) for match in matches])
        
        # Enhanced date range patterns
        date_patterns = [
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2})[\/\-\.]\s*\d{4}\s*(?:to|until|present|current|\d{4})',
            r'\d{4}\s*(?:to|until|present|current|\-)\s*\d{4}',
            r'\d{4}\s*(?:to|until|present|current|\-)\s*(?:present|current|now)',
            r'(?:from|since)\s*\d{4}\s*(?:to|until|present|current)',
            r'\d{4}\s*-\s*\d{4}',
            r'\d{4}\s*to\s*\d{4}',
            r'\d{4}\s*until\s*\d{4}',
            r'\d{4}\s*present',
            r'\d{4}\s*current'
        ]
        
        # Extract date ranges and calculate years
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                years_from_dates = self._calculate_years_from_dates(match)
                if years_from_dates > 0:
                    all_years.append(years_from_dates)
        
        # Look for job positions and calculate experience from them
        job_position_patterns = [
            r'(?:senior|sr\.?|lead|principal|architect|manager|director|head)',
            r'(?:junior|jr\.?|entry|associate|assistant)',
            r'(?:intern|trainee|fresh|graduate)'
        ]
        
        # Count job positions to estimate experience
        job_count = 0
        for pattern in job_position_patterns:
            matches = re.findall(pattern, text_lower)
            job_count += len(matches)
        
        # If we found job positions but no explicit years, estimate
        if job_count > 0 and not all_years:
            # Estimate based on job positions
            estimated_years = min(job_count * 2, 10)  # Max 10 years estimation
            all_years.append(estimated_years)
        
        # Additional method: Look for job titles and estimate experience
        if not all_years:
            # Look for common job progression patterns
            senior_keywords = ['senior', 'sr.', 'lead', 'principal', 'architect', 'manager', 'director', 'head', 'chief']
            mid_keywords = ['developer', 'engineer', 'analyst', 'specialist', 'consultant', 'coordinator']
            junior_keywords = ['junior', 'jr.', 'entry', 'associate', 'assistant', 'intern', 'trainee', 'fresh', 'graduate']
            
            senior_count = sum(1 for keyword in senior_keywords if keyword in text_lower)
            mid_count = sum(1 for keyword in mid_keywords if keyword in text_lower)
            junior_count = sum(1 for keyword in junior_keywords if keyword in text_lower)
            
            # Estimate based on job level
            if senior_count > 0:
                estimated_years = 5 + (senior_count * 2)  # Senior level: 5+ years
            elif mid_count > 0:
                estimated_years = 2 + (mid_count * 1.5)  # Mid level: 2-5 years
            elif junior_count > 0:
                estimated_years = 0.5 + (junior_count * 0.5)  # Junior level: 0-2 years
            else:
                estimated_years = 0
            
            if estimated_years > 0:
                all_years.append(int(estimated_years))
        
        # Take the maximum years found
        if all_years:
            years = max(all_years)
            # Cap at reasonable maximum
            years = min(years, 50)
        
        # Enhanced company extraction
        company_keywords = [
            'company', 'corp', 'inc', 'ltd', 'llc', 'group', 'solutions', 'technologies',
            'systems', 'services', 'consulting', 'enterprises', 'industries', 'international',
            'global', 'software', 'development', 'engineering', 'consultancy', 'firm',
            'organization', 'institution', 'university', 'college', 'hospital', 'clinic',
            'bank', 'financial', 'investment', 'trading', 'retail', 'manufacturing',
            'agency', 'studio', 'lab', 'center', 'centre', 'academy', 'institute',
            'foundation', 'association', 'society', 'club', 'team', 'department'
        ]
        
        lines = text.split('\n')
        for line in lines[:40]:  # Check first 40 lines
            line = line.strip()
            if len(line) > 3 and len(line) < 100:
                line_lower = line.lower()
                # Check if line contains company indicators
                if any(keyword in line_lower for keyword in company_keywords):
                    # Clean up the company name
                    clean_company = re.sub(r'[^\w\s\-&.,]', '', line)
                    if clean_company and len(clean_company.strip()) > 2:
                        companies.append(clean_company.strip())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_companies = []
        for company in companies:
            if company.lower() not in seen:
                seen.add(company.lower())
                unique_companies.append(company)
        
        return {
            'years': years,
            'companies': unique_companies[:5]  # Max 5 companies
        }
    
    def _calculate_years_from_dates(self, date_text):
        """Calculate years from date range text - Enhanced version"""
        try:
            # Extract years from date text
            years = re.findall(r'\d{4}', date_text)
            current_year = 2024
            
            if len(years) >= 2:
                start_year = int(years[0])
                end_year = int(years[1])
                
                # Handle present/current
                if 'present' in date_text.lower() or 'current' in date_text.lower():
                    end_year = current_year
                
                # Calculate difference
                years_diff = end_year - start_year
                
                # Validate reasonable range
                if 0 <= years_diff <= 50:
                    return years_diff
                    
            elif len(years) == 1:
                # If only one year, assume it's start year
                year = int(years[0])
                
                # If it's a recent year, calculate from then to now
                if year >= 1990 and year <= current_year:
                    return current_year - year
                    
                # If it's an old year, might be end year
                elif year < 1990:
                    return 0  # Too old to be relevant
                    
        except Exception as e:
            print(f"Error calculating years from dates: {e}")
            
        return 0
    
    def extract_contact_info(self, text):
        """Extract contact information from resume"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(?:\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}'
        
        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)
        
        return {
            'email': emails[0] if emails else '',
            'phone': phones[0] if phones else ''
        }
    
    def clean_and_preprocess_text(self, text):
        """Clean and preprocess text for TF-IDF - Optimized for speed"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Fast tokenization - use simple split for speed
        tokens = text.split()
        
        # Fast stopword removal and filtering
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                processed_tokens.append(token)
        
        return ' '.join(processed_tokens)

# Initialize the AI ranker
ai_ranker = AIResumeRanker()

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Simple authentication endpoint for demo purposes"""
    try:
        data = request.get_json()
        username = data.get('username', '')
        password = data.get('password', '')
        
        # Simple demo authentication (in production, use proper user management)
        if username == 'admin' and password == 'admin123':
            session_token = generate_session_token()
            active_sessions[session_token] = {
                'username': username,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=24)
            }
            return jsonify({
                'message': 'Login successful',
                'session_token': session_token,
                'expires_in': 86400  # 24 hours in seconds
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout and invalidate session"""
    try:
        session_token = request.headers.get('X-Session-Token')
        if session_token in active_sessions:
            del active_sessions[session_token]
        return jsonify({'message': 'Logout successful'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-resumes', methods=['POST'])
@require_auth
def upload_resumes():
    """Upload and process resume files"""
    global processed_resumes, ai_ranker
    
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        processed_resumes = []
        
        # Process files in optimized batches for speed and memory efficiency
        batch_size = min(20, len(files))  # Dynamic batch size based on file count
        total_files = len(files)
        
        for i in range(0, total_files, batch_size):
            batch_files = files[i:i+batch_size]
            
            # Progress update for large datasets
            if total_files > 50:
                print(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size} ({min(i + batch_size, total_files)}/{total_files} files)")
            
            for file in batch_files:
                if file.filename == '':
                    continue
                
                # Validate file type
                validation = validate_file_type(file.filename)
                if not validation['valid']:
                    return jsonify({'error': validation['error']}), 400
                    
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
                    file.save(tmp_file.name)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Extract text based on file extension
                    if file.filename.lower().endswith('.pdf'):
                        raw_text = ai_ranker.extract_text_from_pdf(tmp_file_path)
                    elif file.filename.lower().endswith('.docx'):
                        raw_text = ai_ranker.extract_text_from_docx(tmp_file_path)
                    else:
                        continue
                    
                    if not raw_text.strip():
                        continue
                    
                    # Fast information extraction
                    contact_info = ai_ranker.extract_contact_info(raw_text)
                    
                    # Encrypt sensitive personal information
                    encrypted_email = encrypt_sensitive_data(contact_info['email']) if contact_info['email'] else b''
                    encrypted_phone = encrypt_sensitive_data(contact_info['phone']) if contact_info['phone'] else b''
                    encrypted_name = encrypt_sensitive_data(ai_ranker.extract_name(raw_text))
                    
                    # Create file hash for integrity
                    file_hash = hash_file_content(file.read())
                    file.seek(0)  # Reset file pointer
                    
                    resume_data = {
                        'filename': file.filename,
                        'name': base64.b64encode(encrypted_name).decode(),  # Store encrypted
                        'skills': ai_ranker.extract_skills(raw_text),
                        'experience': ai_ranker.extract_experience(raw_text),
                        'contact': {
                            'email': base64.b64encode(encrypted_email).decode() if encrypted_email else '',
                            'phone': base64.b64encode(encrypted_phone).decode() if encrypted_phone else ''
                        },
                        'raw_text': raw_text[:2000],  # Limit raw text for speed
                        'processed_text': ai_ranker.clean_and_preprocess_text(raw_text),
                        'file_hash': file_hash,
                        'upload_timestamp': datetime.now().isoformat(),
                        'encrypted': True
                    }
                    
                    # Combine skills and experience for TF-IDF
                    combined_text = ' '.join(resume_data['skills']) + ' '
                    combined_text += ' '.join(resume_data['experience']['companies']) + ' '
                    combined_text += resume_data['processed_text']
                    resume_data['combined_text'] = combined_text
                    
                    processed_resumes.append(resume_data)
                    
                except Exception as e:
                    print(f"Error processing file {file.filename}: {e}")
                    continue
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
        
        return jsonify({
            'message': f'Successfully processed {len(processed_resumes)} resumes',
            'count': len(processed_resumes)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def validate_job_description(job_description):
    """Validate job description"""
    # Check minimum word count
    words = job_description.strip().split()
    if len(words) < 5:
        return {'valid': False, 'error': 'Job description must contain at least 5 words'}
    
    # Check if it's in English (basic check)
    english_words = 0
    total_words = len(words)
    
    # Common English words for basic validation
    common_english = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an',
        'experience', 'skills', 'required', 'preferred', 'years', 'work',
        'development', 'software', 'engineer', 'developer', 'programming',
        'python', 'java', 'javascript', 'react', 'node', 'database', 'sql'
    }
    
    for word in words:
        if word.lower().strip('.,!?;:') in common_english:
            english_words += 1
    
    # If less than 30% of words are English, consider it non-English
    if total_words > 0 and (english_words / total_words) < 0.3:
        return {'valid': False, 'error': 'Job description should be in English'}
    
    return {'valid': True}

def extract_text_from_job_description_file(file):
    """Extract text from job description file (PDF or TXT)"""
    try:
        if file.filename.lower().endswith('.pdf'):
            # Create temporary file for PDF processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                file.save(tmp_file.name)
                tmp_file_path = tmp_file.name
            
            # Use the same PDF extraction method as resumes
            ai_ranker = AIResumeRanker()
            text = ai_ranker.extract_text_from_pdf(tmp_file_path)
            
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            return text
        elif file.filename.lower().endswith('.txt'):
            return file.read().decode('utf-8')
        else:
            return ""
    except Exception as e:
        print(f"Error extracting text from job description file: {e}")
        return ""

def validate_file_type(filename):
    """Validate file type"""
    allowed_extensions = ['.pdf', '.docx']
    file_ext = os.path.splitext(filename.lower())[1]
    
    if file_ext not in allowed_extensions:
        return {'valid': False, 'error': f'File type {file_ext} not supported. Please use PDF or DOCX files only.'}
    
    return {'valid': True}

@app.route('/api/process-job-description-file', methods=['POST'])
def process_job_description_file():
    """Process job description from uploaded file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Extract text from job description file
        job_description = extract_text_from_job_description_file(file)
        
        if not job_description.strip():
            return jsonify({'error': 'Could not extract text from file'}), 400
        
        # Validate job description
        validation = validate_job_description(job_description)
        if not validation['valid']:
            return jsonify({'error': validation['error']}), 400
        
        return jsonify({
            'job_description': job_description,
            'word_count': len(job_description.strip().split()),
            'message': 'Job description extracted successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-job-description', methods=['POST'])
def process_job_description():
    """Process job description and rank candidates"""
    global processed_resumes, ai_ranker, current_results
    
    try:
        data = request.get_json()
        job_description = data.get('job_description', '')
        
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        
        # Validate job description
        validation = validate_job_description(job_description)
        if not validation['valid']:
            return jsonify({'error': validation['error']}), 400
        
        if not processed_resumes:
            return jsonify({'error': 'No resumes uploaded'}), 400
        
        # Optimized TF-IDF processing with 2000 features as requested
        combined_texts = [resume['combined_text'] for resume in processed_resumes]
        
        # Optimized TF-IDF parameters maintaining 2000 features for quality
        ai_ranker.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,  # Keep at 2000 as requested
            stop_words='english',
            ngram_range=(1, 2),  # Keep bigrams for better quality
            min_df=1,
            max_df=0.95
        )
        
        ai_ranker.tfidf_matrix = ai_ranker.tfidf_vectorizer.fit_transform(combined_texts)
        # Process job description
        processed_jd = ai_ranker.clean_and_preprocess_text(job_description)
        jd_vector = ai_ranker.tfidf_vectorizer.transform([processed_jd])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(jd_vector, ai_ranker.tfidf_matrix).flatten()
        
        # Create results with decrypted data for display
        results = []
        for i, resume in enumerate(processed_resumes):
            # Decrypt sensitive data for display
            decrypted_resume = decrypt_resume_data(resume)
            
            # Ensure name is properly decrypted
            display_name = decrypted_resume.get('name', 'Unknown')
            if display_name == 'Unknown' or len(display_name) > 100:  # Check for encrypted data
                try:
                    # Try to decrypt the name directly
                    if resume.get('name'):
                        encrypted_name = base64.b64decode(resume['name'])
                        display_name = decrypt_sensitive_data(encrypted_name)
                except:
                    display_name = 'Unknown'
            
            # Ensure email is properly decrypted
            display_email = decrypted_resume.get('contact', {}).get('email', '')
            if len(display_email) > 100:  # Check for encrypted data
                try:
                    if resume.get('contact', {}).get('email'):
                        encrypted_email = base64.b64decode(resume['contact']['email'])
                        display_email = decrypt_sensitive_data(encrypted_email)
                except:
                    display_email = ''
            
            # Ensure phone is properly decrypted
            display_phone = decrypted_resume.get('contact', {}).get('phone', '')
            if len(display_phone) > 100:  # Check for encrypted data
                try:
                    if resume.get('contact', {}).get('phone'):
                        encrypted_phone = base64.b64decode(resume['contact']['phone'])
                        display_phone = decrypt_sensitive_data(encrypted_phone)
                except:
                    display_phone = ''
            
            result = {
                'id': i,
                'name': display_name,
                'filename': resume['filename'],
                'similarity_score': float(similarity_scores[i]),
                'skills': resume['skills'],
                'experience_years': resume['experience']['years'],
                'companies': resume['experience']['companies'],
                'email': display_email,
                'phone': display_phone,
                'hired': False,  # Default not hired
                'hiring_status': 'rejected'  # Default rejected
            }
            results.append(result)
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        current_results = results
        
        # Auto-train model if we have enough data
        if len(hiring_data) >= 3:  # Minimum 3 decisions for basic training
            try:
                advanced_analytics.hiring_data = hiring_data
                advanced_analytics.analyze_hiring_patterns()
                advanced_analytics.calculate_skill_weights()
                advanced_analytics.build_hiring_prediction_model()
            except Exception as e:
                print(f"Auto-training failed: {e}")
        
        # Return only top 10 candidates
        top_10_results = results[:10]
        return jsonify({
            'message': 'Job description processed successfully',
            'top_candidates': top_10_results,  # Return only top 10 candidates
            'total_candidates': len(results),
            'displayed_candidates': len(top_10_results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/filter-candidates', methods=['POST'])
def filter_candidates():
    """Filter candidates based on criteria"""
    global current_results
    
    try:
        data = request.get_json()
        min_score = data.get('min_score', 0.0)
        min_experience = data.get('min_experience', 0)
        search_term = data.get('search_term', '')
        
        filtered_results = current_results.copy()
        
        # Filter by score
        filtered_results = [r for r in filtered_results if r['similarity_score'] >= min_score]
        
        # Filter by experience
        filtered_results = [r for r in filtered_results if r['experience_years'] >= min_experience]
        
        # Search filter
        if search_term:
            search_term = search_term.lower()
            filtered_results = [r for r in filtered_results if 
                              search_term in r['name'].lower() or
                              any(search_term in skill.lower() for skill in r['skills']) or
                              any(search_term in company.lower() for company in r['companies'])]
        
        return jsonify({
            'filtered_candidates': filtered_results,
            'count': len(filtered_results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/candidate/<int:candidate_id>', methods=['GET'])
def get_candidate_details(candidate_id):
    """Get detailed information about a specific candidate"""
    global processed_resumes
    
    try:
        if candidate_id >= len(processed_resumes):
            return jsonify({'error': 'Candidate not found'}), 404
        
        resume = processed_resumes[candidate_id]
        
        return jsonify({
            'id': candidate_id,
            'name': resume['name'],
            'filename': resume['filename'],
            'skills': resume['skills'],
            'experience': resume['experience'],
            'contact': resume['contact'],
            'raw_text': resume['raw_text']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-candidate/<int:candidate_id>/<format>', methods=['GET'])
def download_candidate(candidate_id, format):
    """Download candidate resume in specified format"""
    global processed_resumes, current_results
    
    try:
        if candidate_id >= len(processed_resumes):
            return jsonify({'error': 'Candidate not found'}), 404
        
        resume = processed_resumes[candidate_id]
        
        if format == 'excel':
            # Create Excel file
            data = [{
                'Name': resume['name'],
                'Filename': resume['filename'],
                'Skills': ', '.join(resume['skills']),
                'Experience (Years)': resume['experience']['years'],
                'Companies': ', '.join(resume['experience']['companies']),
                'Email': resume['contact']['email'],
                'Phone': resume['contact']['phone']
            }]
            
            df = pd.DataFrame(data)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Candidate Details', index=False)
            
            output.seek(0)
            excel_data = output.getvalue()
            
            return send_file(
                io.BytesIO(excel_data),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f"{resume['name'].replace(' ', '_')}_resume.xlsx"
            )
        
        elif format == 'pdf':
            # Create PDF report
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                # Title
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=18,
                    spaceAfter=30,
                    alignment=1
                )
                story.append(Paragraph(f"Resume: {resume['name']}", title_style))
                story.append(Spacer(1, 12))
                
                # Details
                story.append(Paragraph(f"<b>Name:</b> {resume['name']}", styles['Normal']))
                story.append(Paragraph(f"<b>Email:</b> {resume['contact']['email']}", styles['Normal']))
                story.append(Paragraph(f"<b>Phone:</b> {resume['contact']['phone']}", styles['Normal']))
                story.append(Paragraph(f"<b>Experience:</b> {resume['experience']['years']} years", styles['Normal']))
                story.append(Paragraph(f"<b>Skills:</b> {', '.join(resume['skills'])}", styles['Normal']))
                story.append(Paragraph(f"<b>Companies:</b> {', '.join(resume['experience']['companies'])}", styles['Normal']))
                story.append(Spacer(1, 12))
                
                # Full text
                story.append(Paragraph("<b>Full Resume Text:</b>", styles['Heading2']))
                story.append(Paragraph(resume['raw_text'][:2000] + "..." if len(resume['raw_text']) > 2000 else resume['raw_text'], styles['Normal']))
                
                doc.build(story)
                buffer.seek(0)
                
                return send_file(
                    buffer,
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=f"{resume['name'].replace(' ', '_')}_resume.pdf"
                )
                
            except ImportError:
                return jsonify({'error': 'ReportLab not installed'}), 500
        
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-hiring-status', methods=['POST'])
def update_hiring_status():
    """Update hiring status for a candidate"""
    global current_results, hiring_data, advanced_analytics
    
    try:
        data = request.get_json()
        candidate_id = data.get('candidate_id')
        hired = data.get('hired', False)
        
        if candidate_id is None:
            return jsonify({'error': 'Candidate ID is required'}), 400
        
        # Update the candidate's hiring status
        for result in current_results:
            if result['id'] == candidate_id:
                result['hired'] = hired
                result['hiring_status'] = 'hired' if hired else 'rejected'
                
                # Add to hiring data for model retraining
                hiring_record = {
                    'candidate_id': candidate_id,
                    'name': result['name'],
                    'similarity_score': result['similarity_score'],
                    'experience_years': result['experience_years'],
                    'skills': result['skills'],
                    'hired': hired,
                    'hiring_status': 'hired' if hired else 'rejected',
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                # Remove existing record if any
                hiring_data = [h for h in hiring_data if h['candidate_id'] != candidate_id]
                hiring_data.append(hiring_record)
                
                # Auto-retrain model if we have enough data
                if len(hiring_data) >= 5:  # Minimum 5 decisions for retraining
                    try:
                        advanced_analytics.hiring_data = hiring_data
                        advanced_analytics.analyze_hiring_patterns()
                        advanced_analytics.calculate_skill_weights()
                        advanced_analytics.build_hiring_prediction_model()
                    except Exception as e:
                        print(f"Auto-retraining failed: {e}")
                
                break
        
        return jsonify({'message': 'Hiring status updated successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain-model', methods=['POST'])
def retrain_model():
    """Retrain the model based on hiring decisions"""
    global ai_ranker, hiring_data
    
    try:
        if not hiring_data:
            return jsonify({'message': 'No hiring data available for retraining'})
        
        # Save hiring data for analysis
        with open('hiring_data.json', 'w', encoding='utf-8') as f:
            json.dump(hiring_data, f, indent=2, ensure_ascii=False)
        
        # Here you could implement more sophisticated retraining logic
        # For now, we'll just save the data and acknowledge the retraining
        
        return jsonify({
            'message': 'Model retraining data saved successfully',
            'hiring_records': len(hiring_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hiring-data', methods=['GET'])
def get_hiring_data():
    """Get hiring data for analysis"""
    global hiring_data
    
    return jsonify({
        'hiring_data': hiring_data,
        'total_records': len(hiring_data)
    })

@app.route('/api/analyze-patterns', methods=['POST'])
def analyze_hiring_patterns():
    """Analyze hiring patterns and skill success rates"""
    global advanced_analytics, hiring_data
    
    try:
        # Load hiring data into analytics
        advanced_analytics.hiring_data = hiring_data
        
        if not hiring_data:
            return jsonify({'error': 'No hiring data available for analysis'}), 400
        
        # Run pattern analysis
        pattern_analysis = advanced_analytics.analyze_hiring_patterns()
        
        if 'error' in pattern_analysis:
            return jsonify(pattern_analysis), 400
        
        # Calculate skill weights
        skill_weights = advanced_analytics.calculate_skill_weights()
        
        # Get clustering comparison if we have enough data
        clustering_info = {}
        if len(processed_resumes) >= 10:
            try:
                clustering_data = []
                for i, resume in enumerate(processed_resumes):
                    clustering_data.append({
                        'candidate_id': i,
                        'similarity_score': 0.5,  # Default
                        'experience_years': resume['experience']['years'],
                        'skills': resume['skills']
                    })
                clustering_comparison = advanced_analytics.compare_clustering_algorithms(clustering_data)
                clustering_info = clustering_comparison
            except Exception as e:
                clustering_info = {'error': str(e)}
        
        return jsonify({
            'pattern_analysis': pattern_analysis,
            'skill_weights': skill_weights,
            'clustering_info': clustering_info,
            'message': 'Pattern analysis completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/build-prediction-model', methods=['POST'])
def build_prediction_model():
    """Build ML model for hiring prediction"""
    global advanced_analytics, hiring_data
    
    try:
        # Load hiring data into analytics
        advanced_analytics.hiring_data = hiring_data
        
        if not hiring_data:
            return jsonify({'error': 'No hiring data available for model training'}), 400
        
        # Build prediction model
        model_results = advanced_analytics.build_hiring_prediction_model()
        
        if 'error' in model_results:
            return jsonify(model_results), 400
        
        return jsonify({
            'model_performance': model_results,
            'message': 'Prediction model built successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-hiring', methods=['POST'])
def predict_hiring_success():
    """Predict hiring success for a candidate"""
    global advanced_analytics
    
    try:
        data = request.get_json()
        candidate_data = data.get('candidate_data', {})
        
        if not candidate_data:
            return jsonify({'error': 'Candidate data is required'}), 400
        
        # Make prediction
        prediction = advanced_analytics.predict_hiring_success(candidate_data)
        
        if 'error' in prediction:
            return jsonify(prediction), 400
        
        return jsonify({
            'prediction': prediction,
            'message': 'Prediction completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance-report', methods=['GET'])
def get_performance_report():
    """Get comprehensive performance analysis report"""
    global advanced_analytics, hiring_data
    
    try:
        # Load hiring data into analytics
        advanced_analytics.hiring_data = hiring_data
        
        if not hiring_data:
            return jsonify({'error': 'No hiring data available for analysis'}), 400
        
        # Run full analysis
        advanced_analytics.analyze_hiring_patterns()
        advanced_analytics.calculate_skill_weights()
        advanced_analytics.build_hiring_prediction_model()
        
        # Generate comprehensive report
        report = advanced_analytics.generate_performance_report()
        
        return jsonify({
            'report': report,
            'message': 'Performance report generated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-similarity-weights', methods=['POST'])
def update_similarity_weights():
    """Update similarity calculation weights based on learned patterns"""
    global ai_ranker, advanced_analytics
    
    try:
        # Get current base weights (you might want to store these)
        base_weights = {}  # This should be your current TF-IDF weights
        
        # Update weights based on learned patterns
        updated_weights = advanced_analytics.update_similarity_weights(base_weights)
        
        return jsonify({
            'updated_weights': updated_weights,
            'message': 'Similarity weights updated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-visualizations', methods=['POST'])
def generate_visualizations():
    """Generate visualizations for the analysis"""
    global advanced_analytics, hiring_data
    
    try:
        # Load hiring data into analytics
        advanced_analytics.hiring_data = hiring_data
        
        if not hiring_data:
            return jsonify({'error': 'No hiring data available for visualization'}), 400
        
        # Run analysis first
        advanced_analytics.analyze_hiring_patterns()
        advanced_analytics.calculate_skill_weights()
        advanced_analytics.build_hiring_prediction_model()
        
        # Generate visualizations
        viz_result = advanced_analytics.create_visualizations()
        
        if 'error' in viz_result:
            return jsonify(viz_result), 400
        
        return jsonify({
            'visualization': viz_result,
            'message': 'Visualizations generated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare-clustering', methods=['POST'])
def compare_clustering_algorithms():
    """Compare different clustering algorithms"""
    global advanced_analytics, processed_resumes
    
    try:
        if not processed_resumes:
            return jsonify({'error': 'No processed resumes available for clustering'}), 400
        
        # Convert processed resumes to format suitable for clustering
        clustering_data = []
        for i, resume in enumerate(processed_resumes):
            clustering_data.append({
                'candidate_id': i,
                'similarity_score': 0.5,  # Default similarity score
                'experience_years': resume['experience']['years'],
                'skills': resume['skills']
            })
        
        # Compare clustering algorithms
        comparison = advanced_analytics.compare_clustering_algorithms(clustering_data)
        
        if 'error' in comparison:
            return jsonify(comparison), 400
        
        return jsonify({
            'clustering_comparison': comparison,
            'message': 'Clustering algorithms compared successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-top-candidates/<format>', methods=['GET'])
@require_auth
def download_top_candidates(format):
    """Download top 10 candidates in specified format"""
    global current_results
    
    try:
        if not current_results:
            return jsonify({'error': 'No candidates available'}), 400
        
        # Get top 10 candidates
        top_10 = current_results[:10]
        
        if format == 'excel':
            # Create Excel file with top 10 candidates
            data = []
            for i, candidate in enumerate(top_10, 1):
                data.append({
                    'Rank': i,
                    'Name': candidate['name'],
                    'Filename': candidate['filename'],
                    'Match Score': round(candidate['similarity_score'], 4),
                    'Skills': ', '.join(candidate['skills'][:10]),  # Limit skills for Excel
                    'Experience (Years)': candidate['experience_years'],
                    'Companies': ', '.join(candidate['companies'][:3]),  # Limit companies
                    'Email': candidate['email'],
                    'Phone': candidate['phone'],
                    'Hiring Status': candidate.get('hiring_status', 'rejected')
                })
            
            df = pd.DataFrame(data)
            
            # Try Excel generation with fallbacks
            try:
                output = io.BytesIO()
                
                # Try openpyxl first
                try:
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Top 10 Candidates', index=False)
                    
                    output.seek(0)
                    excel_data = output.getvalue()
                    
                    return send_file(
                        io.BytesIO(excel_data),
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        as_attachment=True,
                        download_name="top_10_candidates.xlsx"
                    )
                    
                except ImportError:
                    print("openpyxl not available, trying xlsxwriter...")
                    # Fallback to xlsxwriter
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='Top 10 Candidates', index=False)
                    
                    output.seek(0)
                    excel_data = output.getvalue()
                    
                    return send_file(
                        io.BytesIO(excel_data),
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        as_attachment=True,
                        download_name="top_10_candidates.xlsx"
                    )
                    
            except ImportError:
                print("Neither openpyxl nor xlsxwriter available, falling back to CSV...")
                # Final fallback - create CSV instead of Excel
                csv_output = io.StringIO()
                df.to_csv(csv_output, index=False)
                csv_data = csv_output.getvalue().encode('utf-8')
                csv_output.close()
                
                return send_file(
                    io.BytesIO(csv_data),
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name="top_10_candidates.csv"
                )
        
        elif format == 'pdf':
            # Create PDF report with top 10 candidates
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib import colors
                
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                # Title
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=18,
                    spaceAfter=30,
                    alignment=1
                )
                story.append(Paragraph("Top 10 Candidates Report", title_style))
                story.append(Spacer(1, 12))
                
                # Create table data
                table_data = [['Rank', 'Name', 'Match Score', 'Experience', 'Email', 'Status']]
                for i, candidate in enumerate(top_10, 1):
                    table_data.append([
                        str(i),
                        candidate['name'],
                        f"{candidate['similarity_score']:.3f}",
                        f"{candidate['experience_years']} years",
                        candidate['email'],
                        candidate.get('hiring_status', 'rejected')
                    ])
                
                # Create table
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(table)
                story.append(Spacer(1, 20))
                
                # Add detailed information for each candidate
                for i, candidate in enumerate(top_10, 1):
                    story.append(Paragraph(f"<b>{i}. {candidate['name']}</b>", styles['Heading2']))
                    story.append(Paragraph(f"<b>Match Score:</b> {candidate['similarity_score']:.4f}", styles['Normal']))
                    story.append(Paragraph(f"<b>Experience:</b> {candidate['experience_years']} years", styles['Normal']))
                    story.append(Paragraph(f"<b>Skills:</b> {', '.join(candidate['skills'][:10])}", styles['Normal']))
                    story.append(Paragraph(f"<b>Companies:</b> {', '.join(candidate['companies'][:3])}", styles['Normal']))
                    story.append(Paragraph(f"<b>Contact:</b> {candidate['email']} | {candidate['phone']}", styles['Normal']))
                    story.append(Spacer(1, 12))
                
                doc.build(story)
                buffer.seek(0)
                
                return send_file(
                    buffer,
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name="top_10_candidates.pdf"
                )
                
            except ImportError:
                return jsonify({'error': 'ReportLab not installed'}), 500
        
        else:
            return jsonify({'error': 'Invalid format. Use excel or pdf'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
