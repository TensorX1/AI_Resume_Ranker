# TensorX - AI Resume Ranker

A comprehensive AI-powered resume analysis and candidate ranking system that automatically processes resumes, extracts key information, and ranks candidates based on job requirements using advanced machine learning techniques.

##  Features

### Core Functionality
- Multi-format Resume Processing: Supports PDF and DOCX files
- Intelligent Data Extraction: Automatically extracts names, skills, experience, education, and contact information
- Advanced NLP Processing: Uses spaCy, NLTK, and TF-IDF for text analysis
- Smart Candidate Ranking: Ranks candidates based on similarity scores using cosine similarity
- Real-time Processing: Handles large batches of resumes efficiently

### Advanced Analytics
- Hiring Pattern Analysis: Analyzes past hiring decisions to identify success factors
- Machine Learning Models: Builds predictive models using Logistic Regression, Random Forest, and Gradient Boosting
- Skill Weight Calculation: Dynamically calculates skill importance based on hiring success rates
- Clustering Analysis: Groups candidates using K-Means and DBSCAN algorithms
- Performance Visualization: Generates comprehensive analytics dashboards

### Security & Privacy
- Data Encryption: Encrypts sensitive personal information using Fernet encryption
- Session Management: Secure authentication with JWT tokens
- File Integrity: SHA-256 hashing for file content verification
- Access Control: Role-based authentication system

### User Interface
- Modern Web Interface: Built with Streamlit for intuitive user experience
- Responsive Design: Clean, professional UI with custom CSS styling
- Real-time Updates: Live progress tracking and status updates
- Export Capabilities: Download results in Excel and PDF formats

##  Architecture

The system follows a microservices architecture with three main components:

### 1. Flask Backend (`flask_backend.py`)
- RESTful API endpoints for all operations
- AI resume processing and ranking engine
- Advanced analytics and machine learning models
- Security and authentication management
- File processing and data extraction

### 2. Streamlit Frontend (`streamlit_frontend.py`)
- Interactive web interface
- File upload and job description input
- Real-time candidate ranking display
- Filtering and search capabilities
- Export functionality

### 3. Advanced Analytics (`advanced_analytics.py`)
- Machine learning model training and evaluation
- Hiring pattern analysis
- Skill weight calculation
- Clustering algorithms
- Performance reporting and visualization

##  Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended for large datasets
- Internet connection for initial setup

##  Installation

### 1. Clone the Repository
```bash
git clone <https://github.com/TensorX1/AI_Resume_Ranker/tree/main>
cd TensorX
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Download NLTK Data (Automatic)
The system automatically downloads required NLTK data on first run.

##  Quick Start

### 1. Start the Backend Server
```bash
python flask_backend.py
```
The Flask server will start on `http://localhost:5000`

### 2. Launch the Frontend
```bash
streamlit run streamlit_frontend.py
```
The Streamlit app will open in your browser at `http://localhost:8501`

### 3. Login
- Username: `admin`
- Password: `admin123`

##  Usage Guide

### Step 1: Upload Resumes
1. Click "Choose resume files" in the sidebar
2. Select multiple PDF or DOCX files
3. The system will automatically process and extract information

### Step 2: Add Job Description
Choose one of two methods:
- Paste Text: Directly paste the job description
- Upload File: Upload a PDF or TXT file containing the job description

### Step 3: Process and Rank
1. Click "Process Files" to start analysis
2. The system will:
   - Extract text from all resumes
   - Calculate similarity scores using TF-IDF
   - Rank candidates by match quality
   - Display top 10 candidates

### Step 4: Review and Filter
- Use filters to narrow down candidates by:
  - Minimum match score
  - Minimum experience years
  - Keyword search
- View detailed candidate information
- Mark candidates as hired/rejected

### Step 5: Export Results
- Download top 10 candidates as Excel or PDF
- Export includes all candidate details and rankings

##  API Endpoints

### Authentication
- `POST /api/auth/login` - User authentication
- `POST /api/auth/logout` - User logout

### Resume Processing
- `POST /api/upload-resumes` - Upload and process resume files
- `POST /api/process-job-description` - Process job description and rank candidates
- `POST /api/process-job-description-file` - Process job description from file

### Candidate Management
- `GET /api/candidate/<id>` - Get detailed candidate information
- `POST /api/filter-candidates` - Filter candidates by criteria
- `POST /api/update-hiring-status` - Update hiring status

### Analytics
- `POST /api/analyze-patterns` - Analyze hiring patterns
- `POST /api/build-prediction-model` - Build ML prediction model
- `POST /api/predict-hiring` - Predict hiring success for candidate
- `GET /api/performance-report` - Get comprehensive performance report

### Export
- `GET /api/download-top-candidates/<format>` - Download top candidates (excel/pdf)

##  Machine Learning Features

### Text Processing
- TF-IDF Vectorization: Converts text to numerical features
- N-gram Analysis: Uses unigrams and bigrams for better context
- Stop Word Removal: Filters common words for better analysis
- Text Preprocessing: Cleans and normalizes text data

### Similarity Calculation
- Cosine Similarity: Measures similarity between job description and resumes
- Feature Engineering: Combines skills, experience, and text features
- Dynamic Weighting: Adjusts weights based on hiring success patterns

### Predictive Models
- Logistic Regression: Fast, interpretable baseline model
- Random Forest: Handles non-linear relationships and feature interactions
- Gradient Boosting: High-performance ensemble method
- Model Selection: Automatically selects best model based on F1-score

### Clustering
- K-Means: Groups similar candidates for batch processing
- DBSCAN: Identifies candidate clusters with noise detection
- Automatic Selection: Chooses optimal algorithm based on data characteristics

##  Analytics Dashboard

The system provides comprehensive analytics including:

### Hiring Patterns
- Overall hiring rates and trends
- Skill success rate analysis
- Experience level preferences
- Similarity score distributions

### Performance Metrics
- Model accuracy, precision, recall, and F1-scores
- Cross-validation results
- Feature importance rankings
- Prediction confidence intervals

### Visualizations
- Skill success rate bar charts
- Experience distribution histograms
- Similarity score comparisons
- Model performance comparisons

##  Security Features

### Data Protection
- Encryption: All sensitive data encrypted using Fernet
- Secure Storage: Encrypted storage of personal information
- Access Control: Session-based authentication
- File Integrity: SHA-256 checksums for uploaded files

### Privacy Compliance
- Data Minimization: Only extracts necessary information
- Secure Transmission: HTTPS-ready architecture
- Audit Trail: Logs all data processing activities
- Data Retention: Configurable data retention policies

##  Performance Optimization

### Scalability
- Batch Processing: Handles large resume datasets efficiently
- Memory Management: Optimized for processing 288+ resumes
- Concurrent Processing: Multi-threaded file processing
- Caching: Intelligent caching of processed data

### Speed Optimizations
- Lazy Loading: Loads data only when needed
- Indexing: Fast text search and filtering
- Compression: Efficient data storage and transmission
- Background Processing: Non-blocking operations

##  Configuration

### Environment Variables
```bash
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key
```

### Performance Settings
- `MAX_CONTENT_LENGTH`: Maximum file size (default: 1GB)
- `UPLOAD_TIMEOUT`: Upload timeout (default: 10 minutes)
- `TFIDF_MAX_FEATURES`: Maximum TF-IDF features (default: 2000)

##  Project Structure

```
TensorX/
├── flask_backend.py          # Main Flask API server
├── streamlit_frontend.py     # Streamlit web interface
├── advanced_analytics.py     # ML analytics and models
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── Datasets/
│   └── Resumes/             # Sample resume files
└── __pycache__/             # Python cache files
```


##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Troubleshooting

### Common Issues

1. spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

2. NLTK Data Missing
The system automatically downloads required NLTK data on first run.

3. Memory Issues with Large Datasets
- Reduce batch size in `flask_backend.py`
- Increase system RAM
- Process files in smaller batches

4. File Upload Errors
- Check file format (PDF/DOCX only)
- Verify file size (max 1GB)
- Ensure files are not corrupted

### Performance Tips
- Use SSD storage for faster file processing
- Allocate at least 4GB RAM for large datasets
- Close other applications to free up memory
- Use batch processing for 100+ resumes


##  Future Enhancements

- [ ] Multi-language support (Arabic, Spanish, etc.)
- [ ] Advanced NLP models (BERT, GPT)
- [ ] Real-time collaboration features
- [ ] Mobile app interface
- [ ] Integration with job boards
- [ ] Advanced reporting and dashboards
- [ ] Machine learning model versioning
- [ ] Automated candidate outreach

---
