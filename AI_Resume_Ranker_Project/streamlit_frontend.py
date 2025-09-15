# This script runs the Streamlit front-end for the AI Resume Ranker application.
# It handles user authentication, file uploads, interaction with the Flask
# backend API, and displays the ranked candidate results.
#
# To run: streamlit run main_app.py

import streamlit as st
import requests
import time

#  CONFIGURATION & CONSTANTS

# The base URL for the backend Flask API.
FLASK_URL = "http://localhost:5000"

# Configure the Streamlit page. This should be the first Streamlit command.
st.set_page_config(
    page_title="AI Resume Ranker",
    layout="wide",
    initial_sidebar_state="expanded"
)


#  STYLING

def load_css():
    """Injects custom CSS for a polished and modern UI design."""
    st.markdown("""
    <style>
        /* Import a clean, modern font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        /* --- General Styles --- */
        body { 
            font-family: 'Poppins', sans-serif; 
            color: #4A5568; /* A softer, more professional dark gray */
        }
        .main .block-container { 
            padding: 2rem 4rem; /* Add more breathing room */
        }

        /* --- Headers --- */
        .main-header { 
            font-size: 3rem; 
            font-weight: 700; 
            text-align: center; 
            margin-bottom: 0.5rem; 
            /* A cool gradient effect for the main title */
            background: -webkit-linear-gradient(45deg, #3b82f6, #1e3a8a); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }
        .sub-header { 
            text-align: center; 
            font-size: 1.1rem; 
            color: #64748B; 
            margin-bottom: 2.5rem; 
        }
        .section-header { 
            font-size: 1.5rem; 
            color: #1e3a8a; 
            margin-top: 2rem; 
            margin-bottom: 1.5rem; 
            border-bottom: 2px solid #3b82f6; 
            padding-bottom: 0.5rem; 
        }

        /* --- UI Components & Cards --- */
        .info-box { 
            background-color: #f0f9ff; 
            border-left: 5px solid #3b82f6; 
            border-radius: 12px; 
            padding: 1.5rem; 
            margin: 1rem 0; 
        }
        .candidate-card { 
            background: #ffffff; 
            border: 1px solid #e2e8f0; 
            border-left: 5px solid #3b82f6; 
            border-radius: 12px; 
            padding: 1.5rem; 
            margin-bottom: 1.5rem; 
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
            transition: all 0.3s ease; 
        }
        .candidate-card:hover { 
            transform: translateY(-5px); 
            box-shadow: 0 10px 20px rgba(59, 130, 246, 0.1); 
            border-color: #93c5fd; 
        }
        /* Style for candidates marked as 'Hired' */
        .hired-candidate { 
            border-left: 5px solid #34d399; /* Green accent */
            background-color: #f0fdf4; 
        }
        .feature-card { 
            background-color: #ffffff; 
            border-radius: 12px; 
            padding: 1.5rem; 
            text-align: center; 
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05); 
            border: 1px solid #e2e8f0; 
            height: 100%; 
        }
        .feature-card h3 { 
            color: #1e3a8a; 
            font-size: 1.2rem; 
        }
        .stButton>button { 
            border-radius: 8px; 
            font-weight: 600; 
            transition: all 0.2s ease; 
            background-color: #3b82f6; 
            color: white; 
            border: none; 
        }
        .stButton>button:hover { 
            background-color: #1e3a8a; 
            transform: scale(1.03); 
        }
    </style>
    """, unsafe_allow_html=True)


#  BACKEND API COMMUNICATION

def api_request(method, endpoint, **kwargs):
    """
    A centralized function to handle all API requests to the backend.
    It manages session tokens, error handling, and retries.

    Args:
        method (str): The HTTP method ('get', 'post').
        endpoint (str): The API endpoint path (e.g., '/api/health').
        **kwargs: Additional arguments passed to the requests call (e.g., json, files, headers).

    Returns:
        dict or bytes: The response from the backend. JSON for most endpoints, bytes for file downloads.
                       Returns a dictionary with an 'error' key on failure.
    """
    # Add session token to headers if it exists
    headers = kwargs.get('headers', {})
    if 'session_token' in st.session_state and st.session_state.session_token:
        headers['X-Session-Token'] = st.session_state.session_token
    kwargs['headers'] = headers

    # Define retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.request(method, f"{FLASK_URL}{endpoint}", **kwargs)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            # Return raw content for downloads, otherwise return JSON
            if "download" in endpoint:
                return response.content
            return response.json()

        except requests.exceptions.HTTPError as e:
            error_msg = e.response.json().get('error', f"HTTP Error: {e.response.status_code}")
            return {"error": error_msg}
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                return {"error": f"Connection to backend failed after {max_retries} attempts."}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {str(e)}"}


#  UI RENDERING FUNCTIONS

def render_login_page():
    """Displays the login screen and handles the authentication process."""
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="centered-login">', unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">AI Resume Ranker</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Find the best talent, faster.</p>', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header" style="border:none; margin-top: 2rem;">Authentication Required</h2>',
                    unsafe_allow_html=True)
        st.info("Please login to access the system.")

        if st.button("Login (Demo)"):
            with st.spinner("Logging in..."):
                login_data = {"username": "admin", "password": "admin123"}
                response = api_request("post", "/api/auth/login", json=login_data)

                if "error" not in response and "session_token" in response:
                    st.session_state.session_token = response['session_token']
                    st.rerun()
                else:
                    st.error(f"Login failed: {response.get('error', 'Unknown error')}")
        st.markdown('</div>', unsafe_allow_html=True)


def render_sidebar():
    """
    Renders the sidebar for file uploads and processing controls.
    Handles the logic for initiating the analysis process.
    """
    with st.sidebar:
        if st.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

        st.markdown("---")
        st.markdown('<h2 class="section-header" style="margin-top: 0; border: none;">Controls</h2>',
                    unsafe_allow_html=True)

        # Step 1: Resume Upload
        st.subheader("1. Upload Resumes")
        uploaded_resumes = st.file_uploader("Choose resume files", type=['pdf', 'docx'], accept_multiple_files=True)

        # Step 2: Job Description Input (EITHER/OR logic)
        st.subheader("2. Add Job Description")
        input_method = st.radio(
            "Choose input method:",
            ("Paste Text", "Upload a File"),
            horizontal=True
        )

        job_desc_text = ""
        job_desc_file = None

        if input_method == "Paste Text":
            job_desc_text = st.text_area("Paste job description here:", height=150, label_visibility="collapsed")
        else:
            job_desc_file = st.file_uploader("Upload job description file:", type=['pdf', 'txt'],
                                             label_visibility="collapsed")

        # Step 3: Processing Button
        st.subheader("3. Process")
        if st.button("Process Files"):
            if not uploaded_resumes or (not job_desc_text and not job_desc_file):
                st.error("Please upload resumes and provide a job description.")
                return

            with st.spinner("Analyzing resumes and job description... This may take a moment."):
                process_data(uploaded_resumes, job_desc_text, job_desc_file)


def render_welcome_screen():
    """Displays the initial welcome message and feature cards."""
    st.markdown("""
    <div class="info-box">
        <h3>Welcome to the AI Resume Ranker! 🚀</h3>
        <p>Get started by uploading resumes and a job description in the sidebar on the left. The system will automatically analyze, rank, and present the most qualified candidates for your review.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-header">Core Features</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>📄 Data Extraction & Analysis</h3>
                <p>Automatically extracts key data from resumes (PDF, DOCX) like skills, experience years, and past companies using advanced NLP.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>💡 Intelligent Matching & Ranking</h3>
                <p>Calculates a precise match score between each resume and the job description using TF-IDF, then ranks candidates accordingly.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="feature-card">
                <h3>📈 Hiring Pattern Analysis</h3>
                <p>Builds ML models to analyze past hiring decisions, identify high-demand skills, and predict future candidate success.</p>
            </div>
        """, unsafe_allow_html=True)


def render_results_page():
    """
    Displays the filtering options and the list of ranked candidates.
    Handles user interactions like filtering, viewing details, and marking hires.
    """
    st.markdown('<h2 class="section-header">Candidate Analysis</h2>', unsafe_allow_html=True)

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        min_score = c1.slider("Minimum Match Score", 0.0, 1.0, 0.0, 0.01)
        min_experience = c2.slider("Minimum Experience (Years)", 0, 20, 0)
        search_term = c3.text_input("Search by keyword (e.g., 'Python', 'AWS')")

        if st.button("Apply Filters"):
            filter_data = {"min_score": min_score, "min_experience": min_experience, "search_term": search_term}
            result = api_request("post", "/api/filter-candidates", json=filter_data)
            if "error" in result:
                st.error(f"Filter Error: {result['error']}")
            else:
                st.session_state.filtered_candidates = result['filtered_candidates']

        d1, d2 = st.columns(2)
        if d1.button("Download Top 10 as Excel", use_container_width=True):
            handle_download('excel')
        if d2.button("Download Top 10 as PDF", use_container_width=True):
            handle_download('pdf')

    st.markdown("---")
    st.markdown(
        f'<h3 class="section-header" style="border:none;">Top Candidates ({len(st.session_state.filtered_candidates)} showing)</h3>',
        unsafe_allow_html=True)

    if not st.session_state.filtered_candidates:
        st.warning("No candidates match the current filters.")
    else:
        for i, candidate in enumerate(st.session_state.filtered_candidates, 1):
            render_candidate_card(candidate, i)


def render_candidate_card(candidate, index):
    """
    Renders a single card for a candidate with their details.

    Args:
        candidate (dict): The dictionary containing candidate data.
        index (int): The rank/number of the candidate in the list.
    """
    card_class = "candidate-card hired-candidate" if candidate.get('hired', False) else "candidate-card"
    with st.container():
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.markdown(f"**{index}. {candidate['name']}**")
            st.write(f"**File:** {candidate['filename']}")
            st.write(f"**Match Score:** {candidate['similarity_score']:.2%}")
            st.write(f"**Experience:** {candidate.get('experience_years', 'N/A')} years")

        with col2:
            st.write(f"**Top Skills:** {', '.join(candidate['skills'][:5])}")
            if len(candidate['skills']) > 5:
                st.write(f"_{len(candidate['skills']) - 5} more..._")

            st.write(f"**Companies:** {', '.join(candidate['companies'][:2])}")
            if len(candidate['companies']) > 2:
                st.write(f"_{len(candidate['companies']) - 2} more..._")

        with col3:
            if st.button("View CV", key=f"view_{candidate['id']}"):
                details = api_request("get", f"/api/candidate/{candidate['id']}")
                if "error" not in details:
                    st.session_state[f"details_{candidate['id']}"] = details
                else:
                    st.error(f"Could not fetch details: {details['error']}")

            is_hired = st.checkbox("✅ Hired", value=candidate.get('hired', False), key=f"hired_{candidate['id']}")
            if is_hired != candidate.get('hired', False):
                api_request("post", "/api/update-hiring-status",
                            json={"candidate_id": candidate['id'], "hired": is_hired})
                candidate['hired'] = is_hired

        st.markdown('</div>', unsafe_allow_html=True)

        if f"details_{candidate['id']}" in st.session_state:
            render_cv_details(candidate)


def render_cv_details(candidate):
    """Renders the expandable section with full resume text and details."""
    details = st.session_state[f"details_{candidate['id']}"]
    with st.expander(f"Full CV Details for {candidate['name']}", expanded=True):
        st.write(f"**Name:** {details.get('name', 'N/A')}")
        st.write(f"**Email:** {details.get('contact', {}).get('email', 'N/A')}")
        st.write(f"**Phone:** {details.get('contact', {}).get('phone', 'N/A')}")
        st.write(f"**Experience:** {details.get('experience', {}).get('years', 'N/A')} years")
        st.write(f"**Companies:** {', '.join(details.get('experience', {}).get('companies', []))}")
        st.write(f"**All Skills:** {', '.join(details.get('skills', []))}")
        st.markdown("---")
        st.text_area("Full Resume Text", details.get('raw_text', 'No text available.'), height=300,
                     key=f"text_{candidate['id']}")

        if st.button("Close Details", key=f"close_{candidate['id']}"):
            del st.session_state[f"details_{candidate['id']}"]
            st.rerun()


#  DATA PROCESSING & LOGIC

def initialize_session_state():
    """Initializes keys in Streamlit's session state if they don't exist."""
    if 'session_token' not in st.session_state:
        st.session_state.session_token = None
    if 'candidates' not in st.session_state:
        st.session_state.candidates = []
    if 'filtered_candidates' not in st.session_state:
        st.session_state.filtered_candidates = []


def process_data(uploaded_resumes, job_desc_text, job_desc_file):
    """
    Orchestrates the multi-step process of uploading resumes and the job
    description to the backend for analysis.
    """
    job_description = job_desc_text
    if job_desc_file:
        if job_desc_file.type == "text/plain":
            job_description = job_desc_file.read().decode('utf-8')
        elif job_desc_file.type == "application/pdf":
            files = {'file': (job_desc_file.name, job_desc_file.getvalue(), job_desc_file.type)}
            result = api_request("post", "/api/process-job-description-file", files=files, timeout=60)
            if "error" in result:
                st.error(f"PDF Parsing Error: {result['error']}")
                return
            job_description = result['job_description']

    if len(job_description.strip().split()) < 5:
        st.error("Job description must be at least 5 words.")
        return

    num_files = len(uploaded_resumes)
    timeout = 180 + (num_files * 2)
    files_to_upload = [('files', (f.name, f.getvalue(), f.type)) for f in uploaded_resumes]
    upload_result = api_request("post", "/api/upload-resumes", files=files_to_upload, timeout=timeout)
    if "error" in upload_result:
        st.error(f"Upload Error: {upload_result['error']}")
        return

    jd_data = {"job_description": job_description}
    process_result = api_request("post", "/api/process-job-description", json=jd_data, timeout=120)
    if "error" in process_result:
        st.error(f"Processing Error: {process_result['error']}")
    else:
        st.session_state.candidates = process_result.get('top_candidates', [])
        st.session_state.filtered_candidates = process_result.get('top_candidates', [])
        st.success(f"Successfully processed {process_result.get('total_candidates', 0)} candidates!")


def handle_download(file_format):
    """
    Requests a file from the backend and provides it as a download link.

    Args:
        file_format (str): 'excel' or 'pdf'.
    """
    with st.spinner(f"Generating {file_format.upper()} file..."):
        content = api_request("get", f"/api/download-top-candidates/{file_format}")

        if isinstance(content, dict) and "error" in content:
            st.error(f"Download Error: {content['error']}")
        elif content:
            file_name = f"top_10_candidates.{'xlsx' if file_format == 'excel' else 'pdf'}"
            st.download_button(f"⬇️ Download {file_format.upper()}", content, file_name)
        else:
            st.error("Failed to download file. No content received.")


#  MAIN APPLICATION EXECUTION

def main():
    """The main function that orchestrates the entire application flow."""
    load_css()
    initialize_session_state()

    if not st.session_state.session_token:
        render_login_page()
        return

    st.markdown('<h1 class="main-header">AI Resume Ranker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find the best talent, faster.</p>', unsafe_allow_html=True)

    if not api_request("get", "/api/health"):
        st.error("Cannot connect to backend. Please ensure the server is running.")
        return

    render_sidebar()

    if not st.session_state.candidates:
        render_welcome_screen()
    else:
        render_results_page()


if __name__ == "__main__":
    main()