import streamlit as st
import pdfplumber
import docx2txt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from datetime import datetime
import io
import base64
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up the page
st.set_page_config(
    page_title="Automated Resume Relevance Check System",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .score-high {
        color: #4CAF50;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .score-medium {
        color: #FF9800;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .score-low {
        color: #F44336;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .missing-item {
        color: #F44336;
    }
    .present-item {
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = ""

class ResumeEvaluator:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.skill_keywords = [
            'python', 'java', 'sql', 'machine learning', 'deep learning', 
            'tensorflow', 'pytorch', 'scikit-learn', 'nlp', 'computer vision',
            'data analysis', 'tableau', 'power bi', 'excel', 'r', 'spark', 
            'hadoop', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 
            'jenkins', 'ci/cd', 'agile', 'scrum', 'project management',
            'javascript', 'html', 'css', 'react', 'angular', 'vue', 'node',
            'flask', 'django', 'fastapi', 'mongodb', 'mysql', 'postgresql',
            'git', 'linux', 'rest api', 'graphql', 'microservices'
        ]
    
    def extract_text_from_pdf(self, pdf_file):
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
    def extract_text_from_docx(self, docx_file):
        return docx2txt.process(docx_file)
    
    def extract_text(self, file):
        if file.type == "application/pdf":
            return self.extract_text_from_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self.extract_text_from_docx(file)
        return ""
    
    def clean_text(self, text):
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower()
        return text
    
    def extract_skills(self, text):
        found_skills = []
        for skill in self.skill_keywords:
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                found_skills.append(skill)
        return found_skills
    
    def extract_experience(self, text):
        # Simple experience extraction - looking for years and experience patterns
        experience_patterns = [
            r'(\d+)\s*(?:\+)?\s*years?[\s\w]*experience',
            r'experience[\s\w]*(\d+)\s*(?:\+)?\s*years?',
            r'(\d+)\s*\+?\s*yr',
        ]
        
        for pattern in experience_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                return match.group(1)
        return "Not specified"
    
    def extract_education(self, text):
        education_keywords = [
            'bachelor', 'b\.?s\.?', 'b\.?a\.?', 'master', 'm\.?s\.?', 'm\.?a\.?', 
            'phd', 'doctorate', 'mba', 'b\.?tech', 'm\.?tech', 'b\.?e\.?', 'm\.?e\.?',
            'high school', 'diploma', 'associate', 'certificate'
        ]
        
        education = []
        for edu in education_keywords:
            if re.search(r'\b' + edu + r'\b', text):
                education.append(edu)
        return education if education else ["Not specified"]
    
    def hard_match_score(self, resume_text, jd_text):
        # Extract skills from both texts
        resume_skills = set(self.extract_skills(resume_text))
        jd_skills = set(self.extract_skills(jd_text))
        
        # Calculate skill match
        if jd_skills:
            skill_match = len(resume_skills.intersection(jd_skills)) / len(jd_skills)
        else:
            skill_match = 0
        
        # Check for experience match
        resume_exp = self.extract_experience(resume_text)
        jd_exp = self.extract_experience(jd_text)
        
        if jd_exp.isdigit() and resume_exp.isdigit():
            exp_match = min(1.0, int(resume_exp) / int(jd_exp))
        else:
            exp_match = 0.5  # Default value if not specified
        
        # Education match
        resume_edu = set(self.extract_education(resume_text))
        jd_edu = set(self.extract_education(jd_text))
        
        if jd_edu and "Not specified" not in jd_edu:
            edu_match = 1.0 if resume_edu.intersection(jd_edu) else 0.0
        else:
            edu_match = 0.7  # Default value if not specified in JD
        
        # Weighted score
        hard_score = 0.6 * skill_match + 0.3 * exp_match + 0.1 * edu_match
        return hard_score * 100, list(jd_skills - resume_skills)
    
    def soft_match_score(self, resume_text, jd_text):
        # TF-IDF based similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100
    
    def calculate_final_score(self, hard_score, soft_score):
        # Weighted combination
        return 0.6 * hard_score + 0.4 * soft_score
    
    def get_verdict(self, score):
        if score >= 75:
            return "High Suitability", "score-high"
        elif score >= 50:
            return "Medium Suitability", "score-medium"
        else:
            return "Low Suitability", "score-low"
    
    def evaluate_resume(self, resume_text, jd_text):
        # Clean texts
        clean_resume = self.clean_text(resume_text)
        clean_jd = self.clean_text(jd_text)
        
        # Calculate scores
        hard_score, missing_skills = self.hard_match_score(clean_resume, clean_jd)
        soft_score = self.soft_match_score(clean_resume, clean_jd)
        final_score = self.calculate_final_score(hard_score, soft_score)
        verdict, score_class = self.get_verdict(final_score)
        
        # Extract additional information
        experience = self.extract_experience(clean_resume)
        education = self.extract_education(clean_resume)
        skills = self.extract_skills(clean_resume)
        
        return {
            "hard_score": hard_score,
            "soft_score": soft_score,
            "final_score": final_score,
            "verdict": verdict,
            "score_class": score_class,
            "missing_skills": missing_skills,
            "experience": experience,
            "education": education,
            "skills": skills
        }

# Initialize evaluator
evaluator = ResumeEvaluator()

# Main app
st.markdown('<h1 class="main-header">Automated Resume Relevance Check System</h1>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Upload & Evaluate", "Results Dashboard", "How It Works"])

with tab1:
    st.markdown('<div class="sub-header">Upload Job Description and Resumes</div>', unsafe_allow_html=True)
    
    # Job Description Upload
    jd_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"], key="jd_uploader")
    
    if jd_file:
        jd_text = evaluator.extract_text(jd_file)
        st.session_state.jd_text = jd_text
        st.success("Job Description uploaded successfully!")
        
        with st.expander("View Job Description"):
            st.text(jd_text[:1000] + "..." if len(jd_text) > 1000 else jd_text)
    
    # Resume Upload
    if st.session_state.jd_text:
        resume_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], 
                                      accept_multiple_files=True, key="resume_uploader")
        
        if resume_files and st.button("Evaluate Resumes"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, resume_file in enumerate(resume_files):
                status_text.text(f"Processing {i+1}/{len(resume_files)}: {resume_file.name}")
                progress_bar.progress((i) / len(resume_files))
                
                # Extract text from resume
                resume_text = evaluator.extract_text(resume_file)
                
                # Evaluate resume
                result = evaluator.evaluate_resume(resume_text, st.session_state.jd_text)
                result['file_name'] = resume_file.name
                result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Store result
                st.session_state.evaluation_results.append(result)
                
                # Display individual result
                with st.expander(f"Result for {resume_file.name}"):
                    st.markdown(f"**Relevance Score:** <span class='{result['score_class']}'>{result['final_score']:.2f}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Verdict:** {result['verdict']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Hard Match Score:**")
                        st.markdown(f"{result['hard_score']:.2f}")
                        st.markdown("**Soft Match Score:**")
                        st.markdown(f"{result['soft_score']:.2f}")
                    
                    with col2:
                        st.markdown("**Experience:**")
                        st.markdown(f"{result['experience']} years")
                        st.markdown("**Education:**")
                        st.markdown(", ".join(result['education']))
                    
                    if result['missing_skills']:
                        st.markdown("**Missing Skills:**")
                        for skill in result['missing_skills']:
                            st.markdown(f"<div class='missing-item'>- {skill}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("**All required skills are present!**", help="Great job! The resume contains all the skills mentioned in the job description.")
            
            progress_bar.progress(1.0)
            status_text.text("Evaluation complete!")
            st.success(f"Processed {len(resume_files)} resumes!")

with tab2:
    st.markdown('<div class="sub-header">Evaluation Results Dashboard</div>', unsafe_allow_html=True)
    
    if not st.session_state.evaluation_results:
        st.info("No evaluation results yet. Upload and evaluate some resumes first.")
    else:
        # Create a dataframe for better visualization
        df_data = []
        for result in st.session_state.evaluation_results:
            df_data.append({
                "File Name": result['file_name'],
                "Score": f"{result['final_score']:.2f}",
                "Verdict": result['verdict'],
                "Hard Score": f"{result['hard_score']:.2f}",
                "Soft Score": f"{result['soft_score']:.2f}",
                "Experience": result['experience'],
                "Education": ", ".join(result['education']),
                "Missing Skills": len(result['missing_skills']),
                "Timestamp": result['timestamp']
            })
        
        df = pd.DataFrame(df_data)
        
        # Display dataframe
        st.dataframe(df, use_container_width=True)
        
        # Add filters
        st.subheader("Filter Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_score = st.slider("Minimum Score", 0, 100, 0)
        
        with col2:
            verdict_filter = st.selectbox("Verdict", ["All", "High Suitability", "Medium Suitability", "Low Suitability"])
        
        with col3:
            max_missing = st.slider("Maximum Missing Skills", 0, 20, 20)
        
        # Apply filters
        filtered_df = df.copy()
        filtered_df['Score_Num'] = filtered_df['Score'].astype(float)
        filtered_df = filtered_df[filtered_df['Score_Num'] >= min_score]
        filtered_df = filtered_df[filtered_df['Missing Skills'] <= max_missing]
        
        if verdict_filter != "All":
            filtered_df = filtered_df[filtered_df['Verdict'] == verdict_filter]
        
        st.subheader("Filtered Results")
        st.dataframe(filtered_df.drop('Score_Num', axis=1), use_container_width=True)
        
        # Display some charts
        st.subheader("Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            verdict_counts = df['Verdict'].value_counts()
            st.bar_chart(verdict_counts)
        
        with col2:
            scores = df['Score'].astype(float)
            st.write("Score Distribution")
            st.line_chart(scores)

with tab3:
    st.markdown('<div class="sub-header">How the System Works</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
    This Automated Resume Relevance Check System uses a hybrid approach to evaluate resumes against job descriptions:
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Evaluation Process
    
    1. **Text Extraction**: The system extracts text from uploaded PDF or DOCX files.
    
    2. **Hard Matching**: 
       - Skills matching between resume and job description
       - Experience level comparison
       - Education requirements checking
       - Calculates a score based on exact matches
    
    3. **Soft Matching**:
       - Uses TF-IDF and cosine similarity
       - Analyzes semantic similarity between resume and JD
       - Captures contextual relevance beyond keywords
    
    4. **Scoring & Verdict**:
       - Combined score (60% hard match, 40% soft match)
       - Classification into High, Medium, or Low suitability
       - Identification of missing skills/qualifications
    
    ### Technology Stack
    
    - **Python**: Primary programming language
    - **Streamlit**: Web application framework
    - **pdfplumber & docx2txt**: Text extraction from documents
    - **NLTK**: Text processing and normalization
    - **scikit-learn**: TF-IDF and cosine similarity calculations
    
    ### Benefits
    
    - **Consistency**: Standardized evaluation criteria
    - **Efficiency**: Processes multiple resumes simultaneously
    - **Actionable Feedback**: Identifies specific gaps for improvement
    - **Scalability**: Can handle large volumes of applications
    """)