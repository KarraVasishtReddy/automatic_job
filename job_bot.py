import os
import json
import re
from openai import OpenAI
from pydantic import BaseModel
from typing import List

# 1. Initialize GitHub Models Client
# Ensure GITHUB_TOKEN is added to your environment variables or GitHub Secrets
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN"),
)

# 2. Define Structured Output Model for the AI Response
class JobAnalysis(BaseModel):
    company_name: str
    job_title: str
    is_remote_in_hyderabad: bool
    match_score: int 
    extracted_tech_stack: List[str]
    missing_skills: List[str]
    justification: str

def setup_directories():
    """Create systemic storage hierarchy if not existing."""
    dirs = [
        "storage/raw_ingest",
        "storage/parsed_jobs",
        "storage/failed_checks",
        "output_materials"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def sanitize_filename(name: str) -> str:
    """Clean company names to prevent OS filesystem errors."""
    return re.sub(r'[\\/*?:"<>| ]', '_', name).strip('_')

def load_master_resume() -> str:
    if not os.path.exists("master_resume.txt"):
        raise FileNotFoundError("Please create 'master_resume.txt' containing your profile details first.")
    with open("master_resume.txt", "r", encoding="utf-8") as f:
        return f.read()

def pipeline_orchestrator():
    setup_directories()
    try:
        resume = load_master_resume()
    except FileNotFoundError as e:
        print(e)
        return

    raw_folder = "storage/raw_ingest"
    files = [f for f in os.listdir(raw_folder) if f.endswith(".txt")]

    if not files:
        print("📥 No data found in 'storage/raw_ingest/'. Place raw text files there to begin.")
        return

    for filename in files:
        file_path = os.path.join(raw_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_job_content = f.read()

        print(f"\n⚡ Processing file: {filename}")
        
        # Evaluate and Extract using structured schema
        analysis = analyze_job_text(raw_job_content, resume)
        if not analysis:
            continue

        clean_company = sanitize_filename(analysis.company_name)
        clean_title = sanitize_filename(analysis.job_title)
        save_target_name = f"{clean_company}_{clean_title}.json"

        # Strategic Routing Decisions based on score matrices
        if analysis.match_score >= 70 and analysis.is_remote_in_hyderabad:
            print(f"🎯 High Match ({analysis.match_score}%). Saving metadata and generating personalized materials...")
            
            # Save parsed JSON database object
            with open(os.path.join("storage/parsed_jobs", save_target_name), "w", encoding="utf-8") as f:
                f.write(analysis.model_dump_json(indent=4))

            # Build distinct application directory asset package
            generate_application_package(analysis, raw_job_content, resume)
            status = "Assets_Generated"
        else:
            print(f"⏩ Low Match or Location Mismatch ({analysis.match_score}%). Archiving parsing profile to cold storage.")
            with open(os.path.join("storage/failed_checks", save_target_name), "w", encoding="utf-8") as f:
                f.write(analysis.model_dump_json(indent=4))
            status = "Archived_Low_Match"

        # Update tracking log and clean up raw ingestion queue
        update_tracking_log(filename, analysis, status)
        os.remove(file_path)

def analyze_job_text(job_text: str, resume: str) -> JobAnalysis:
    prompt = f"""
    You are an AI recruiting parser evaluating technical roles.
    Parse the following text and determine alignment with a Data/ML Engineering profile.

    CRITICAL FILTERS:
    - is_remote_in_hyderabad: Must be explicitly 'true' only if the role indicates working remotely for a company based in or hiring from Hyderabad/India.
    - match_score: Evaluate match strictly on scale of 1-100 based on tools like Python, PySpark, MLOps, SQL, and Cloud architectures.

    RESUME CONTEXT:
    {resume}

    RAW JOB TEXT:
    {job_text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise data extraction engine that outputs valid JSON schemas."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return JobAnalysis.model_validate_json(response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ Error during model parsing execution: {e}")
        return None

def generate_application_package(analysis: JobAnalysis, job_text: str, resume: str):
    clean_company = sanitize_filename(analysis.company_name)
    package_dir = os.path.join("output_materials", clean_company)
    os.makedirs(package_dir, exist_ok=True)

    generation_prompt = f"""
    You are an expert technical resume writer. Review the job profile and generate matching assets.
    Candidate profile metrics: 3+ years engineering experience, 45% ETL throughput optimization, 99.9% reliability.

    Create a context-mapped cover letter (<250 words) bridging the tech stack ({analysis.extracted_tech_stack}) to the candidate resume achievements.
    
    ROLE: {analysis.job_title} at {analysis.company_name}
    JOB REQUIREMENTS: {job_text}
    CANDIDATE RESUME: {resume}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional career agent writing hyper-targeted submission materials."},
                {"role": "user", "content": generation_prompt}
            ],
            temperature=0.7
        )
        
        # Save individual file output
        with open(os.path.join(package_dir, "cover_letter.txt"), "w", encoding="utf-8") as f:
            f.write(response.choices[0].message.content)
            
        print(f"💾 File Package saved successfully inside: {package_dir}/")
    except Exception as e:
        print(f"⚠️ Error creating text assets: {e}")

def update_tracking_log(filename, analysis: JobAnalysis, status: str):
    log_file = "applications.json"
    
    # Initialize file if it doesn't exist or is empty
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump([], f)

    # Read existing data safely
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = []
    except json.JSONDecodeError:
        data = []

    # Append new log data
    data.append({
        "source_file": filename,
        "company": analysis.company_name,
        "title": analysis.job_title,
        "score": analysis.match_score,
        "justification": analysis.justification,
        "missing_skills": analysis.missing_skills,
        "status": status
    })
    
    # Corrected argument position syntax: json.dump(data, file_object, indent=4)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    pipeline_orchestrator()
