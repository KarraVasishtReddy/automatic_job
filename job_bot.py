import os
import json
import ollama
from pydantic import BaseModel
from typing import List

class JobAnalysis(BaseModel):
    company_name: str
    job_title: str
    is_remote_in_hyderabad: bool
    match_score: int 
    missing_skills: List[str]
    justification: str

def load_context():
    with open("master_resume.txt", "r") as f:
        resume = f.read()
    
    if not os.path.exists("applications.json"):
        with open("applications.json", "w") as f:
            json.dump([], f)
            
    return resume

def process_new_jobs(master_resume):
    raw_folder = "raw_jobs"
    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder)
        print("Created 'raw_jobs/'. Drop your job descriptions there as .txt files.")
        return

    for filename in os.listdir(raw_folder):
        if not filename.endswith(".txt"):
            continue
            
        file_path = os.path.join(raw_folder, filename)
        with open(file_path, "r") as f:
            job_description = f.read()

        print(f"\n🤖 Analyzing {filename}...")
        
        # Tailored prompt for a Data/ML Engineer
        prompt = f"""
        You are an expert technical recruiter matching candidates to Data Engineering and AI/ML roles.
        Analyze the following Job Description against the Candidate's Resume.
        
        The candidate is a highly skilled AI/ML & Data Engineer proficient in Python, PySpark, TensorFlow, and MLOps.
        
        CRITICAL RULES:
        1. Set 'is_remote_in_hyderabad' to true ONLY if the job mentions 'Remote' AND the company hires out of Hyderabad/India.
        2. Provide an objective 'match_score' (1-100) based on alignment with big data pipelines, cloud architecture, and ML model deployment.

        RESUME:
        {master_resume}

        JOB DESCRIPTION:
        {job_description}
        """

        response = ollama.chat(
            model='llama3.1',
            messages=[{'role': 'user', 'content': prompt}],
            format=JobAnalysis.model_json_schema()
        )
        
        analysis = JobAnalysis.model_validate_json(response.message.content)
        print(f"📈 Match Score: {analysis.match_score}/100 | Target Location Match: {analysis.is_remote_in_hyderabad}")

        if analysis.match_score >= 70 and analysis.is_remote_in_hyderabad:
            print("✨ Strong match! Generating tailored assets...")
            generate_tailored_assets(analysis.company_name, analysis.job_title, job_description, master_resume)
            status = "Assets_Generated"
        else:
            print("❌ Skipping: Low match score or not remote/Hyderabad target.")
            status = "Archived_Low_Match"

        update_tracking_log(filename, analysis, status)
        os.remove(file_path)

def generate_tailored_assets(company, title, job_desc, resume):
    os.makedirs("output", exist_ok=True)
    clean_name = "".join(x for x in company if x.isalnum())
    
    # Prompt explicitly uses your track record of throughput gains and pipeline reliability
    generation_prompt = f"""
    You are an elite career coach. Write a highly tailored, compelling cover letter (under 250 words) for a remote position.
    The candidate has a proven track record of achieving 45% ETL throughput gains, 60% query speedups, and 99.9% pipeline reliability.
    Map these specific achievements to the core problems outlined in the job description.
    
    ROLE: {title} at {company}
    CANDIDATE RESUME: {resume}
    JOB DETAILS: {job_desc}
    """
    
    gen_response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': generation_prompt}]
    )
    
    output_path = f"output/{clean_name}_Cover_Letter.txt"
    with open(output_path, "w") as f:
        f.write(gen_response.message.content)
    print(f"💾 Saved tailored cover letter to: {output_path}")

def update_tracking_log(filename, analysis: JobAnalysis, status: str):
    with open("applications.json", "r") as f:
        data = json.load(f)
        
    data.append({
        "source_file": filename,
        "company": analysis.company_name,
        "title": analysis.job_title,
        "score": analysis.match_score,
        "justification": analysis.justification,
        "missing_skills": analysis.missing_skills,
        "status": status
    })
    
    with open("applications.json", "w") as f:
        json.dump(data, indent=4, f)

if __name__ == "__main__":
    resume_data = load_context()
    process_new_jobs(resume_data)
