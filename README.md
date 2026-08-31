🩺 Doctor RAG – AI-Based Patient Risk Prediction System
📌 Introduction

Doctor RAG is an AI-assisted healthcare information and patient risk analysis system designed to help doctors and administrators quickly review patient records and retrieve relevant medical knowledge.

The system uses Retrieval-Augmented Generation (RAG) concepts with a local medical knowledge base and FAISS vector search to retrieve relevant information from stored documents.

It provides structured summaries for:

👨‍⚕️ Patient-specific medical records
🩺 General health topics
⚠️ Clinical risk factors
📊 Risk score and risk level
🔮 Possible clinical risks
📅 Possible long-term outcomes
💊 Medication adherence
📝 Doctor-focused observations
✨ Features
👤 Patient Analysis
Search patient records using Patient ID such as P001, P002, etc.
View patient age, gender and medical history.
Check medication adherence and missed doses.
Review recent symptoms and doctor notes.
Generate a structured clinical risk summary.
⚠️ Risk Analysis

The system analyzes available patient information and identifies:

Good/protective factors
Active risk factors
Predicted clinical risks
Possible long-term outcomes
Doctor attention points
Reasoning behind the assigned risk

Risk levels are classified as:

🟢 Low Risk | 🟠 Medium Risk | 🔴 High Risk

📚 General Health Knowledge

The system can retrieve information about topics such as:

Fever
Diabetes
Blood Pressure
Hypertension
Cold
Heart-related conditions
Precautions
🔎 Smart Retrieval

The system uses:

Text document loading
Text chunking
Hugging Face embeddings
FAISS vector database
Similarity-based document retrieval
Patient ID and topic-based filtering
⚙️ How It Works
                 User Query
                     │
                     ▼
             Query Processing
                     │
                     ▼
          FAISS Similarity Search
                     │
                     ▼
             Relevant Documents
                     │
              ┌──────┴──────┐
              ▼             ▼
        Patient Record   Knowledge Base
              │             │
              ▼             ▼
        Risk Analysis    Topic Summary
              │             │
              └──────┬──────┘
                     ▼
             Structured Summary
                     │
                     ▼
              Gradio Interface
🧠 RAG Pipeline
Medical Documents
       ↓
Document Loading
       ↓
Text Splitting
       ↓
Hugging Face Embeddings
       ↓
FAISS Vector Store
       ↓
User Query
       ↓
Similarity Retrieval
       ↓
Relevant Medical Information
       ↓
Structured Patient / Topic Summary
🛠️ Technologies Used
Technology	Purpose
Python	Core programming language
Gradio	Web-based user interface
LangChain	Document processing and retrieval
Hugging Face	Text embeddings
Sentence Transformers	Embedding model
FAISS	Vector similarity search
HTML/CSS	Custom interface styling
Embedding Model
sentence-transformers/all-MiniLM-L6-v2
📁 Project Structure
doctor-rag-project/
│
├── data/
│   ├── patient records
│   └── medical knowledge files
│
├── vectorstore/
│   └── doctor_patient_index
│
├── doctor_rag_app.py
├── requirements.txt
└── README.md
🚀 Installation & Setup
1. Clone the Repository
git clone https://github.com/kudukunal35-cpu/doctor-rag-project.git
cd doctor-rag-project
2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
▶️ Run the Project

Run the application using:

python doctor_rag_app.py

The application will launch the Gradio interface.

The application is configured to run locally on:

http://127.0.0.1:7860

A temporary public Gradio share link may also be generated because the application uses:

demo.launch(share=True)
💡 Example Queries
Patient Queries
Give a short summary of patient P001 for doctor review
What are the main risks for patient P003?
Predict future complications for patient P003
Does patient P001 need medication review?
General Health Queries
Give summary about fever precautions
What to do in diabetes?
What are blood pressure precautions?
Give summary for cold and cough care
📊 Risk Analysis Logic

The system calculates a rule-based risk score using available patient information such as:

Medication adherence
Missed doses
Symptoms
Medical history
Medication information
Doctor notes
Laboratory summary

The resulting score is mapped to:

Score ≥ 7  → High Risk
Score 4–6  → Medium Risk
Score < 4  → Low Risk
🔐 Data & Privacy

The project uses local text files as its primary data source.

Do not upload real patient-identifiable or confidential medical information to a public repository.

Use only synthetic, anonymized, or publicly permitted data when sharing this project.

⚠️ Medical Disclaimer

This project is developed for educational, research, and demonstration purposes only.

The generated risk analysis and health information should not be considered a medical diagnosis, treatment recommendation, or substitute for a qualified healthcare professional.

Always consult an appropriate medical professional for actual patient care.

🔮 Future Scope
Integration with a secure clinical database
Authentication and role-based access
Improved patient dashboard
Advanced medical document retrieval
More comprehensive clinical risk models
Doctor/admin analytics dashboard
Secure deployment for controlled environments
Mobile-friendly interface
👨‍💻 Project

Doctor RAG – AI-Based Patient Risk Prediction System

Built using Python, LangChain, Hugging Face, FAISS and Gradio.
