################## IMPORT DES BIBLIOTHEQUES ################## 

import streamlit as st
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import tempfile
import os
from huggingface_hub import login

################## RECUPERATION DE MES IDENTIFIANTS HUGGING FACE ################## 

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token:
    login(token)
else:
    raise ValueError("Le token Hugging Face n'est pas défini.")


################## TRAITEMENT DES FICHIERS et QUESTIONS ################## 

def extract_questions_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])
    
    project_questions = re.findall(r"<projectQuestion>(.*?)</projectQuestion>", full_text, re.DOTALL)
    general_questions = re.findall(r"<generalQuestion>(.*?)</generalQuestion>", full_text, re.DOTALL)
    
    return project_questions, general_questions

def index_project_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)
    
    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(docs, embeddings, persist_directory="./projectBDD")
    return db

def generate_answer(question, retriever):
    context = retriever.get_relevant_documents(question)
    context_text = "\n".join([doc.page_content for doc in context])
    
    prompt = f"Question: {question}\nContext: {context_text}\nRéponse:"
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


################## CONFIGURATION STREAMLIT ################## 


# Config du template du streamlit 

with st.sidebar:
    st.image("SOS.png", use_column_width=True)
    st.image("D4G.png", use_column_width=True)
    st.write("### Projet DataForGood")

st.title("Assistant IA pour les appels à projet")

# Barre de chargement des fichiers 

uploaded_project_call = st.file_uploader("Chargez le PDF de l'appel à projet", type="pdf")
uploaded_project_doc = st.file_uploader("Chargez le PDF du projet", type="pdf")

if uploaded_project_call and uploaded_project_doc:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file1:
        temp_file1.write(uploaded_project_call.read())
        project_call_path = temp_file1.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file2:
        temp_file2.write(uploaded_project_doc.read())
        project_doc_path = temp_file2.name
    
    st.write("Extraction des questions...")
    project_questions, general_questions = extract_questions_from_pdf(project_call_path)
    
    if project_questions:
        st.write(f"Première question : {project_questions[0]}") # je commecnce par la 1ere question avt de boucler sur toutes
        
        st.write("Indexation du document de projet...")
        db = index_project_pdf(project_doc_path)
        retriever = db.as_retriever()
        
        st.write("Génération de la réponse...")
        answer = generate_answer(project_questions[0], retriever)
        
        st.write(f"Réponse générée : {answer}")
    else:
        st.write("Aucune question trouvée.")
    
    os.remove(project_call_path)
    os.remove(project_doc_path)