# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import jwt
from functools import wraps
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
import bcrypt

app = Flask(__name__)
CORS(app)  

# Configuration
app.config['SECRET_KEY'] = os.urandom(24) 
GROQ_API_KEY = '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'
MODEL_NAME = "llama2-70b-4096"
PERSIST_DIRECTORY = 'db'

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# User storage (replace with database in production)
users = {}
user_documents = {}

# JWT token helper functions
def generate_token(user_id):
    return jwt.encode(
        {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=1)
        },
        app.config['SECRET_KEY'],
        algorithm='HS256'
    )

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].replace('Bearer ', '')
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated

# API Routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing required fields'}), 400
    
    username = data['username']
    password = data['password']
    
    if username in [users[u]['username'] for u in users]:
        return jsonify({'message': 'Username already exists'}), 400
    
    # Hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    user_id = str(len(users) + 1)
    users[user_id] = {
        'username': username,
        'password': hashed
    }
    user_documents[user_id] = []
    
    token = generate_token(user_id)
    
    return jsonify({
        'message': 'Registration successful',
        'token': token,
        'username': username
    })

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing required fields'}), 400
    
    username = data['username']
    password = data['password']
    
    user_id = None
    for uid, user_data in users.items():
        if user_data['username'] == username:
            user_id = uid
            break
    
    if user_id and bcrypt.checkpw(password.encode('utf-8'), users[user_id]['password']):
        token = generate_token(user_id)
        return jsonify({
            'token': token,
            'username': username
        })
    
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/api/documents', methods=['GET'])
@token_required
def get_documents(current_user_id):
    documents = user_documents.get(current_user_id, [])
    return jsonify({'documents': documents})

@app.route('/api/documents', methods=['POST'])
@token_required
def upload_document(current_user_id):
    if 'document' not in request.files:
        return jsonify({'message': 'No file uploaded'}), 400
    
    file = request.files['document']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'message': 'Only PDF files are supported'}), 400
    
    try:
        # Process and store the document
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # Create vector store
        vector_store = Chroma.from_texts(
            chunks,
            embeddings,
            persist_directory=f"{PERSIST_DIRECTORY}/{current_user_id}/{file.filename}"
        )
        
        # Store document info
        if current_user_id not in user_documents:
            user_documents[current_user_id] = []
            
        user_documents[current_user_id].append({
            'name': file.filename,
            'path': f"{PERSIST_DIRECTORY}/{current_user_id}/{file.filename}"
        })
        
        return jsonify({
            'message': 'Document uploaded successfully',
            'document': {
                'name': file.filename,
                'path': f"{PERSIST_DIRECTORY}/{current_user_id}/{file.filename}"
            }
        })
        
    except Exception as e:
        return jsonify({'message': f'Error uploading document: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user_id):
    try:
        data = request.json
        document_name = data.get('document_name')
        question = data.get('question')
        
        if not document_name or not question:
            return jsonify({'message': 'Missing required fields'}), 400
        
        # Load vector store for the document
        vector_store = Chroma(
            persist_directory=f"{PERSIST_DIRECTORY}/{current_user_id}/{document_name}",
            embedding_function=embeddings
        )
        
        # Initialize Groq chat model
        chat_model = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME
        )
        
        # Create conversation chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        
        # Get response
        response = qa_chain({
            "question": question,
            "chat_history": []
        })
        
        return jsonify({
            'answer': response['answer'],
            'sources': [str(doc.metadata) for doc in response['source_documents']]
        })
        
    except Exception as e:
        return jsonify({'message': f'Error processing chat: {str(e)}'}), 500

if __name__ == '__main__':
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    app.run(debug=True)