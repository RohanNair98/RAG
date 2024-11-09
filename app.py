# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session,jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
import bcrypt
import json

app = Flask(__name__)
app.secret_key = os.urandom(24) 
GROQ_API_KEY='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'
# User management setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize vector store
PERSIST_DIRECTORY = 'db'
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username

# User storage (replace with database in production)
users = {}
user_documents = {}

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id, users[user_id]['username'])
    return None

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in [users[u]['username'] for u in users]:
            flash('Username already exists')
            return redirect(url_for('register'))
        
        # Hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        user_id = str(len(users) + 1)
        users[user_id] = {
            'username': username,
            'password': hashed
        }
        user_documents[user_id] = []
        
        flash('Registration successful')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user_id = None
        for uid, user_data in users.items():
            if user_data['username'] == username:
                user_id = uid
                break
        
        if user_id and bcrypt.checkpw(password.encode('utf-8'), users[user_id]['password']):
            user = User(user_id, username)
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', documents=user_documents[current_user.id])

@app.route('/upload', methods=['POST'])
@login_required
def upload_document():
    if 'document' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('dashboard'))
    
    file = request.files['document']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('dashboard'))
    
    if not file.filename.endswith('.pdf'):
        flash('Only PDF files are supported')
        return redirect(url_for('dashboard'))
    
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
        persist_directory=f"{PERSIST_DIRECTORY}/{current_user.id}/{file.filename}"
    )
    
    # Store document info
    user_documents[current_user.id].append({
        'name': file.filename,
        'path': f"{PERSIST_DIRECTORY}/{current_user.id}/{file.filename}"
    })
    
    flash('Document uploaded successfully')
    return redirect(url_for('dashboard'))

@app.route('/chat/<document_name>')
@login_required
def chat(document_name):
    return render_template('chat.html', document_name=document_name)

@app.route('/api/chat', methods=['POST'])
@login_required
def process_chat():
    data = request.json
    document_name = data['document_name']
    question = data['question']
    
    # Load vector store for the document
    vector_store = Chroma(
        persist_directory=f"{PERSIST_DIRECTORY}/{current_user.id}/{document_name}",
        embedding_function=embeddings
    )
    
    # Initialize Groq chat model
    chat_model = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b-32768"
    )
    
    # Create conversation chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    
    # Get response
    response = qa_chain({"question": question, "chat_history": []})
    
    return jsonify({
        'answer': response['answer'],
        'sources': [str(doc.metadata) for doc in response['source_documents']]
    })

if __name__ == '__main__':
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    app.run(debug=True)