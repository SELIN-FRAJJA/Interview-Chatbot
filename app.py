from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_session import Session
from pymongo import MongoClient  # Import MongoDB Client
import random
import os
import time
import subprocess
import base64
import cv2
import whisper
import numpy as np
from deepface import DeepFace
from chat import get_response
from threading import Thread
from groq import Groq
from dotenv import load_dotenv
 
app = Flask(__name__)
app.secret_key = 'your_secret_key'
load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')  
app.config['SESSION_PERMANENT'] = False  
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
Session(app)

# MongoDB Configuration
client = MongoClient("mongodb://localhost:27017/")  # Connect to MongoDB server
db = client["interview_chatbot"]  # Database Name
users_collection = db["users"]  # Users Collection
responses_collection = db["responses"]  # Responses Collection

# Define the folder where audio files will be stored
UPLOAD_FOLDER = 'uploads'
TRANSCRIPTION_FOLDER = 'transcriptions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTION_FOLDER, exist_ok=True)

def generate_followup_question(transcription_text):
    try:
        prompt = f"""
        The candidate's response: {transcription_text}

        As an expert interviewer, generate a single, concise follow-up question. 
        Focus on probing deeper into their experience or reasoning. 
        Output only the question.
        """
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional interviewer. Generate a relevant follow-up question. Output only the question.",
                },
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=100
        )
        question = response.choices[0].message.content.strip()
        # Clean up formatting
        question = question.strip('"\'').split('?')[0] + '?'
        return question
    except Exception as e:
        print(f"Error generating question: {e}")
        return "Can you elaborate on that?"
    
@app.route('/')
def home():
    return render_template('index.html')  # Main home page

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        # Insert user into MongoDB
        users_collection.insert_one({"email": email, "password": password})
        return redirect('/login')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Find user in MongoDB
        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            session['user_id'] = str(user["_id"])  # Store MongoDB ObjectId
            return redirect('/instructions')  # Redirect to chatbot page after login
        return "Invalid credentials."
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/')

@app.route('/instructions')
def instructions():
    if 'user_id' not in session:
        return redirect('/login')  # Ensure the user is logged in
    
    # Clear previous interview session data
    session.pop('questions', None)
    session.pop('question_count', None)
    session.pop('interview_started', None)

    return render_template('instructions.html')

@app.route('/chatbot')
def chatbot():
    if 'user_id' not in session:
        return redirect('/login')
    # Initialize session only once per interview
    if 'interview_started' not in session:
        session['questions'] = ["Tell me about yourself."]
        session['question_count'] = 1
        session['interview_started'] = True  # Flag to track interview start
    return render_template('chatbot.html', questions=session['questions'])

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    global last_detected_emotion
    data = request.json
    frame_data = data['frame']
    frame_bytes = base64.b64decode(frame_data.split(',')[1])
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(results, list):
            result = results[0]
        else:
            result = results
        last_detected_emotion = result.get('dominant_emotion', 'Unknown')
    except Exception as e:
        last_detected_emotion = 'Error'

    return jsonify({'emotion': last_detected_emotion})

@app.route('/save', methods=['POST'])
def save_emotion():
    global last_detected_emotion
    with open('emo.txt', 'w') as file:
        file.write(last_detected_emotion)
    return jsonify({'message': 'Emotion saved successfully.'})

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    audio_file = request.files['audio']
    timestamp = str(int(time.time()))
    audio_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}.wav")
    txt_path = os.path.join(TRANSCRIPTION_FOLDER, f"{timestamp}.txt")

    try:
        audio_file.save(audio_path)
        print(f"Audio saved to: {audio_path}")  # Log audio save path

        # Run Whisper via Python module
        result = subprocess.run(
            ["python", "-m", "whisper", audio_path, "--model", "medium", "--output_dir", TRANSCRIPTION_FOLDER, "--output_format", "txt"],
            capture_output=True,
            text=True,
            check=True
        )
        print("Whisper Output:", result.stdout)  # Log Whisper output
        print("Whisper Errors:", result.stderr)  # Log Whisper errors

        # Check if the expected file exists
        whisper_output = os.path.join(TRANSCRIPTION_FOLDER, f"{timestamp}.txt")
        if not os.path.exists(whisper_output):
            raise Exception("Whisper did not generate the expected output file.")

        # Read the transcription
        with open(whisper_output, 'r') as f:
            transcription_text = f.read().strip()

        # Generate next question if under limit
        next_question = None
        current_count = session.get('question_count', 1)
        print(f"Current question count: {current_count}")  # Debugging

        if current_count < 5:
            next_question = generate_followup_question(transcription_text)
            print(f"Generated question: {next_question}")  # Debugging
            session.setdefault('questions', []).append(next_question)
            session['question_count'] = current_count + 1
            session.modified = True
        else:
            print("Question limit reached")

        return jsonify({
            'success': True,
            'transcriptionPath': whisper_output,
            'next_question': next_question  # Ensure this is included
        })

    except subprocess.CalledProcessError as e:
        error_msg = f"Whisper Error: {e.stderr}"
        print(error_msg)
        return jsonify({'success': False, 'error': error_msg}), 500
    except Exception as e:
        print(f"General Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/view_result', methods=['POST'])
def view_result():
    try:
        # Step 1: Run vocab.py
        vocab_process = subprocess.run(
            ['python', 'vocab.py'], capture_output=True, text=True, check=True
        )
        vocab_output = vocab_process.stdout.strip()

        # Step 2: Run Audiotest.py
        audiotest_process = subprocess.run(
            ['python', 'Audiotest.py'], capture_output=True, text=True, check=True
        )
        audiotest_output = audiotest_process.stdout.strip()

        # Step 3: Save results in session
        session['vocab_result'] = vocab_output
        session['confidence_result'] = audiotest_output

        # Respond to the front-end
        return jsonify({
            'status': 'success',
            'vocab_result': vocab_output,
            'confidence_result': audiotest_output
        })

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or str(e)
        return jsonify({'status': 'failure', 'error': error_msg}), 500

    except Exception as e:
        return jsonify({'status': 'failure', 'error': str(e)}), 500

@app.route('/result')
def result():
    if 'user_id' not in session:
        return redirect('/login')

    # Fetch results from the session
    vocab_result = session.get('vocab_result', 'No vocab result available.')
    confidence_result = session.get('confidence_result', 'No confidence result available.')

    # Read the emotion result from emo.txt
    emo_result = 'No emotion result available.'
    emo_file_path = 'emo.txt'
    if os.path.exists(emo_file_path):
        with open(emo_file_path, 'r') as file:
            emo_result = file.read().strip()

    # Pass results to the result.html template
    return render_template('result.html', vocab_result=vocab_result, confidence_result=confidence_result, emo_result=emo_result)

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True)