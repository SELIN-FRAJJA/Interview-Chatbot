from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_session import Session
import sqlite3
import random
import os
import time
import subprocess
import base64
import cv2
import numpy as np
from deepface import DeepFace
from chat import get_response

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')  # Folder to store session data
app.config['SESSION_PERMANENT'] = False  # Sessions won't persist after the app restarts
Session(app)

# Define the folder where audio files will be stored
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  

QUESTIONS = [
    "Tell me about yourself.",
    "What are your strengths?",
    "What are your weaknesses?",
    "Where do you see yourself in 5 years?",
    "Why should we hire you?"
]

def get_random_questions():
    return random.sample(QUESTIONS, 3)  # Select 3 random questions

def get_db_connection():
    return sqlite3.connect('database/app.db')

@app.route('/')
def home():
    return render_template('index.html')  # Main home page

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        with get_db_connection() as conn:
            try:
                conn.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, password))
                conn.commit()
                return redirect('/login')
            except:
                return "User already exists."
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        with get_db_connection() as conn:
            user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
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
    return render_template('instructions.html')  # Render the instructions page

@app.route('/chatbot')
def chatbot():
    if 'user_id' not in session:
        return redirect('/login')
    questions = get_random_questions()  # Get 3 random questions
    return render_template('chatbot.html', questions=questions)

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

@app.route('/save-response', methods=['POST'])
def save_response():
    if 'user_id' not in session:
        return redirect('/login')

    # Fetch user data from session
    user_id = session['user_id']
    question = request.form.get('question')  # Get the question
    audio = request.files.get('audio')  # Get the audio file
    custom_filename = request.form.get('filename')  # Get the custom filename from the user input

    if not question or not audio or not custom_filename:
        return "Invalid question, audio file, or filename!", 400

    # Secure the custom filename to prevent security issues
    secure_name = secure_filename(custom_filename)

     # Get the current time in the format hhmmss
    timestamp = time.strftime("%H%M%S", time.localtime(int(time.time())))

# Fetch the existing file count (for record1, record2, etc.)
    record_count = len([f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(f"{timestamp}_record")])

    # Generate the filename like hhmmss_record1.wav, hhmmss_record2.wav
    unique_filename = f"{timestamp}_record{record_count + 1}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)

    try:
        # Save the audio file with the custom filename
        audio.save(filepath)

        # Log for debugging
        print(f"Audio saved: {filepath}")

        # Insert response into the database
        with get_db_connection() as conn:
            conn.execute('INSERT INTO responses (user_id, question, audio_path) VALUES (?, ?, ?)',
                         (user_id, question, filepath))
            conn.commit()

        return f"Saved successfully as {unique_filename}!"

    except Exception as e:
        print(f"Error saving response: {e}")
        return "Failed to save response!", 500
    
@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'user_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized access."}), 401

    try:
        # Run transcribe.py and capture its output
        process = subprocess.run(['python', 'transcribe.py'], capture_output=True, text=True)
        if process.returncode == 0:
            return jsonify({"status": "success", "message": "All audios have been transcribed!"})
        else:
            print(f"Transcription error: {process.stderr}")
            return jsonify({"status": "error", "message": "Transcription failed. Try again."}), 500
    except Exception as e:
        print(f"Error running transcribe.py: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

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
    text=request.get_json().get("message")
    response=get_response(text)
    message={"answer": response}
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True)