import sqlite3

# Connect to SQLite database (will create app.db if it doesn't exist)
conn = sqlite3.connect(r'C:\Users\Selin Frajja\Interview Chatbot\database\app.db')

# Create users table
conn.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);
''')

# Create responses table
conn.execute('''
CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    audio_path TEXT NOT NULL,
    transcription_path TEXT,  -- New column for transcription file path
    FOREIGN KEY(user_id) REFERENCES users(id)
);
''')

# Close the connection
conn.close()
print("Database initialized!")
