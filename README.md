# Face and Voice Authentication System

## Overview
This project is a full-stack biometric authentication system that combines advanced face and voice recognition for secure user verification. It features a Python Flask backend for biometric processing and a modern React-based frontend for user interaction. The system leverages deep learning models, liveness detection, and a PostgreSQL database to ensure robust security and user management.

---

## Features
- **Face Recognition:** Uses `face_recognition` and deep learning models for accurate face matching.
- **Voice Recognition:** Utilizes `resemblyzer` and `Wav2Vec2` for speaker verification.
- **Liveness Detection:** Prevents spoofing with MediaPipe-based face liveness checks.
- **Secure Data Storage:** Biometric data is encrypted and stored in a PostgreSQL database.
- **RESTful API:** Flask-based API for registration, authentication, and biometric operations.
- **Modern UI:** React (with Tailwind CSS) frontend for seamless user experience.
- **CORS Support:** Allows secure cross-origin requests from trusted frontends.

---

## Project Structure
```
Face-and-Voice-Authentication/
│
├── python/                # Backend source code
│   ├── app.py             # Main Flask application
│   ├── config.py          # Configuration and DB connection pool
│   ├── face_liveness_handler.py # Liveness detection logic
│   ├── db_migrations.py   # Database schema migrations
│   └── requirements.txt   # Python dependencies
│
├── db/
│   ├── biometric.db       # (Legacy/backup) SQLite DB (main DB is PostgreSQL)
│   ├── biometrics.db      # (Legacy/backup) SQLite DB
│   └── reset_postgres_password.sql # Utility script
│
├── biometric_data/
│   └── users/             # Encrypted biometric data per user
│
├── UI/
│   └── index.html         # React-based frontend (SPA)
│
└── README.md              # Project documentation
```

---

## Backend Setup

### 1. Prerequisites
- Python 3.10+
- PostgreSQL database
- (Optional) Node.js for frontend development

### 2. Install Python Dependencies
```bash
pip install -r python/requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the `python/` directory with the following (or use defaults):
```
DB_NAME=biometric_db
DB_USER=postgres
DB_PASSWORD=biometric123
DB_HOST=localhost
DB_PORT=5432
```

### 4. Run Database Migrations
This will create the `users` table and indexes in your PostgreSQL DB:
```bash
python python/db_migrations.py
```

### 5. Start the Flask API
```bash
python python/app.py
```
The API will be available at `http://localhost:5000` by default.

---

## Frontend Setup

The frontend is a single-page React app embedded in `UI/index.html` and uses CDN links for React, Tailwind CSS, and Axios. You can open this file directly in your browser or serve it via a static server.

- **Endpoints:** The frontend communicates with the Flask API at `http://localhost:5000` (or your deployed backend).
- **Features:**
  - User registration and login with face and voice capture
  - Real-time liveness and quality checks
  - Device fingerprinting and session management

---

## Biometric Data Storage
- User data is stored in `biometric_data/users/<user_id>/` with encrypted face images, voice samples, and metadata.
- All sensitive data is encrypted using Fernet symmetric encryption.

---

## Security Notes
- Liveness detection is enforced to prevent spoofing (see `face_liveness_handler.py`).
- All API endpoints are CORS-protected and support secure headers.
- Database credentials should be kept secret and not committed to version control.

---

## Main Python Dependencies
- `face_recognition`, `resemblyzer`, `transformers`, `torch`, `opencv-python`, `mediapipe`, `librosa`, `cryptography`, `psycopg2-binary`, `flask`, `flask-cors`, `pydub`, `Pillow`

---

## License
This project is for educational and research purposes. Please review and comply with the licenses of all third-party libraries used.

---

## Authors
- [Your Name or Team]

---

## Acknowledgements
- Open source libraries and models used in this project.
- Special thanks to the contributors of `face_recognition`, `transformers`, and `mediapipe`.
