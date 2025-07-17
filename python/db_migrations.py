import psycopg2
import logging
from config import DB_CONFIG, validate_config

logger = logging.getLogger(__name__)

def run_migrations():
    try:
        validate_config()
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Create users table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                face_model BYTEA,
                voice_model BYTEA,
                face_encoding BYTEA,
                voice_encoding BYTEA,
                face_image BYTEA,
                voice_audio BYTEA,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cur.execute('CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)')
        
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_migrations()
