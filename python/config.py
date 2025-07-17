import os
from dotenv import load_dotenv
import psycopg2.pool

load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'biometric_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'biometric123'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

pool = psycopg2.pool.SimpleConnectionPool(1, 20, **DB_CONFIG)

def get_connection():
    return pool.getconn()

def release_connection(conn):
    pool.putconn(conn)
