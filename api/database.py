# Database configuration and session management

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from urllib.parse import quote_plus

# Database connection settings
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5432")
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASS = os.getenv("PGPASSWORD", "postgres")
DB_NAME = os.getenv("PGDATABASE", "water_analytics")

# Create database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"postgresql://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

def get_engine():
    """Create a SQLAlchemy engine"""
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        # If connection fails, use SQLite as fallback for development
        print(f"Error connecting to PostgreSQL: {str(e)}")
        print("Using SQLite database as fallback")
        return create_engine("sqlite:///./water_analytics.db", connect_args={"check_same_thread": False})

# Create engine
engine = get_engine()

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

# Function to get database connection
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
