from pydantic import BaseModel
from .db import get_connection, close_connection

class Job(BaseModel):
    name: str = "baseline"
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    total_training_time: float = 0.0
    total_time: float = 0.0

def get_sql_schema() -> str: 
    return f"""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        train_accuracy REAL,
        test_accuracy REAL,
        total_training_time REAL,
        total_time REAL
    );
    """

def insert_job(job: Job): 
    con = get_connection()
    con.execute(f"INSERT INTO jobs (name, train_accuracy, test_accuracy, total_training_time, total_time) VALUES (?, ?, ?, ?, ?)", (job.name, job.train_accuracy, job.test_accuracy, job.total_training_time, job.total_time))
    con.commit()
    close_connection()

def sync_db(): 
    con = get_connection()
    con.execute(get_sql_schema())
    close_connection()

if __name__ == "__main__":
    sync_db()
