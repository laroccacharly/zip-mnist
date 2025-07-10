from pydantic import BaseModel
from .db import get_connection, close_connection

class Job(BaseModel):
    model_name: str = "xgb"
    test_accuracy: float = 0.0
    total_training_time: float = 0.0

def get_sql_schema() -> str: 
    return f"""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        test_accuracy REAL,
        total_training_time REAL
    );
    """

def insert_job(job: Job): 
    con = get_connection()
    con.execute(f"INSERT INTO jobs (model_name, test_accuracy, total_training_time) VALUES (?, ?, ?)", (job.model_name, job.test_accuracy, job.total_training_time))
    con.commit()
    close_connection()

def sync_db(): 
    con = get_connection()
    con.execute(get_sql_schema())
    close_connection()

if __name__ == "__main__":
    sync_db()
