import os
from .db import DB_PATH

if __name__ == "__main__":
    os.remove(DB_PATH)      
    print("Database removed")