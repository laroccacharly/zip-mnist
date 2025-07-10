from .db import get_connection

if __name__ == "__main__":
    con = get_connection()
    con.execute("DELETE FROM jobs")
    con.commit()
    con.close()
    print("Database cleared")