# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "tabulate",
# ]
# ///
import sqlite3
import pandas as pd
import os

db_path = "data/db.sqlite"

def show_tables(db_path, num_rows=10):
    """
    Connects to a SQLite database, reads all tables, and prints the first `num_rows` of each table.
    """
    file_size = os.path.getsize(db_path)
    file_size_mb = file_size / 1024 / 1024
    print(f"Database size: {file_size_mb:.2f} MB\n")

    try:
        con = sqlite3.connect(db_path)
        cursor = con.cursor()

        # Get a list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("No tables found in the database.")
            return

        for table_name in tables:
            table_name = table_name[0]
            
            # Get the total number of rows in the table
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            row_count = cursor.fetchone()[0]
            
            print(f"--- Table: {table_name} (Total rows: {row_count}) ---")

            cursor.execute(f"PRAGMA table_info('{table_name}')")
            column_info = cursor.fetchall()
            columns = [info[1] for info in column_info if info[1] != 'id']

            if not columns:
                print("No non-id columns to display.")
                print("\n" + "="*50 + "\n")
                continue

            select_clause = ', '.join(f'"{col}"' for col in columns)
            query = f'SELECT {select_clause} FROM "{table_name}" LIMIT {num_rows}'

            try:
                df = pd.read_sql_query(query, con)
                print(df.to_markdown(index=False))

            except pd.io.sql.DatabaseError as e:
                print(f"Could not read table '{table_name}': {e}")

            print("\n" + "="*50 + "\n")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if 'con' in locals() and con:
            con.close()

if __name__ == "__main__":
    show_tables(db_path)