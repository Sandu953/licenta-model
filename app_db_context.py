import pg8000
from dotenv import load_dotenv
import os

class AppDbContext:
    def __init__(self):
        load_dotenv()
        self.connection = None

    def __enter__(self):
        self.connection = pg8000.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=int(os.getenv("DB_PORT", 5432))
        )
        return self  # return itself

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            if exc_type is None:
                self.connection.commit()  # Commit if no exceptions
            else:
                self.connection.rollback()  # Rollback if error
            self.connection.close()

    def cursor(self):
        return self.connection.cursor()

    def execute(self, query, params=None):
        cur = self.connection.cursor()
        try:
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            try:
                return cur.fetchall()
            except pg8000.dbapi.InterfaceError:
                return None
        finally:
            cur.close()

