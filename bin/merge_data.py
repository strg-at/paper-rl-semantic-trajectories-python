import os

import dotenv
import duckdb

dotenv.load_dotenv()

DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
DATA_FILES_EXTENSION = os.getenv("DATA_FILES_EXTENSION", "csv")
OUTPUT_FILE = os.getenv("OUTPUT_FILE")


if __name__ == "__main__":
    final_path = os.path.join(DATA_FOLDER, f"*.{DATA_FILES_EXTENSION}")
    duckdb.sql("SET preserve_insertion_order=false;")
    print("Merging data, this will take a while...")
    duckdb.sql(f"COPY (SELECT * FROM '{final_path}' ORDER BY event_time) TO '{OUTPUT_FILE}'")
