import pickle
import os
import csv
import datetime

import duckdb
import igraph as ig
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()

ALL_DATA_PARQUET = os.getenv("ALL_DATA_PARQUET", "data/alldata.parquet")
TEXTGEN_DUCKDB = os.getenv("TEXTGEN_DUCKDB", "data/text_gen.duckdb")
EDGELIST_OUTPUT_PATH = os.getenv("EDGELIST_OUTPUT_PATH", "data/edgelist.csv")
TRAJECTORIES_OUTPUT_PATH = os.getenv("TRAJECTORIES_OUTPUT_PATH", "data/trajectories.parquet")
ELABORATE_PER_MONTH = os.getenv("ELABORATE_PER_MONTH", "0") in [
    "1",
    "True",
    "true",
]
GRAPH_PATH = os.getenv("GRAPH_PATH", "data/graph.pkl")
REMOVE_PRODUCTS_WITHOUT_DESC = os.getenv("REMOVE_PRODUCTS_WITHOUT_DESC", "1") in [
    "1",
    "True",
    "true",
]

# WARNING: 10 threads are reasonable if you have more than 32GB of RAM. Adjust according to your preferences
HIGH_RAM_DEVICE = os.getenv("HIGH_RAM_DEVICE", "0") in ["1", "True", "true"]
N_THREADS = int(os.getenv("N_THREADS", "10"))
PERCENTILE_MAX = float(os.getenv("PERCENTILE_MAX", "0.99"))
SKIP_TRAJECTORY_SAVING = os.getenv("SKIP_TRAJECTORY_SAVING", "0") in [
    "1",
    "True",
    "true",
]


EDGE_LIST = """
WITH elements AS (
    -- list slicing works differently. In python this would be equivalent to zip(l, l[1:])
    SELECT unnest(list_zip(list(product_id), list(product_id)[2:]), recursive := true)
    FROM valid_trajectories
    WHERE user_session IN ?
    GROUP BY user_session
)
SELECT element1, element2
FROM elements
WHERE element1 IS NOT NULL and element2 IS NOT NULL and element1 != element2
"""


def compute_from_start_to_end_dates(
    conn: duckdb.DuckDBPyConnection, start_date: datetime.date, end_date: datetime.date
):
    user_sessions = conn.sql(
        f"SELECT user_session, COUNT(*) AS user_session_length FROM '{ALL_DATA_PARQUET}' WHERE event_time BETWEEN ? AND ? GROUP BY user_session",
        params=(start_date, end_date),
    )
    percentile = conn.sql(
        f"SELECT quantile_cont(user_session_length, {PERCENTILE_MAX}) as perc FROM user_sessions"
    ).fetchone()[0]
    # We remove all trajectories shorter than 1 and longer than the percentile
    remove_join = ""
    if REMOVE_PRODUCTS_WITHOUT_DESC:
        remove_join = "JOIN products p ON a.product_id = p.product_id"
    valid_trajectories = conn.sql(
        f"""
        WITH sessions AS (
            SELECT user_session
            FROM  user_sessions
            WHERE user_session_length BETWEEN 2 AND {percentile}
        )
        SELECT a.product_id, a.user_session
        FROM '{ALL_DATA_PARQUET}' a
        JOIN sessions s
        ON s.user_session = a.user_session
        {remove_join}
    """
    )
    sessions = conn.sql("SELECT user_session FROM user_sessions")
    pbar = tqdm(desc="Computing and writing edge list to disk...")

    # We add the start/end date to the different file names we have. We also need to remove the extension
    # from the file name, so we can add the start and end date
    edgelist_file_name, ext = os.path.splitext(EDGELIST_OUTPUT_PATH)
    edgelist_file_name = f"{edgelist_file_name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}{ext}"

    with open(edgelist_file_name, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        while s := sessions.fetchmany(100):
            s = [x[0] for x in s]
            edges = conn.sql(EDGE_LIST, params=(s,))
            while e := edges.fetchmany(100):
                writer.writerows(e)
            pbar.update(100)

    print("Creating graph...")
    with open(edgelist_file_name, "r") as f:
        graph = ig.Graph.Read_Edgelist(f, directed=False)

    graph_filename, ext = os.path.splitext(GRAPH_PATH)
    graph_filename = f"{graph_filename}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}{ext}"
    with open(graph_filename, "wb") as f:
        pickle.dump(graph, f)

    if SKIP_TRAJECTORY_SAVING:
        return
    trajectories_file_name, ext = os.path.splitext(TRAJECTORIES_OUTPUT_PATH)
    trajectories_file_name = (
        f"{trajectories_file_name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}{ext}"
    )
    conn.sql(
        f"COPY (SELECT user_session, list(product_id) AS trajectory FROM valid_trajectories GROUP BY user_session) TO '{trajectories_file_name}'"
    )


if __name__ == "__main__":
    print(f"Computing {int(PERCENTILE_MAX * 100)}th percentile...")
    conn = duckdb.connect(TEXTGEN_DUCKDB)

    if ELABORATE_PER_MONTH:
        months_iterator = conn.sql(
            f"SELECT DISTINCT date_trunc('month', event_time) AS month FROM '{ALL_DATA_PARQUET}' ORDER BY month"
        )
        for month in months_iterator.fetchall():
            start_date = month[0].replace(day=1)
            end_date = (start_date + datetime.timedelta(days=31)).replace(day=1)
            print(f"Processing month {start_date} to {end_date}")
            compute_from_start_to_end_dates(conn, start_date, end_date)
    else:
        start_date = datetime.date.min
        end_date = datetime.date.max
        print(f"Processing month {start_date} to {end_date}")
        compute_from_start_to_end_dates(conn, start_date, end_date)
