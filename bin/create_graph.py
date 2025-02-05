import pickle
import os
import csv

import duckdb
import igraph as ig
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()

ALL_DATA_PARQUET = os.getenv("ALL_DATA_PARQUET", "data/alldata.parquet")
TEXTGEN_DUCKDB = os.getenv("TEXTGEN_DUCKDB", "data/text_gen.duckdb")
EDGELIST_OUTPUT_PATH = os.getenv("EDGELIST_OUTPUT_PATH", "data/edgelist.csv")
TRAJECTORIES_OUTPUT_PATH = os.getenv(
    "TRAJECTORIES_OUTPUT_PATH", "data/trajectories.csv"
)
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


EDGE_LIST = """
WITH elements AS (
    -- list slicing works differently. In python this would be equivalent to zip(l, l[1:])
    SELECT unnest(list_zip(list(product_id), list(product_id)[2:]), recursive := true)
    FROM {table_name}
    GROUP BY user_session
)
SELECT element1, element2
FROM elements
WHERE element1 IS NOT NULL and element2 IS NOT NULL and element1 != element2
"""

if __name__ == "__main__":
    print(f"Computing {int(PERCENTILE_MAX * 100)}th percentile...")
    conn = duckdb.connect(TEXTGEN_DUCKDB)

    user_sessions = conn.sql(
        f"SELECT user_session, COUNT(*) AS user_session_length FROM '{ALL_DATA_PARQUET}' GROUP BY user_session"
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

    edge_query = EDGE_LIST.format(table_name="valid_trajectories")

    print("Computing and writing edge list to disk...")
    conn.execute(f"SET threads = {N_THREADS};")
    edges = conn.sql(edge_query)

    conn.execute("SET preserve_insertion_order = false;")
    conn.sql(f"COPY edges TO '{EDGELIST_OUTPUT_PATH}' (delim ' ')")

    print("Creating graph...")
    with open(EDGELIST_OUTPUT_PATH, "r") as f:
        graph = ig.Graph.Read_Edgelist(f, directed=False)

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(graph, f)

    if HIGH_RAM_DEVICE:
        conn.sql(
            f"COPY (SELECT user_session, list(product_id) AS trajectory FROM valid_trajectories GROUP BY user_session) TO '{TRAJECTORIES_OUTPUT_PATH}' (DELIM ' ')"
        )
    else:
        results = conn.sql(
            "SELECT user_session, product_id FROM valid_trajectories ORDER BY user_session, event_time"
        )
        with open(TRAJECTORIES_OUTPUT_PATH, "w") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerow(["user_session", "trajectory"])
            current_session = ""
            trajectory = []
            while tqdm(
                res := results.fetchmany(10_000), desc="Writing trajectories to disk"
            ):
                for session, product in res:
                    if not current_session or session == current_session:
                        trajectory.append(product)
                    else:
                        writer.writerow([current_session, trajectory])
                        current_session = session
                        trajectory[product]
