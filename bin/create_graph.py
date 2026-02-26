import datetime
import os
import pickle

import dotenv
import duckdb
import igraph as ig

dotenv.load_dotenv()

ALL_DATA_PARQUET = os.getenv("ALL_DATA_PARQUET", "data/alldata.parquet")
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
DIRECTED_GRAPH = os.getenv("DIRECTED_GRAPH", "1") in ["1", "True", "true"]

# WARNING: 10 threads are reasonable if you have more than 32GB of RAM. Adjust according to your preferences
HIGH_RAM_DEVICE = os.getenv("HIGH_RAM_DEVICE", "0") in ["1", "True", "true"]
N_THREADS = int(os.getenv("N_THREADS", "10"))
PERCENTILE_MAX = float(os.getenv("PERCENTILE_MAX", "0.99"))
SKIP_TRAJECTORY_SAVING = os.getenv("SKIP_TRAJECTORY_SAVING", "0") in [
    "1",
    "True",
    "true",
]


EDGE_LIST_QUERY = """
COPY (
    WITH elements AS (
        -- list slicing works differently. In python this would be equivalent to zip(l, l[1:])
        SELECT unnest(list_zip(list(product_id), list(product_id)[2:]), recursive := true)
        FROM valid_trajectories
        GROUP BY user_session
    )
    SELECT element1, element2
    FROM elements
    WHERE element1 IS NOT NULL and element2 IS NOT NULL and element1 != element2
) TO '{csvname}' (delimiter ' ')
"""


def compute_from_start_to_end_dates(
    conn: duckdb.DuckDBPyConnection, start_date: datetime.date, end_date: datetime.date
):
    filtered_data = conn.sql(
        f"SELECT * FROM '{ALL_DATA_PARQUET}' WHERE event_time BETWEEN ? AND ?",
        params=(start_date, end_date),
    )
    user_sessions = conn.sql(
        "SELECT user_session, COUNT(*) AS user_session_length FROM filtered_data GROUP BY user_session"
    )
    percentile = conn.sql(  # pyright: ignore[reportOptionalSubscript]
        f"SELECT quantile_cont(user_session_length, {PERCENTILE_MAX}) as perc FROM user_sessions"
    ).fetchone()[0]

    # We remove all trajectories shorter than 1 and longer than the percentile
    remove_join = ""
    if REMOVE_PRODUCTS_WITHOUT_DESC:
        remove_join = "JOIN products p ON a.product_id = p.product_id"
    sessions = conn.sql(
        "SELECT user_session FROM user_sessions WHERE user_session_length BETWEEN 2 AND ?", params=(percentile,)
    )
    valid_trajectories = conn.sql(
        f"""
        SELECT d.product_id, d.user_session
        FROM filtered_data d
        JOIN sessions s
        ON s.user_session = d.user_session
        {remove_join}
    """
    )

    # We add the start/end date to the different file names we have. We also need to remove the extension
    # from the file name, so we can add the start and end date
    edgelist_file_name, ext = os.path.splitext(EDGELIST_OUTPUT_PATH)
    edgelist_file_name = f"{edgelist_file_name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}{ext}"

    conn.sql(EDGE_LIST_QUERY.format(csvname=edgelist_file_name))

    print("Creating graph...")
    graph = ig.Graph.Read_Ncol(edgelist_file_name, directed=DIRECTED_GRAPH)

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
    conn = duckdb.connect()
    print(f"Generating directed graphs: {DIRECTED_GRAPH}")

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
