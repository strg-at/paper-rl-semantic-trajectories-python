import pickle

import duckdb
import igraph as ig
import pandas as pd

parquet_file = "evaluating_trajectories/dataset/trajectories.parquet"
duckdb_path = "evaluating_trajectories/dataset/text_gen.duckdb"
trajectories_output = "evaluating_trajectories/dataset/trajectories.pkl"
graph_output = "evaluating_trajectories/dataset/graph.pkl"

conn = duckdb.connect(duckdb_path)

ids_dict = {(entry[0], entry[1], entry[2], entry[3]): i for i, entry in enumerate(conn.sql("select * from products order by product_id, price").fetchall())}  # type: ignore


def get_default(dictionary, key):
    if key in dictionary:
        return dictionary[key]
    return -1


trajectories = []
result = duckdb.sql(f"select * from '{parquet_file}'")
while batch := result.fetchmany(100):
    for entries in batch:
        trajectory = [
            get_default(
                ids_dict,
                (
                    entry["product_id"],
                    entry["category_code"],
                    entry["brand"],
                    entry["price"],
                ),
            )
            for entry in entries[0]
        ]
        if not -1 in trajectory:
            trajectories.append(trajectory)


edges = set()
for trajectory in trajectories:
    for source, target in zip(trajectory, trajectory[1:]):
        # avoid self loops
        if source != target:
            edges.add((source, target))
            edges.add((target, source))


df = pd.DataFrame(edges)
graph = ig.Graph.DataFrame(df, directed=False, use_vids=False)

with open(graph_output, "wb") as f:
    pickle.dump(graph, f)

with open(trajectories_output, "wb") as f:
    pickle.dump(trajectories, f)
