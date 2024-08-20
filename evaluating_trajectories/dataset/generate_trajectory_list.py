import pickle

import duckdb

dataset_path = "2019-Oct.csv"
output_file = "trajectories.pkl"

columns = duckdb.sql(f"SELECT * FROM '{dataset_path}' LIMIT 1").columns
data = duckdb.sql(f"SELECT list(struct_pack({','.join(columns)})) FROM '{dataset_path}' GROUP BY user_session")


with open(output_file, "wb") as f:
    pickle.dump(df, f)
