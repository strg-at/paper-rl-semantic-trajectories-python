import pickle

import duckdb

dataset_path = "2019-Oct.csv"
output_file = "trajectories.parquet"

columns = duckdb.sql(f"SELECT * FROM '{dataset_path}' LIMIT 1").columns
duckdb.sql(
    f"COPY (SELECT list(struct_pack({','.join(columns)})) FROM '{dataset_path}' GROUP BY user_session) TO '{output_file}'"
)
