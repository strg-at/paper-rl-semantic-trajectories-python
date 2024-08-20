import pickle

import duckdb

dataset_path = "2019-Oct.csv"
output_file = "trajectories.pkl"

df = duckdb.sql(f"SELECT * FROM '{dataset_path}'").df()
events_list = (
    df.sort_values(by=["user_session", "event_time"])
    .groupby("user_session", sort=False)
    .apply(lambda x: x.to_dict(orient="records"))
    .tolist()
)  # needs a LOT of memory!


with open(output_file, "wb") as f:
    pickle.dump(df, f)
