import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer

duckdb_path = "text_gen.duckdb"
output_path = "embs.npy"

conn = duckdb.connect(duckdb_path)
to_encode = conn.sql("select * from products order by product_id, price")["product_description"].df().values.tolist()
# to_encode is a list of lists with one element, so unwrap
to_encode = [item[0] for item in to_encode]

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embs = model.encode(to_encode, batch_size=512, show_progress_bar=True)

with open(output_path, "wb") as f:
    np.save(f, embs)
