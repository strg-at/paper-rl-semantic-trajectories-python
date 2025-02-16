from llama_cpp import Llama
from tqdm import tqdm
import duckdb
import argparse

llm_system_prompt = "You generate the description of a fictional product for an online shopping portal given a category, its brand and its price. If no category is provided, you are allowed to make up a product in the approximate price range. If multiple interpretations are possible, pick one you find most likely. If insufficient information is provided, output 'information insufficient'. You description should begin with a header of the product name."

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process a portion of the dataset.")
parser.add_argument('--num-parts', type=int, default=6, help="Choose how many parts the dataset should be sliced into")
parser.add_argument('--part', type=int, required=True, help="Choose which part of the dataset to process")
args = parser.parse_args()

duckdb_path = f"text_gen_{args.part}.duckdb"

# Load model
llm = Llama.from_pretrained(repo_id="google/gemma-2-2b-it-GGUF", filename="2b_it_v2.gguf", verbose=True, n_gpu_layers=300, chat_format='gemma')

# Load dataset and determine total count
ds = duckdb.sql("select distinct on (product_id) * from 'data/alldata.parquet'")
count = duckdb.sql("select distinct on (product_id) * from 'data/alldata.parquet'").count("*").fetchone()[0]

# Determine size of each part
part_size = count // args.num_parts
offset = args.part * part_size

# Adjust for the last part (part 5) to include all remaining records
if args.part == args.num_parts-1:
    part_size = count - offset

# Select the appropriate part of the dataset
ds_part = duckdb.sql(f"select * from 'data/alldata.parquet' limit {part_size} offset {offset}")

pbar = tqdm(desc="generating", total=part_size)

conn = duckdb.connect(duckdb_path)
conn.sql(
    "CREATE TABLE products (product_id integer, category_code text, brand text, price double, product_description text)"
)

# Process the selected part of the dataset
while (d := ds_part.fetchone()):
    d = dict(zip(ds_part.columns, d))
    gen = llm.create_chat_completion(messages=[
        {"role": "system", "content": llm_system_prompt},
        {
            "role": "user", 
            "content": f"Generate a product description for a product with category {d['category_code']}, brand {d['brand']}, and price {d['price']} dollars"
    }])
    conn.execute(
        "INSERT INTO products (product_id, category_code, brand, price, product_description) VALUES(?, ?, ?, ?, ?)",
        (
            d["product_id"],
            d["category_code"],
            d["brand"],
            d["price"],
            gen['choices'][0]['message']['content'],
        ),
    )
    pbar.update()

pbar.close()

