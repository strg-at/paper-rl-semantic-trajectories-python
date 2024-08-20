from threading import Thread

import duckdb
from openai import OpenAI
from tqdm import tqdm

duckdb_path = "text_gen.duckdb"
dataset_parquet_file = "2019-Oct.parquet"

llm_system_prompt = "You generate the description of a fictional product for an online shopping portal given a category, its brand and its price. If no category is provided, you are allowed to make up a product in the approximate price range. If multiple interpretations are possible, pick one you find most likely. If insufficient information is provided, output 'information insufficient'. You description should begin with a header of the product name."


def generate_product_descriptions(ds, columns, llm, conn):
    conn = conn.cursor()
    for d in tqdm(ds):
        product_dict = dict(zip(columns, d))
        messages = [
            {"role": "system", "content": llm_system_prompt},
            {
                "role": "user",
                "content": f"category: {product_dict['category_code']}; brand: {product_dict['brand']}; price: {product_dict['price']}",
            },
        ]
        gen_text = llm.chat.completions.create(model="llama-model", messages=messages)
        description = gen_text.choices[0].message.content
        conn.execute(
            "INSERT INTO products (product_id, category_code, brand, price, product_description) VALUES(?, ?, ?, ?, ?)",
            (
                product_dict["product_id"],
                product_dict["category_code"],
                product_dict["brand"],
                product_dict["price"],
                description,
            ),
        )


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def generate_product_description(
    duckdb_output_file: str, dataset_parquet_file: str, llms
):
    conn = duckdb.connect(duckdb_path)
    conn.sql(
        "CREATE TABLE products (product_id integer, category_code text, brand text, price double, product_description text)"
    )
    data = duckdb.sql(
        f"select distinct product_id product_id, category_code, brand, price FROM '{dataset_parquet_file}' WHERE (brand is not NULL or category_code is not NULL) and price > 0"
    )
    # while (d := data.fetchone()) is not None:
    fetched = data.fetchall()
    columns = data.columns
    threads = []
    for llm, task in zip(llms, split(fetched, len(llms))):
        threads.append(
            Thread(
                target=generate_product_descriptions, args=(task, columns, llm, conn)
            )
        )
        threads[-1].start()

    for thread in threads:
        thread.join()

    conn.close()


if __name__ == "__main__":
    # for use with llama docker containers
    llms = [
        OpenAI(base_url=f"http://0.0.0.0:734{i}/v1", api_key="sk-xxx") for i in range(8)
    ]
    generate_product_description(duckdb_path, dataset_parquet_file, llms)
