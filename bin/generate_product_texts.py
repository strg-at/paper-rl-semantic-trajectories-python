import duckdb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


duckdb_path = "data/text_gen.duckdb"
dataset_parquet_file = "data/alldata.parquet"


class DescriptionWorker:
    def __init__(self):
        self.checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint).to(self.device)
        self.llm_prompt = """\
You generate the product name and description of a fictional product for an online shopping portal given a category, its brand and its price. If no category or brand is provided,
you are allowed to make up a product in the approximate price range. If multiple interpretations are possible, pick one you find most likely. If
insufficient information is provided, output 'information insufficient'. Your answer should follow the following format:
Product name:
{come up with a product name}

Description:
{5 to 10 lines description for the product}
"""

    def generate_product_descriptions(self, category_code: str, brand: str, price: float) -> str:
        complete_prompt = f"""\
        {self.llm_prompt}

        Your input:
        ```

        Category: {category_code}
        Brand: {brand}
        Price: {price}
        ```
        """.strip()
        messages = [{"role": "user", "content": complete_prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.2, top_p=0.9, do_sample=True)
        output_str = self.tokenizer.decode(outputs[0])
        assistant_start = "<|im_start|>assistant"
        assistant_end = "<|im_end|>"
        output_str = output_str[output_str.index(assistant_start) + len(assistant_start) :]
        output_str = output_str[: output_str.index(assistant_end)]
        return output_str.strip()

    def process_batch(self, batch: list[dict], pbar) -> list[tuple]:
        results = []
        for row in batch:
            try:
                desc = self.generate_product_descriptions(row["category_code"], row["brand"], row["price"])
                results.append((row["product_id"], row["category_code"], row["brand"], row["price"], desc))
                pbar.update()
            except Exception as e:
                continue
        return results


def generate_and_store_product_descriptions(dataset_parquet_file: str, batch_size: int):
    conn = duckdb.connect(duckdb_path)
    conn.sql(
        "CREATE TABLE IF NOT EXISTS products (product_id integer, category_code text, brand text, price double, product_description text)"
    )

    data = conn.sql(
        f"""
        WITH source_data AS (
            SELECT DISTINCT ON (product_id, category_code, brand)
                product_id, category_code, brand, price
            FROM '{dataset_parquet_file}'
            WHERE (brand IS NOT NULL OR category_code IS NOT NULL) AND price > 0
        )
        SELECT s.product_id, s.category_code, s.brand, s.price
        FROM source_data s
        LEFT JOIN products p
        ON s.product_id = p.product_id
           AND s.category_code = p.category_code
           AND s.brand = p.brand
        WHERE p.product_id IS NULL
        """
    )
    columns = data.columns
    total_rows = data.count("*").fetchone()[0]

    all_rows = [dict(zip(columns, row)) for row in data.fetchall()]
    batches = [all_rows[i : i + batch_size] for i in range(0, len(all_rows), batch_size)]

    worker = DescriptionWorker()
    with tqdm(total=total_rows, desc="Generating descriptions") as pbar:
        for batch in batches:
            batch_results = worker.process_batch(batch, pbar)
            conn.executemany(
                "INSERT INTO products (product_id, category_code, brand, price, product_description) VALUES (?, ?, ?, ?, ?)",
                batch_results,
            )

    conn.close()


if __name__ == "__main__":
    import os

    batch_size = int(os.getenv("BATCH_SIZE", 10))
    generate_and_store_product_descriptions(dataset_parquet_file, batch_size)
