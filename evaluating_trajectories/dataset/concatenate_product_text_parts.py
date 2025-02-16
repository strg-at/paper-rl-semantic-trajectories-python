import duckdb

# List of DuckDB files
duckdb_files = ['text_gen_0.duckdb', 'text_gen_1.duckdb', 'text_gen_2.duckdb', 'text_gen_3.duckdb', 'text_gen_4.duckdb', 'text_gen_5.duckdb']

# Create or open the combined DuckDB
con = duckdb.connect('text_gen.duckdb')

# Create the combined table based on the first file's table structure
for i, db_file in enumerate(duckdb_files):
    con.execute(f"ATTACH '{db_file}' AS db{i}")
con.execute(f"CREATE TABLE products AS SELECT * FROM db0.products WHERE FALSE")

# Loop over each file and insert data into the combined table
for i, db_file in enumerate(duckdb_files):
    con.execute(f"INSERT INTO products SELECT * FROM db{i}.products")

# Close the connection
con.close()

