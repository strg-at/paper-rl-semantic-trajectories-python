# paper-evaluation-trajectories-python

Python code for the Research paper on evaluating user trajectories

## Cloning and pulling

This git repo also provides GloVe as a submodule. When cloning it you can do `git clone --recursive https://github.com/strg-at/paper-evaluation-trajectories-python.git`
to automatically also download the submodule.

Pulling changes for the submodules should _always_ be done with:

```bash
git submodule update --remote --rebase
```

## Running

### Installing

This Python project utilizes modern packaging standards defined in the [pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) configuration file. Project management tasks like environment setup and dependency installation are handled using [UV](https://github.com/astral-sh/uv).
The `pyproject.toml` configuration file can be and is also used to store 3rd party tools configurations, such as ruff, basedpyright etc.

### Using UV

UV is a command-line tool that needs to be installed first. You can typically install it using pip, pipx, or your system's package manager if available.

Refer to the [official UV installation guide](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) for the most up-to-date methods. A common way is:

```bash
# Using pip (ensure pip is up-to-date)
pip install uv

# Or using pipx (recommended for CLI tools)
pipx install uv
```

You can then install the project dependencies in a virtual environment with:

```bash
uv sync --extra cpu  # use --extra gpu if you're running on nvidia or --extra metal if running on Mac Metal cpus
```

If you also want development tools and libraries:

```bash
uv sync --extra cpu --extra dev
```

You can then enable the environment with:

```bash
source .venv/bin/activate
```

_Notice_: UV can also manage python versions. See [Install Python](https://docs.astral.sh/uv/guides/install-python/).

### Dependency management

You can add a dependency with:

```bash
uv add package-name
```

If the dependency should only be used in some cases (e.g., dev dependencies), then use:

```bash
uv add --group dev package-name
```

#### Building GloVe

Run:

```bash
make all
```

to compile both our glove_ffi extension, and glove original code

### Dataset

The original `.csv` files of the dataset we use can be downloaded from <https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store>

Notice that there is more than one file: the whole dataset goes from October data to April data.

### Generating GloVe embeddings

In order to generate GloVe embeddings you need to have:

1. a .txt file containing the training dataset (`glove_trajectories.txt`)
1. a vocabulary .txt file (`vocab.txt`);
1. a cooccurrence binary file (`cooccurrence.shuf.bin`)

We will generate the first file with the `bin/glove_preprocessing.py` script.
The other two files can be generated with GloVe original C code (which you should have in the `submodules` directory, see [Cloning and pulling](#cloning-and-pulling)).

Once you have downloaded the dataset, use the following python code to merge the csv files and generate a single .parquet file:

```python
import duckdb

duckdb.sql("COPY (SELECT * FROM read_csv(['2019-Oct.csv.gz', '2019-Nov.csv.gz', ...])) TO 'alldata.parquet'")
```

You should then run the `bin/create_graph.py` script which will generate several files. Look at the top of the script for configuration options (via env variables).

```bash
python bin/create_graph.py
```

Then, you can generate the `glove_trajectories.txt` file with:

```bash
python bin/glove_preprocessing.py --path trajectories.csv --output-file glove_trajectories.txt
```

Feel free to adjust the parameters to your liking.

Now, we can generate the vocabulary and the cooccurence.shuf.bin file.
Before running this, check the `demo.sh` file and see if you want to change any parameter:

```bash
cd submodules/GloVe && ./demo.sh
```

#### Running on CPU

If you cannot or do not want to run on GPU, add the following line to `demo.sh`:

```bash
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
```

This will compute the embeddings using Glove C code, and will output a `vectors.txt` file.

#### Running on GPU

For this, you will need the `evaluating_trajectories/experiments/train_glove_embeddings.py` script. Run it with:

```bash
python evaluating_trajectories/experiments/train_glove_embeddings.py --vocab-file vocab.txt --cooccurr-file cooccurence.shuf.bin --embeddings-size 100 --embs-save-path glove_vectors.pt --glove-vectors-save-path glove_vectors.txt
```

Notice that the script also exposes a [tinygrad](https://github.com/tinygrad/tinygrad) model, in case you have an AMD or Intel GPU.
You can run it with:

```bash
python evaluating_trajectories/experiments/train_glove_embeddings.py --vocab-file vocab.txt --cooccurr-file cooccurence.shuf.bin --embeddings-size 100 --embs-save-path glove_vectors.pt --glove-vectors-save-path glove_vectors.txt --use-tinygrad
```

### Using GloVe embeddings

The generated embeddings will be in a file called `vectors.txt` or `glove_vectors.txt`.
To see an example on how to load or use the embeddings, look at the `bin/embedding_visualization.py` script.
