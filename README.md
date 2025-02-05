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

This project uses [Python Poetry](https://python-poetry.org/docs/basic-usage/) to manage its dependencies.

In order to install the project run:

```bash
poetry install --with dev
```

If you want to use an Nvidia GPU, please follow the instructions in the comments in the `pyproject.toml` file and then run:

```bash
poetry install --with release
```

You can then activate the environment sourcing the `activate.sh` script at the root of this repository:

```bash
source ./activate.sh
```

Finally, run:

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

Then, you can generate the `glove_trajectories.txt` file with:

```bash
python bin/glove_preprocessing.py --path alldata.parquet --output-file glove_trajectories.txt --min-trajectory-length 3
```

Feel free to adjust the paremeters to your liking.

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

To see an example on how to load or use the embeddings, look at the `bin/embedding_visualization.py` script.
