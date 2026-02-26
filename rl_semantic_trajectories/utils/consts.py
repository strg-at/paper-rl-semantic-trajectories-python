import typing


# WARNING: If you change this, be aware that this will invalidate the computed embeddings. They should be recomputed with the new model, and no new experiments should be
# run before that is done.
SENTENCE_TRANSFORMER_MODEL_NAME: typing.Final[str] = "paraphrase-multilingual-MiniLM-L12-v2"
