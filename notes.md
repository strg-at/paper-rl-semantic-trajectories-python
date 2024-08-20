# Paper

## Environments

- One environment based on text embeddings;
- One environment based on GloVe embeddings:
  - can we check iid of trajectories (occupancy)?
  - we can train the embeddings on all the data, or just the training set. In former case, we can only use it for evaluation.

## Training and evaluation

- Train IQ Learn agents on both environments;
- Evaluate with:
  - WS distance, comparing distributions: not time aware nor semantic aware;
  - Needleman-Wunsch based distance;
  - Other sequence matching algorithms -> only for related works;
- Evaluation should be done taking into account different "time frames", e.g.:
  - training on one month, testing on the others;
  - viceversa;
  - leveraging domain specific knowledge: testing against christmas/black friday and viceversa;

## Nice to have

- Training RL agent using a NW based reward <- how do we evaluate this? Not easy;
- Training a transformer with a masked node objective <- very easy to do
