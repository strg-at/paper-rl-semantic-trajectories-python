import tinygrad as tg


class GloVeTG:
    def __init__(self, vocab_size: int, embedding_size: int, x_max: int, alpha: float):
        self.weight = tg.nn.Embedding(vocab_size=vocab_size, embed_size=embedding_size)
        self.weight_tilde = tg.nn.Embedding(vocab_size=vocab_size, embed_size=embedding_size)
        self.bias = tg.Tensor.uniform(vocab_size)
        self.bias_tilde = tg.Tensor.uniform(vocab_size)
        self.weighting_func = lambda x: (x / x_max).pow(alpha).clamp(0, 1)

    def __call__(self, i: tg.Tensor, j: tg.Tensor, x: tg.Tensor):
        loss = (self.weight(i) * self.weight_tilde(j)).sum(axis=-1)
        loss = (loss + self.bias[i] + self.bias_tilde[j] - x.log()).square()
        loss = self.weighting_func(x).mul(loss).mean()
        return loss
