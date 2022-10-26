def batch_iter(iterable, batch_size: int = 1):
    l = len(iterable)
    for offset in range(0, l, batch_size):
        yield iterable[offset : min(offset + batch_size, l)]
