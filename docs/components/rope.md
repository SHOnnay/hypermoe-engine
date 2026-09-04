# Rotary position embedding

`RoPE` is a CPU reference transformation over contiguous token/head data. It
rotates adjacent feature pairs using a configurable theta and absolute sequence
position. Token count, head count, even head dimension, position range, and
storage size are validated before mutation.

The attention configuration applies RoPE independently to all query heads and
key/value key heads before keys enter the cache. The correctness oracle contains
a separate scalar implementation so tests do not validate RoPE by calling the
production primitive twice.

This foundation uses one global theta and the adjacent-pair convention. Scaling
variants, partial rotary dimensions, and model-specific frequency layouts must
come from explicitly imported architecture metadata rather than name-based
runtime guesses.
