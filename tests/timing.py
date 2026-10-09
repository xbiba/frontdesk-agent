"""The wall-clock bound shared by the tests that guard against running time that grows too fast."""

# A 100,000-character message must take time linear in its length. The slowest case here, text
# whose characters expand when normalized, takes 0.2 s, or 1.0 s when a tracer such as coverage
# slows the pure-Python loops (one laptop). The quadratic personal-data patterns take about 23 s
# at that size, so 5 s tells the two apart plain, traced and on a slower machine.
LINEAR_TIME_LIMIT_SECONDS = 5.0
