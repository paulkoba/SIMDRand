SIMDRand is small header-only collection of fast pseudorandom number generators.\
Usage example:
```CPP
#include <xorshift128plus.h>
...
xorshift128plus::xorshift128plus generator(time(0));
uint64_t random_uint = generator.next();
xorshift128plus::xorshift128plus_4 generator_simd(time(0));
__m256i_u random_vector = generator_simd.next();
...
```
Comparison of performance of several different random number generators:

| Random generator                   | Required instruction sets | Throughput | Time per operation |
|------------------------------------|---------------------------|------------|--------------------|
| rand()                             |                           | 4.3 Gb/s   | 7.35 ns            |
| std::mt19937_64                    |                           | 8.7 Gb/s   | 7.33 ns            |
| splitmix::splitmix64               |                           | 59.7 Gb/s  | 1.07 ns            |
| xorshift64::xorshift64             |                           | 34.8 Gb/s  | 1.84 ns            |
| xorshift64::xorshift64_2           | AVX                       | 68.8 Gb/s  | 1.86 ns            |
| xorshift64::xorshift64_4           | AVX-2                     | 136.0 Gb/s | 1.88 ns            |
| xorshift128plus::xorshift128plus   |                           | 50.1 Gb/s  | 1.28 ns            |
| xorshift128plus::xorshift128plus_2 | AVX                       | 104.8 Gb/s | 1.22 ns            |
| xorshift128plus::xorshift128plus_4 | AVX-2                     | 206.6 Gb/s | 1.24 ns            |
