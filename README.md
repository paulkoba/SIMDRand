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
Comparison of performance of several different random number generators on test machine:

| Random generator                             | Required instruction sets | Throughput   | Time per operation |
|----------------------------------------------|---------------------------|--------------|--------------------|
| std::mt19937_64                              |                           | 30.31 Gb/s   | 2.112 ns           |
| splitmix::splitmix64                         |                           | 52.053 Gb/s  | 1.23 ns            |
| xorshift64::xorshift64                       |                           | 34.653 Gb/s  | 1.847 ns           |
| xoroshiro128plus::xoroshiro128plus           |                           | 48.07 Gb/s   | 1.331 ns           |
| xorshift128plus::xorshift128plus             |                           | 46.198 Gb/s  | 1.385 ns           |
| xoroshiro128plusplus::xoroshiro128plusplus   |                           | 43.998 Gb/s  | 1.455 ns           |
| xoshiro256ss::xoshiro256ss                   |                           | 56.935 Gb/s  | 1.124 ns           |
| xoshiro256plusplus::xoshiro256plusplus       |                           | 56.612 Gb/s  | 1.13 ns            |
| xoshiro256plusplus::xoshiro256plusplus_2     | AVX                       | 67.937 Gb/s  | 1.884 ns           |
| xoshiro256plusplus::xoshiro256ss_2           | AVX                       | 59.987 Gb/s  | 2.134 ns           |
| xoroshiro128plusplus::xoroshiro128plusplus_2 | AVX                       | 59.724 Gb/s  | 2.143 ns           |
| xoroshiro128plusplus::xoroshiro128plus_2     | AVX                       | 78.167 Gb/s  | 1.638 ns           |
| xorshift64::xorshift64_2                     | AVX                       | 69.928 Gb/s  | 1.83 ns            |
| xorshift128plus::xorshift128plus_2           | AVX                       | 98.661 Gb/s  | 1.297 ns           |
| xoshiro256plusplus::xoshiro256plusplus_4     | AVX-2                     | 135.834 Gb/s | 1.885 ns           |
| xoshiro256plusplus::xoshiro256ss_4           | AVX-2                     | 119.99 Gb/s  | 2.134 ns           |
| xoroshiro128plusplus::xoroshiro128plusplus_4 | AVX-2                     | 119.527 Gb/s | 2.142 ns           |
| xoroshiro128plusplus::xoroshiro128plus_4     | AVX-2                     | 156.195 Gb/s | 1.639 ns           |
| xorshift64::xorshift64_4                     | AVX-2                     | 139.962 Gb/s | 1.829 ns           |
| xorshift128plus::xorshift128plus_4           | AVX-2                     | 197.166 Gb/s | 1.298 ns           |
