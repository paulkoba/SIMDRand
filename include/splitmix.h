#ifndef SPLITMIX_H_INCLUDED
#define SPLITMIX_H_INCLUDED

//Reference implementation: http://xoshiro.di.unimi.it/splitmix64.c

#include <immintrin.h>
#include <nmmintrin.h>
#include <cstdint>

namespace splitmix {

//AVX512VL and AVX512DQ are required for _mm256_mullo_epi64() and _mm_mullo_epi64(), so not vectorizable only with AVX or AVX2.
#if defined __AVX512VL__ && defined __AVX512DQ__
#ifdef __AVX512F__

const __m128i_u _add_avx = _mm_set1_epi64x(0x9e3779b97f4a7c15 * 2);
const __m256i_u _add_avx2 = _mm256_set1_epi64x(0x9e3779b97f4a7c15 * 4);
const __m512i_u _add_avx512 = _mm512_set1_epi64(0x9e3779b97f4a7c15 * 8);

const __m128i_u _mul1_avx = _mm_set1_epi64x(0xbf58476d1ce4e5b9);
const __m256i_u _mul1_avx2 = _mm256_set1_epi64x(0xbf58476d1ce4e5b9);
const __m512i_u _mul1_avx512 = _mm512_set1_epi64(0xbf58476d1ce4e5b9);

const __m128i_u _mul2_avx = _mm_set1_epi64x(0x94d049bb133111eb);
const __m256i_u _mul2_avx2 = _mm256_set1_epi64x(0x94d049bb133111eb);
const __m512i_u _mul2_avx512 = _mm512_set1_epi64(0x94d049bb133111eb);

//splitmix64 implementation using AVL-512F, AVX-512VL and AVX-512DQ sets to generate random __m512i_u.
struct splitmix64_8 {
    explicit splitmix64_8(__m512i_u state) noexcept
        : m_state(state)
    {
    }

    splitmix64_8(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f, uint64_t g, uint64_t h) noexcept
    {
        m_state = _mm512_set_epi64(a, b, c, d, e, f, g, h);
    }

    explicit splitmix64_8(uint64_t a) noexcept
    {
        m_state = _mm512_set_epi64(a, a + 0x9e3779b97f4a7c15, a + 2 * 0x9e3779b97f4a7c15, a + 3 * 0x9e3779b97f4a7c15, a + 4 * 0x9e3779b97f4a7c15,
            a + 5 * 0x9e3779b97f4a7c15, a + 6 * 0x9e3779b97f4a7c15, a + 7 * 0x9e3779b97f4a7c15);
    }

    explicit constexpr operator __m512i_u() const noexcept { return m_state; }

    //Generates random __m512i_u.
    __m512i_u next() noexcept
    {
        __m512i_u z = m_state = _mm512_add_epi64(_add_avx512, m_state);
        z = _mm512_mullo_epi64(_mul1_avx512, _mm512_xor_si512(z, _mm512_srli_epi64(z, 30)));
        z = _mm512_mullo_epi64(_mul2_avx512, _mm512_xor_si512(z, _mm512_srli_epi64(z, 27)));
        return _mm512_xor_si512(z, _mm512_srli_epi64(z, 31));
    }

    //Compares internal states of two engines for equality.
    bool operator==(const splitmix64_8& other) const noexcept
    {
        __mmask8 cmp0 = _mm512_cmpeq_epi64_mask(other.m_state, m_state);
        //TODO: proper fast implementation.
        uint16_t mask0 = _mm_movemask_epi8(_mm_movm_epi64(cmp0));
        return (mask0 == 0xffffU);
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const splitmix64_8& other) const noexcept
    {
        return !(*this == other);
    }

private:
    __m512i_u m_state;
};

#endif // __AVX512F__

//splitmix64 implementation using AVX-512VL and AVX-512DQ sets to generate random __m256i_u.
struct splitmix64_4 {
    explicit splitmix64_4(__m256i_u state) noexcept
        : m_state(state)
    {
    }

    splitmix64_4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) noexcept
    {
        m_state = _mm256_set_epi64x(a, b, c, d);
    }

    explicit splitmix64_4(uint64_t a) noexcept
    {
        m_state = _mm256_set_epi64x(a, a + 0x9e3779b97f4a7c15, a + 2 * 0x9e3779b97f4a7c15, a + 3 * 0x9e3779b97f4a7c15);
    }

    explicit constexpr operator __m256i_u() const noexcept { return m_state; }

    //Generates random __m256i_u.
    __m256i_u next() noexcept
    {
        __m256i_u z = m_state = _mm256_add_epi64(_add_avx2, m_state);
        z = _mm256_mullo_epi64(_mul1_avx2, _mm256_xor_si256(z, _mm256_srli_epi64(z, 30)));
        z = _mm256_mullo_epi64(_mul2_avx2, _mm256_xor_si256(z, _mm256_srli_epi64(z, 27)));
        return _mm256_xor_si256(z, _mm256_srli_epi64(z, 31));
    }

    //Compares internal states of two engines for equality.
    bool operator==(const splitmix64_4& other) const noexcept
    {
        __m256i_u cmp0 = _mm256_cmpeq_epi32(other.m_state, m_state);
        unsigned mask0 = _mm256_movemask_epi8(cmp0);
        return mask0 == 0xffffffffU;
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const splitmix64_4& other) const noexcept
    {
        return !(*this == other);
    }

private:
    __m256i_u m_state;
};

//splitmix64 implementation using AVX-512VL and AVX-512DQ sets to generate random __m128i_u.
struct splitmix64_2 {
    explicit splitmix64_2(__m128i_u state) noexcept
        : m_state(state)
    {
    }

    splitmix64_2(uint64_t a, uint64_t b) noexcept
    {
        m_state = _mm_set_epi64x(a, b);
    }

    explicit splitmix64_2(uint64_t a) noexcept
    {
        m_state = _mm_set_epi64x(a, a + 0x9e3779b97f4a7c15);
    }

    explicit constexpr operator __m128i_u() const noexcept { return m_state; }

    //Generates random __m128i_u.
    __m128i_u next() noexcept
    {
        __m128i_u z = m_state = _mm_add_epi64(_add_avx, m_state);
        z = _mm_mullo_epi64(_mul1_avx, _mm_xor_si128(z, _mm_srli_epi64(z, 30)));
        z = _mm_mullo_epi64(_mul2_avx, _mm_xor_si128(z, _mm_srli_epi64(z, 27)));
        return _mm_xor_si128(z, _mm_srli_epi64(z, 31));
    }

    //Compares internal states of two engines for equality.
    bool operator==(const splitmix64_2& other) const noexcept
    {
        __m128i_u cmp = _mm_cmpeq_epi8(m_state, other.m_state);
        uint16_t test = _mm_movemask_epi8(cmp);
        return test == 0xffffU;
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const splitmix64_2& other) const noexcept
    {
        return !(*this == other);
    }

private:
    __m128i_u m_state;
};

#endif // defined

//splitmix64 implementation used to generate random uint64_t.
struct splitmix64 {
    splitmix64(uint64_t state) noexcept
        : m_state(state)
    {
    }

    constexpr operator uint64_t() const noexcept { return m_state; }

    //Generates random uint64_t.
    constexpr uint64_t next() noexcept
    {
        uint64_t z = (m_state += 0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

    //Compares internal states of two engines for equality.
    constexpr bool operator==(const splitmix64& other) const noexcept { return other.m_state == m_state; }

    //Compares internal states of two engines for inequality.
    constexpr bool operator!=(const splitmix64& other) const noexcept { return other.m_state != m_state; }

private:
    uint64_t m_state;
};

}
#endif // SPLITMIX_H_INCLUDED
