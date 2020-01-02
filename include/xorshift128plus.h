#ifndef XORSHIFT128PLUS_H_INCLUDED
#define XORSHIFT128PLUS_H_INCLUDED

//Reference implementation: http://vigna.di.unimi.it/ftp/papers/xorshiftplus.pdf

#include <cstdint>

#include "splitmix.h"

namespace xorshift128plus {

#ifdef __AVX__

//xorshift128plus implementation using AVX to generate random __m128i_u.
struct xorshift128plus_2 {
    xorshift128plus_2(__m128i_u a, __m128i_u b) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
    }

    xorshift128plus_2(uint64_t a, uint64_t b, uint64_t c, uint64_t d) noexcept
    {
        m_state[0] = _mm_set_epi64x(a, b);
        m_state[1] = _mm_set_epi64x(c, d);
    }

    explicit xorshift128plus_2(splitmix::splitmix64 gen) noexcept
    {
        m_state[0] = _mm_set_epi64x(gen.next(), gen.next());
        m_state[1] = _mm_set_epi64x(gen.next(), gen.next());
    }

    //Generates random __m128i_u.
    __m128i_u next() noexcept
    {
        __m128i_u s1 = m_state[0];
        __m128i_u s0 = m_state[1];
        m_state[0] = s0;
        s1 = _mm_xor_si128(_mm_slli_epi64(s1, 23), s1);
        m_state[1] = _mm_xor_si128(_mm_xor_si128(_mm_srli_epi64(s1, 26), s0), _mm_xor_si128(_mm_srli_epi64(s0, 17), s1));
        return _mm_add_epi64(m_state[0], m_state[1]);
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xorshift128plus_2& other) const noexcept
    {
        __m128i_u cmp0 = _mm_cmpeq_epi32(other.m_state[0], m_state[0]);
        __m128i_u cmp1 = _mm_cmpeq_epi32(other.m_state[1], m_state[1]);
        uint16_t mask0 = _mm_movemask_epi8(cmp0);
        uint16_t mask1 = _mm_movemask_epi8(cmp1);
        return (mask0 == 0xffffU && mask1 == 0xffffU);
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xorshift128plus_2& other) const noexcept
    {
        return !(*this == other);
    }

private:
    __m128i_u m_state[2];
};

#ifdef __AVX2__

//xorshift128plus implementation using AVX-2 to generate random __m256i_u.
struct xorshift128plus_4 {
    xorshift128plus_4(__m256i_u a, __m256i_u b) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
    }

    explicit xorshift128plus_4(splitmix::splitmix64 gen) noexcept
    {
        m_state[0] = _mm256_set_epi64x(gen.next(), gen.next(), gen.next(), gen.next());
        m_state[1] = _mm256_set_epi64x(gen.next(), gen.next(), gen.next(), gen.next());
    }

    //Generates random __m256i_u.
    __m256i_u next() noexcept
    {
        __m256i_u s1 = m_state[0];
        __m256i_u s0 = m_state[1];
        m_state[0] = s0;
        s1 = _mm256_xor_si256(_mm256_slli_epi64(s1, 23), s1);
        m_state[1] = _mm256_xor_si256(_mm256_xor_si256(_mm256_srli_epi64(s1, 26), s0), _mm256_xor_si256(_mm256_srli_epi64(s0, 17), s1));
        return _mm256_add_epi32(m_state[0], m_state[1]);
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xorshift128plus_4& other) const noexcept
    {
        __m256i_u cmp0 = _mm256_cmpeq_epi32(other.m_state[0], m_state[0]);
        __m256i_u cmp1 = _mm256_cmpeq_epi32(other.m_state[1], m_state[1]);
        uint32_t mask0 = _mm256_movemask_epi8(cmp0);
        uint32_t mask1 = _mm256_movemask_epi8(cmp1);
        return (mask0 == 0xffffffffU && mask1 == 0xffffffffU);
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xorshift128plus_4& other) const noexcept
    {
        return !(*this == other);
    }

private:
    __m256i_u m_state[2];
};

#ifdef __AVX512F__

//xorshift128plus implementation using AVX512F to generate random __m512i_u.
struct xorshift128plus_8 {
    xorshift128plus_8(__m512i_u a, __m512i_u b) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
    }

    explicit xorshift128plus_8(splitmix::splitmix64 gen) noexcept
    {
        m_state[0] = _mm512_set_epi64(gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next());
        m_state[1] = _mm512_set_epi64(gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next());
    }

    //Generates random __m512i_u.
    __m512i_u next() noexcept
    {
        __m512i_u s1 = m_state[0];
        __m512i_u s0 = m_state[1];
        m_state[0] = s0;
        s1 = _mm512_xor_si512(_mm512_slli_epi64(s1, 23), s1);
        m_state[1] = _mm512_xor_si512(_mm512_xor_si512(_mm512_srli_epi64(s1, 26), s0), _mm512_xor_si512(_mm512_srli_epi64(s0, 17), s1));
        return _mm512_add_epi32(m_state[0], m_state[1]);
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xorshift128plus_8& other) const noexcept
    {
        __mmask8 cmp0 = _mm512_cmpeq_epi64_mask(other.m_state[0], m_state[0]);
        __mmask8 cmp1 = _mm512_cmpeq_epi64_mask(other.m_state[1], m_state[1]);
        //TODO: proper fast implementation.
        uint16_t mask0 = _mm_movemask_epi8(_mm_movm_epi64(cmp0));
        uint16_t mask1 = _mm_movemask_epi8(_mm_movm_epi64(cmp1));
        return (mask0 == 0xffffU && mask1 == 0xffffU);
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xorshift128plus_8& other) const noexcept
    {
        return !(*this == other);
    }

private:
    __m512i_u m_state[2];
};

#endif // __AVX512F__
#endif // __AVX2__
#endif // __AVX__

//xorshift128plus implementation used to generate random uint64_t.
struct xorshift128plus {
    xorshift128plus(uint64_t a, uint64_t b) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
    }

    explicit xorshift128plus(splitmix::splitmix64 gen) noexcept
    {
        m_state[0] = gen.next();
        m_state[1] = gen.next();
    }

    //Generates random uint64_t.
    constexpr uint64_t next() noexcept
    {
        uint64_t s1 = m_state[0];
        const uint64_t s0 = m_state[1];
        m_state[0] = s0;
        s1 ^= s1 << 23;
        m_state[1] = s1 ^ s0 ^ (s1 >> 26) ^ (s0 >> 17);
        return m_state[1] + s0;
    }

    //Compares internal states of two engines for equality.
    constexpr bool operator==(const xorshift128plus& other) const noexcept { return m_state[0] == other.m_state[0] && m_state[1] == other.m_state[1]; }

    //Compares internal states of two engines for inequality.
    constexpr bool operator!=(const xorshift128plus& other) const noexcept { return m_state[0] != other.m_state[0] || m_state[1] != other.m_state[1]; }

private:
    uint64_t m_state[2];
};

}

#endif // XORSHIFT128PLUS_H_INCLUDED
