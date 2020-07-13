#ifndef XOROSHIRO128PLUS_H_INCLUDED
#define XOROSHIRO128PLUS_H_INCLUDED

#include <cstdint>

#include "splitmix.h"

namespace xoroshiro128plus {

#ifdef __AVX__

//xoroshiro128+ implementation using AVX instruction set to generate random __m128i_u.
struct xoroshiro128plus_2 {
    xoroshiro128plus_2(__m128i_u a, __m128i_u b) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
    }

    xoroshiro128plus_2(uint64_t a, uint64_t b, uint64_t c, uint64_t d) noexcept
    {
        m_state[0] = _mm_set_epi64x(a, b);
        m_state[1] = _mm_set_epi64x(c, d);
    }

    explicit xoroshiro128plus_2(splitmix::splitmix64 gen) noexcept
    {
        m_state[0] = _mm_set_epi64x(gen.next(), gen.next());
        m_state[1] = _mm_set_epi64x(gen.next(), gen.next());
    }

    //Generates random __m128i_u.
    __m128i_u next() noexcept
    {
        const __m128i_u s0 = m_state[0];
        __m128i_u s1 = m_state[1];
        const __m128i_u result = _mm_add_epi64(s0, s1);

        s1 = _mm_xor_si128(s1, s0);
        m_state[0] = _mm_xor_si128(rotl(s0, 24), _mm_xor_si128(s1, _mm_slli_epi64(s1, 16)));
        m_state[1] = rotl(s1, 37);

        return result;
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xoroshiro128plus_2& other) const noexcept
    {
        __m128i_u cmp0 = _mm_cmpeq_epi32(other.m_state[0], m_state[0]);
        __m128i_u cmp1 = _mm_cmpeq_epi32(other.m_state[1], m_state[1]);
        uint16_t mask0 = _mm_movemask_epi8(cmp0);
        uint16_t mask1 = _mm_movemask_epi8(cmp1);
        return (mask0 == 0xffffU && mask1 == 0xffffU);
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xoroshiro128plus_2& other) const noexcept
    {
        return !(*this == other);
    }

private:
    static inline __m128i_u rotl(const __m128i_u x, int k) noexcept
    {
        __m128i_u a = _mm_slli_epi64(x, k);
        __m128i_u b = _mm_srli_epi64(x, 64 - k);
        return _mm_or_si128(a, b);
    }

    __m128i_u m_state[2];
};

#ifdef __AVX2__

struct xoroshiro128plus_4 {
    xoroshiro128plus_4(__m256i_u a, __m256i_u b) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
    }

    explicit xoroshiro128plus_4(splitmix::splitmix64 gen) noexcept
    {
        m_state[0] = _mm256_set_epi64x(gen.next(), gen.next(), gen.next(), gen.next());
        m_state[1] = _mm256_set_epi64x(gen.next(), gen.next(), gen.next(), gen.next());
    }

    //Generates random __m256i_u.
    __m256i_u next() noexcept
    {
        const __m256i_u s0 = m_state[0];
        __m256i_u s1 = m_state[1];
        const __m256i_u result = _mm256_add_epi64(s0, s1);

        s1 = _mm256_xor_si256(s1, s0);
        m_state[0] = _mm256_xor_si256(rotl(s0, 24), _mm256_xor_si256(s1, _mm256_slli_epi64(s1, 16)));
        m_state[1] = rotl(s1, 37);

        return result;
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xoroshiro128plus_4& other) const noexcept
    {
        __m256i_u cmp0 = _mm256_cmpeq_epi32(other.m_state[0], m_state[0]);
        __m256i_u cmp1 = _mm256_cmpeq_epi32(other.m_state[1], m_state[1]);
        uint32_t mask0 = _mm256_movemask_epi8(cmp0);
        uint32_t mask1 = _mm256_movemask_epi8(cmp1);
        return (mask0 == 0xffffffffU && mask1 == 0xffffffffU);
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xoroshiro128plus_4& other) const noexcept
    {
        return !(*this == other);
    }

private:
    static inline __m256i_u rotl(const __m256i_u x, int k) noexcept
    {
        __m256i_u a = _mm256_slli_epi64(x, k);
        __m256i_u b = _mm256_srli_epi64(x, 64 - k);
        return _mm256_or_si256(a, b);
    }

    __m256i_u m_state[2];
};

#ifdef __AVX512F__

//xoroshiro128+ implementation using AVX512F instruction set to generate random __m512i_u.
struct xoroshiro128plus_8 {
    xoroshiro128plus_8(__m512i_u a, __m512i_u b) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
    }

    explicit xoroshiro128plus_8(splitmix::splitmix64 gen) noexcept
    {
        m_state[0] = _mm512_set_epi64(gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next());
        m_state[1] = _mm512_set_epi64(gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next());
    }

    //Generates random __m512i_u.
    __m512i_u next() noexcept
    {
        const __m512i_u s0 = m_state[0];
        __m512i_u s1 = m_state[1];
        const __m512i_u result = _mm512_add_epi64(s0, s1);

        s1 = _mm512_xor_si512(s1, s0);
        m_state[0] = _mm512_xor_si512(rotl(s0, 24), _mm512_xor_si512(s1, _mm512_slli_epi64(s1, 16)));
        m_state[1] = rotl(s1, 37);

        return result;
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xoroshiro128plus_8& other) const noexcept
    {
        __mmask8 cmp0 = _mm512_cmpeq_epi64_mask(other.m_state[0], m_state[0]);
        __mmask8 cmp1 = _mm512_cmpeq_epi64_mask(other.m_state[1], m_state[1]);
        //TODO: proper fast implementation.
        uint16_t mask0 = _mm_movemask_epi8(_mm_movm_epi64(cmp0));
        uint16_t mask1 = _mm_movemask_epi8(_mm_movm_epi64(cmp1));
        return (mask0 == 0xffffU && mask1 == 0xffffU);
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xoroshiro128plus_8& other) const noexcept
    {
        return !(*this == other);
    }

private:
    static inline __m512i_u rotl(const __m512i_u x, int k) noexcept
    {
        __m512i_u a = _mm512_slli_epi64(x, k);
        __m512i_u b = _mm512_srli_epi64(x, 64 - k);
        return _mm512_or_si512(a, b);
    }

    __m512i_u m_state[2];
};

#endif // __AVX512F__

#endif

#endif

struct xoroshiro128plus {
    xoroshiro128plus(uint64_t a, uint64_t b) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
    }

    explicit xoroshiro128plus(splitmix::splitmix64 gen) noexcept
    {
        m_state[0] = gen.next();
        m_state[1] = gen.next();
    }

    //Generates random uint64_t.
    constexpr uint64_t next() noexcept
    {
        const uint64_t s0 = m_state[0];
        uint64_t s1 = m_state[1];
        const uint64_t result = s0 + s1;

        s1 ^= s0;
        m_state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
        m_state[1] = rotl(s1, 37); // c

        return result;
    }

    //Compares internal states of two engines for equality.
    constexpr bool operator==(const xoroshiro128plus& other) const noexcept { return m_state[0] == other.m_state[0] && m_state[1] == other.m_state[1]; }

    //Compares internal states of two engines for inequality.
    constexpr bool operator!=(const xoroshiro128plus& other) const noexcept { return m_state[0] != other.m_state[0] || m_state[1] != other.m_state[1]; }

private:
    uint64_t m_state[2];

    constexpr static inline uint64_t rotl(const uint64_t x, int k) noexcept
    {
        return (x << k) | (x >> (64 - k));
    }
};
}

#endif // XOROSHIRO128PLUS_H_INCLUDED
