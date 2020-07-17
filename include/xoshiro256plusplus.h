#ifndef XOSHIRO256PLUSPLUS_H_INCLUDED
#define XOSHIRO256PLUSPLUS_H_INCLUDED

#include <cstdint>

#include "splitmix.h"

namespace xoshiro256plusplus {

#ifdef __AVX__

//xoshiro256++ implementation using AVX instruction set to generate random __m128i_u.
struct xoshiro256plusplus_2 {
    xoshiro256plusplus_2(__m128i_u a, __m128i_u b, __m128i_u c, __m128i_u d) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
        m_state[2] = c;
        m_state[3] = d;
    }

    xoshiro256plusplus_2(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f, uint64_t g, uint64_t h) noexcept
    {
        m_state[0] = _mm_set_epi64x(a, b);
        m_state[1] = _mm_set_epi64x(c, d);
        m_state[2] = _mm_set_epi64x(e, f);
        m_state[3] = _mm_set_epi64x(g, h);
    }

    explicit xoshiro256plusplus_2(splitmix::splitmix64 gen) noexcept
    {
        for (size_t i = 0; i < 4; ++i)
            m_state[i] = _mm_set_epi64x(gen.next(), gen.next());
    }

    //Generates random __m128i_u.
    __m128i_u next() noexcept
    {
        const __m128i_u result = _mm_add_epi64(rotl(_mm_add_epi64(m_state[0], m_state[3]), 23), m_state[0]);

        const __m128i_u t = _mm_slli_epi64(m_state[1], 17);

        m_state[2] = _mm_xor_si128(m_state[0], m_state[2]);
        m_state[3] = _mm_xor_si128(m_state[1], m_state[3]);
        m_state[1] = _mm_xor_si128(m_state[2], m_state[1]);
        m_state[0] = _mm_xor_si128(m_state[3], m_state[0]);

        m_state[2] = _mm_xor_si128(t, m_state[2]);

        m_state[3] = rotl(m_state[3], 45);

        return result;
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xoshiro256plusplus_2& other) const noexcept
    {
        //TODO: Check whether its faster than creating all 4 masks simultaniously and then comparing all of them with 0xffffU in return statement.
        for (size_t i = 0; i < 4; ++i) {
            __m128i_u cmp = _mm_cmpeq_epi32(other.m_state[i], m_state[i]);
            uint16_t mask = _mm_movemask_epi8(cmp);
            if (mask != 0xffffU)
                return false;
        }

        return true;
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xoshiro256plusplus_2& other) const noexcept
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

    __m128i_u m_state[4];
};

#ifdef __AVX2__

struct xoshiro256plusplus_4 {
    xoshiro256plusplus_4(__m256i_u a, __m256i_u b, __m256i_u c, __m256i_u d) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
        m_state[0] = c;
        m_state[1] = d;
    }

    explicit xoshiro256plusplus_4(splitmix::splitmix64 gen) noexcept
    {
        for (int i = 0; i < 4; ++i)
            m_state[i] = _mm256_set_epi64x(gen.next(), gen.next(), gen.next(), gen.next());
    }

    //Generates random __m256i_u.
    __m256i_u next() noexcept
    {
        const __m256i_u result = _mm256_add_epi64(rotl(_mm256_add_epi64(m_state[0], m_state[3]), 23), m_state[0]);

        const __m256i_u t = _mm256_slli_epi64(m_state[1], 17);

        m_state[2] = _mm256_xor_si256(m_state[0], m_state[2]);
        m_state[3] = _mm256_xor_si256(m_state[1], m_state[3]);
        m_state[1] = _mm256_xor_si256(m_state[2], m_state[1]);
        m_state[0] = _mm256_xor_si256(m_state[3], m_state[0]);

        m_state[2] = _mm256_xor_si256(t, m_state[2]);

        m_state[3] = rotl(m_state[3], 45);

        return result;
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xoshiro256plusplus_4& other) const noexcept
    {
        for (size_t i = 0; i < 4; ++i) {
            __m256i_u cmp = _mm256_cmpeq_epi32(other.m_state[i], m_state[i]);
            uint16_t mask = _mm256_movemask_epi8(cmp);
            if (mask != 0xffffffffU)
                return false;
        }

        return true;
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xoshiro256plusplus_4& other) const noexcept
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

    __m256i_u m_state[4];
};

#ifdef __AVX512F__

//xoroshiro256++ implementation using AVX512F instruction set to generate random __m512i_u.
struct xoshiro256plusplus_8 {
    xoshiro256plusplus_8(__m512i_u a, __m512i_u b, __m512i_u c, __m512i_u d) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
        m_state[2] = c;
        m_state[3] = d;
    }

    explicit xoshiro256plusplus_8(splitmix::splitmix64 gen) noexcept
    {
        for (size_t i = 0; i < 4; ++i)
            m_state[i] = _mm512_set_epi64(gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next());
    }

    //Generates random __m512i_u.
    __m512i_u next() noexcept
    {
        const __m512i_u result = _mm512_add_epi64(rotl(_mm512_add_epi64(m_state[0], m_state[3]), 23), m_state[0]);

        const __m512i_u t = _mm512_slli_epi64(m_state[1], 17);

        m_state[2] = _mm512_xor_si512(m_state[0], m_state[2]);
        m_state[3] = _mm512_xor_si512(m_state[1], m_state[3]);
        m_state[1] = _mm512_xor_si512(m_state[2], m_state[1]);
        m_state[0] = _mm512_xor_si512(m_state[3], m_state[0]);

        m_state[2] = _mm512_xor_si512(t, m_state[2]);

        m_state[3] = rotl(m_state[3], 45);

        return result;
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xoshiro256plusplus_8& other) const noexcept
    {
        for (size_t i = 0; i < 4; ++i) {
            __mmask8 cmp = _mm512_cmpeq_epi64_mask(other.m_state[i], m_state[i]);
            uint16_t mask = _mm_movemask_epi8(_mm_movm_epi64(cmp));
            if (mask != 0xffffU)
                return false;
        }

        return true;
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xoshiro256plusplus_8& other) const noexcept
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

    __m512i_u m_state[4];
};

#endif // __AVX512F__

#endif

#endif

struct xoshiro256plusplus {
    xoshiro256plusplus(uint64_t a, uint64_t b, uint64_t c, uint64_t d) noexcept
    {
        m_state[0] = a;
        m_state[1] = b;
        m_state[2] = c;
        m_state[3] = d;
    }

    explicit xoshiro256plusplus(splitmix::splitmix64 gen) noexcept
    {
        for (size_t i = 0; i < 4; ++i)
            m_state[i] = gen.next();
    }

    //Generates random uint64_t.
    constexpr uint64_t next() noexcept
    {
        const uint64_t result = rotl(m_state[0] + m_state[1], 23) + m_state[0];

        const uint64_t t = m_state[1] << 17;

        m_state[2] ^= m_state[0];
        m_state[3] ^= m_state[1];
        m_state[1] ^= m_state[2];
        m_state[0] ^= m_state[3];

        m_state[2] ^= t;

        m_state[3] = rotl(m_state[3], 45);

        return result;
    }

    //Compares internal states of two engines for equality.
    constexpr bool operator==(const xoshiro256plusplus& other) const noexcept { return m_state[0] == other.m_state[0] && m_state[1] == other.m_state[1] && m_state[2] == other.m_state[2] && m_state[3] == other.m_state[3]; }

    //Compares internal states of two engines for inequality.
    constexpr bool operator!=(const xoshiro256plusplus& other) const noexcept { return m_state[0] != other.m_state[0] || m_state[1] != other.m_state[1] || m_state[2] != other.m_state[2] || m_state[3] != other.m_state[3]; }

private:
    uint64_t m_state[4];

    constexpr static inline uint64_t rotl(const uint64_t x, int k) noexcept
    {
        return (x << k) | (x >> (64 - k));
    }
};
}

#endif // XOSHIRO256PLUSPLUS_H_INCLUDED
