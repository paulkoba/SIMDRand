#ifndef XORSHIFT64_H_INCLUDED
#define XORSHIFT64_H_INCLUDED

//Reference implementation: https://en.wikipedia.org/wiki/Xorshift

#include <cstdint>

#include "splitmix.h"

namespace xorshift64 {

#ifdef __AVX__

//xorshift64 implementation using AVX to generate random __m128i_u.
struct xorshift64_2 {
    explicit xorshift64_2(__m128i_u state) noexcept
        : m_state(state)
    {
    }

    xorshift64_2(uint64_t a, uint64_t b) noexcept
    {
        m_state = _mm_set_epi64x(a, b);
    }

    explicit xorshift64_2(splitmix::splitmix64 gen) noexcept
    {
        m_state = _mm_set_epi64x(gen.next(), gen.next());
    }

    constexpr operator __m128i_u() const noexcept { return m_state; }

    //Generates random __m128i_u.
    __m128i_u next() noexcept
    {
        m_state = _mm_xor_si128(m_state, _mm_srli_epi64(m_state, 13));
        m_state = _mm_xor_si128(m_state, _mm_slli_epi64(m_state, 17));
        m_state = _mm_xor_si128(m_state, _mm_srli_epi64(m_state, 5));
        return m_state;
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xorshift64_2& other) const noexcept
    {
        __m128i_u cmp = _mm_cmpeq_epi32(other.m_state, m_state);
        uint16_t mask = _mm_movemask_epi8(cmp);
        return (mask == 0xffffU);
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xorshift64_2& other) const noexcept
    {
        return !(*this == other);
    }

private:
    __m128i_u m_state;
};

#ifdef __AVX2__

//xorshift64 implementation using AVX-2 to generate random __m256i_u.
struct xorshift64_4 {
    explicit xorshift64_4(__m256i_u state) noexcept
        : m_state(state)
    {
    }

    xorshift64_4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) noexcept
    {
        m_state = _mm256_set_epi64x(a, b, c, d);
    }

    explicit xorshift64_4(splitmix::splitmix64 gen) noexcept
    {
        m_state = _mm256_set_epi64x(gen.next(), gen.next(), gen.next(), gen.next());
    }

    constexpr operator __m256i_u() const noexcept { return m_state; }

    //Generates random __m128i_u.
    __m256i_u next() noexcept
    {
        m_state = _mm256_xor_si256(m_state, _mm256_srli_epi64(m_state, 13));
        m_state = _mm256_xor_si256(m_state, _mm256_slli_epi64(m_state, 17));
        m_state = _mm256_xor_si256(m_state, _mm256_srli_epi64(m_state, 5));
        return m_state;
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xorshift64_4& other) const noexcept
    {
        __m256i_u cmp0 = _mm256_cmpeq_epi32(other.m_state, m_state);
        uint32_t mask0 = _mm256_movemask_epi8(cmp0);
        return mask0 == 0xffffffffU;
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xorshift64_4& other) const noexcept
    {
        return !(*this == other);
    }

private:
    __m256i_u m_state;
};

#ifdef __AVX512F__

//xorshift64 implementation using AVX512F to generate random __m512i_u.
struct xorshift64_8 {
    explicit xorshift64_8(__m512i_u state) noexcept
        : m_state(state)
    {
    }

    xorshift64_8(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f, uint64_t g, uint64_t h) noexcept
    {
        m_state = _mm512_set_epi64(a, b, c, d, e, f, g, h);
    }

    explicit xorshift64_8(splitmix::splitmix64 gen) noexcept
    {
        m_state = _mm512_set_epi64(gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next(), gen.next());
    }

    constexpr operator __m512i_u() const noexcept { return m_state; }

    //Generates random __m512i_u.
    __m512i_u next() noexcept
    {
        m_state = _mm512_xor_si512(m_state, _mm512_srli_epi64(m_state, 13));
        m_state = _mm512_xor_si512(m_state, _mm512_slli_epi64(m_state, 17));
        m_state = _mm512_xor_si512(m_state, _mm512_srli_epi64(m_state, 5));
        return m_state;
    }

    //Compares internal states of two engines for equality.
    bool operator==(const xorshift64_8& other) const noexcept
    {
        __mmask8 cmp0 = _mm512_cmpeq_epi64_mask(other.m_state, m_state);
        //TODO: proper fast implementation.
        uint16_t mask0 = _mm_movemask_epi8(_mm_movm_epi64(cmp0));
        return (mask0 == 0xffffU);
    }

    //Compares internal states of two engines for inequality.
    bool operator!=(const xorshift64_8& other) const noexcept
    {
        return !(*this == other);
    }

private:
    __m512i_u m_state;
};

#endif // __AVX512F__
#endif // __AVX2__
#endif // __AVX__

//xorshift64 implementation used to generate random uint64_t.
struct xorshift64 {
    explicit xorshift64(uint64_t state) noexcept
        : m_state(state)
    {
    }

    explicit constexpr operator uint64_t() const noexcept { return m_state; }

    //Generates random uint64_t.
    constexpr uint64_t next() noexcept
    {
        m_state ^= m_state << 13;
        m_state ^= m_state >> 17;
        m_state ^= m_state << 5;
        return m_state;
    }

    //Compares internal states of two engines for equality.
    constexpr bool operator==(const xorshift64& other) const noexcept { return other.m_state == m_state; }

    //Compares internal states of two engines for inequality.
    constexpr bool operator!=(const xorshift64& other) const noexcept { return other.m_state != m_state; }

private:
    uint64_t m_state;
};

}

#endif // XORSHIFT64_H_INCLUDED
