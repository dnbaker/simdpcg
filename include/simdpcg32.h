#ifndef SIMDXORSHIFT128PLUS_H
#define SIMDXORSHIFT128PLUS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h> // life is short, please use a C99-compliant compiler

#if defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) &&                               \
    (defined(__VEC__) || defined(__ALTIVEC__))
/* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
/* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define AVX512PCG
typedef struct avx512_pcg_state_setseq_64 { // Internals are *Private*.
  __m512i state;      // (8x64bits) RNG state.  All values are possible.
  __m512i inc;        // (8x64bits)Controls which RNG sequences (stream) is
                      // selected. Must *always* be odd. You probably want
                      // distinct sequences
  __m512i multiplier; // set to _mm512_set1_epi64(0x5851f42d4c957f2d);
} avx512_pcg32_random_t;
static inline __m256i avx512_pcg32_random_r(avx512_pcg32_random_t *rng) {
  const __m512i oldstate = rng->state;
  rng->state = _mm512_add_epi64(_mm512_mullo_epi64(rng->multiplier, rng->state),
                                rng->inc);
  __m512i xorshifted = _mm512_srli_epi64(
      _mm512_xor_epi64(_mm512_srli_epi64(oldstate, 18), oldstate), 27);
  __m512i rot = _mm512_srli_epi64(oldstate, 59);
  return _mm512_cvtepi64_epi32(_mm512_rorv_epi32(xorshifted,rot));
}


/*
#else
#error("simdpcg is only available under AVX512, required AVX512F and AVX512DQ")
*/
#endif
#if __AVX2__
typedef struct avx256_pcg_state_setseq_64 { // Internals are *Private*.
  __m256i state;      // (4x64bits) RNG state.  All values are possible.
  __m256i inc;        // (4x64bits)Controls which RNG sequences (stream) is
                      // selected. Must *always* be odd. You probably want
                      // distinct sequences
  __m256i multiplier; // set to _mm256_set1_epi64(0x5851f42d4c957f2d);
} avx256_pcg32_random_t;
static inline __m128i avx256_pcg32_random_r(avx256_pcg32_random_t *rng) {
  const __m256i oldstate = rng->state;
  rng->state = _mm256_add_epi64(_mm256_mullo_epi64(rng->multiplier, rng->state),
                                rng->inc);
  __m256i xorshifted = _mm256_srli_epi64(
      _mm256_xor_si256(_mm256_srli_epi64(oldstate, 18), oldstate), 27);
  __m256i rot = _mm256_srli_epi64(oldstate, 59);
  return _mm256_cvtepi64_epi32(_mm256_rorv_epi32(xorshifted,rot));
}
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
