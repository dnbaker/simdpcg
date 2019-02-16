#ifndef SIMD_PCG_HPP_
#define SIMD_PCG_HPP_
#include "simdpcg32.h"
#include <cstring>
#include <random>
#include <iterator>
#include <stdexcept>
#include <string>

#ifndef TYPES_TEMPLATES
#define TYPES_TEMPLATES
namespace types {
    template<class T, T v>
    struct integral_constant {
        static constexpr T value = v;
        typedef T value_type;
        typedef integral_constant type; // using injected-class-name
        constexpr operator value_type() const noexcept { return value; }
        constexpr value_type operator()() const noexcept { return value; } //since c++14
    };
    template <bool B>
    using bool_constant = integral_constant<bool, B>;
    using true_type = bool_constant<true>;
    using false_type = bool_constant<false>;

    template<typename T>
    struct is_integral: false_type {};
    template<>struct is_integral<unsigned char>: true_type {};
    template<>struct is_integral<signed char>: true_type {};
    template<>struct is_integral<unsigned short>: true_type {};
    template<>struct is_integral<signed short>: true_type {};
    template<>struct is_integral<unsigned int>: true_type {};
    template<>struct is_integral<signed int>: true_type {};
    template<>struct is_integral<unsigned long>: true_type {};
    template<>struct is_integral<signed long>: true_type {};
    template<>struct is_integral<unsigned long long>: true_type {};
    template<>struct is_integral<signed long long>: true_type {};
    template<class T> inline constexpr bool is_integral_v = is_integral<T>::value;

    template<typename T> struct is_simd: false_type {};
    template<typename T> struct is_simd_int: false_type {};
    template<typename T> struct is_simd_float: false_type {};

#if __SSE2__
    template<>struct is_simd<__m128i>: true_type {};
    template<>struct is_simd<__m128>:  true_type {};
    template<>struct is_simd_int<__m128i>: true_type {};
    template<>struct is_simd_float<__m128>: true_type {};
#endif
#if __AVX2__
    template<>struct is_simd<__m256i>: true_type {};
    template<>struct is_simd<__m256>:  true_type {};
    template<>struct is_simd_int<__m256i>: true_type {};
    template<>struct is_simd_float<__m256>: true_type {};
#endif
#if __AVX512__
    template<>struct is_simd<__m512i>: true_type {};
    template<>struct is_simd<__m512>:  true_type {};
    template<>struct is_simd_int<__m512i>: true_type {};
    template<>struct is_simd_float<__m512>: true_type {};
#endif
    template<class T> inline constexpr bool is_simd_v = is_simd<T>::value;
    template<class T> inline constexpr bool is_simd_int_v = is_simd_int<T>::value;
    template<class T> inline constexpr bool is_simd_float_v = is_simd_float<T>::value;
} // namespace types
#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif // ifdef __GNUC__
#endif // ifndef TYPES_TEMPLATES

namespace pcg {
using namespace std::literals;

// Constants
static const uint64_t INC64 = 0x5851f42d4c957f2d;
__attribute__((aligned(64))) static const __m512i INC = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
__attribute__((aligned(64))) const __m512i MULTIPLIER = _mm512_set1_epi64(0x5851f42d4c957f2d);

// Typedefs for generators.

using AVX512 = avx512_pcg32_random_t;

// Type-ge

auto generate(AVX512 *c) {return avx512_pcg32_random_r(c);}

namespace detail {

template<size_t index>
struct pcg_unroller_t {
    static void apply(__m256i *g, AVX512 *c) {
        *g++ = generate(c++);
        pcg_unroller_t<index - 1>::apply(g, c);
    }
    static void apply_unaligned(__m256i *g, AVX512 *c) {
        _mm256_storeu_si256(g++, generate(c++));
        pcg_unroller_t<index - 1>::apply_unaligned(g, c);
    }
};

template<>
struct pcg_unroller_t<0> {
    template<typename... Args>
    static void apply(Args &&... args) {}
    template<typename... Args>
    static void apply_unaligned(Args &&... args) {}
};

namespace {
__m512i make_seeds(uint64_t seed) {
    std::mt19937_64 mt(seed);
    __m512i ret;
    for(size_t i = 0; i < sizeof(ret) / sizeof(uint64_t); ++i) {
        uint64_t v = mt();
        std::memcpy((uint64_t *)&ret + i, &v, sizeof(v));
    }
    return ret;
}
}

} // namespace detail

template<typename GeneratedType, size_t UNROLL_COUNT=4,
         typename=std::enable_if_t<types::is_simd_int_v<GeneratedType> || types::is_integral_v<GeneratedType>>>
class PCGenerator {
    using SIMDType = __m512i;
    using SIMDGeneratedType = __m256i;

    avx512_pcg32_random_t state_;
    __m256i  gen_[UNROLL_COUNT]; // Usable data
    uint32_t offset_;

    using VectorType        = SIMDType;
public:
    static constexpr size_t NBYTES_TOTAL = sizeof(gen_);
    static constexpr size_t LAST_OFFSET = sizeof(gen_) - sizeof(GeneratedType);
    void generate_values() {
        for(size_t i = 0; i < UNROLL_COUNT; ++i)
            gen_[i] = avx512_pcg32_random_r(&state_);
        offset_ = 0;
    }
    template<typename T, typename=std::enable_if_t<types::is_integral_v<T> || types::is_simd_int_v<T>>>
    void buffill(T *dest, size_t nbytes=sizeof(__m256i)) {
        auto ptr = reinterpret_cast<__m256i *>(dest);
        while(nbytes >= sizeof(__m256i)) {
            _mm256_storeu_si256(ptr++, avx512_pcg32_random_r(&state_));
            nbytes -= sizeof(__m256i);
        }
        if(nbytes) {
            __m256i lv = avx512_pcg32_random_r(&state_);
            std::memcpy((uint8_t *)dest, (uint8_t *)&lv, nbytes);
        }
    }
    void seed(uint64_t seed) {
        state_.state = detail::make_seeds(seed);
    }
    PCGenerator(__m512i seed): state_{seed, INC, MULTIPLIER} {
        generate_values();
    }
    PCGenerator(uint64_t seed): PCGenerator(detail::make_seeds(seed)) {}
    GeneratedType operator()() {
        GeneratedType ret;
        if(__builtin_expect(offset_ > LAST_OFFSET, 0)) generate_values();
        std::memcpy(&ret, reinterpret_cast<const uint8_t *>(gen_) + offset_, sizeof(ret));
        offset_ += sizeof(ret);
        return ret;
    }
    const uint8_t *buf() const {return reinterpret_cast<const uint8_t *>(gen_);}
    size_t bufsize() const {return sizeof(gen_);}
    using ThisType = PCGenerator<GeneratedType, UNROLL_COUNT>;

    template<typename T, bool manual_override=false,
             typename=std::enable_if_t<
                manual_override || types::is_integral_v<T> || types::is_simd_int_v<T>
                >
             >
    class buffer_view {
        ThisType &ref;
    public:
        buffer_view(ThisType &ctr): ref{ctr} {}
        using const_pointer = const T *;
        using pointer       = T *;
        const_pointer cbegin() const {
            return reinterpret_cast<const_pointer>(reinterpret_cast<const uint8_t *>(gen_));
        }
        const_pointer cend() const {
            return reinterpret_cast<const_pointer>(reinterpret_cast<const uint8_t *>(gen_ + sizeof(gen_)));
        }
        pointer begin() {
            return reinterpret_cast<pointer>(reinterpret_cast<const uint8_t *>(gen_));
        }
        pointer end() {
            return reinterpret_cast<pointer>(reinterpret_cast<const uint8_t *>(gen_ + sizeof(gen_)));
        }
    };
    template<typename T, bool manual_override=false>
    buffer_view<T, manual_override> view() {return buffer_view<T, manual_override>(*this);}
}; // class PCGenerator

} // namespace pcg

#endif // #ifndef SIMD_PCG_HPP_
