/**
 * @file define.h
 * @brief Core definitions for the DMC Bare-Metal Inference Engine.
 *
 * Implements low-level hardware-aware operations:
 * 1. Platform-agnostic memory access (Flash vs RAM).
 * 2. Integer-based Activation Functions (ReLU/ReLU6).
 * 3. Bit-Unpacking Macros for 8-bit, 4-bit, and 2-bit quantization.
 */

#ifndef DMC_DEFINE_H
#define DMC_DEFINE_H

#include <stdint.h>
#include <math.h> 

// ============================================================================
// 1. Platform Abstraction Layer
// ============================================================================
/*
 * Abstracts memory access to support Harvard Architectures (AVR/Arduino)
 * where constant data (weights) is stored in Flash (PROGMEM).
 */
#if defined(__AVR__) || defined(ARDUINO_ARCH_AVR)
    #include <avr/pgmspace.h>
    #define dmc_pgm_read_byte(addr)  pgm_read_byte(addr)
    #define dmc_pgm_read_word(addr)  pgm_read_word(addr)
    #define dmc_pgm_read_dword(addr) pgm_read_dword(addr)
    #define dmc_pgm_read_float(addr) pgm_read_float(addr)
#else
    // Standard Von Neumann (ARM, x86, RISC-V) - Flash is mapped to memory
    #define dmc_pgm_read_byte(addr)  (*(const uint8_t*)(addr))
    #define dmc_pgm_read_word(addr)  (*(const uint16_t*)(addr))
    #define dmc_pgm_read_dword(addr) (*(const uint32_t*)(addr))
    #define dmc_pgm_read_float(addr) (*(const float*)(addr))
#endif

// ============================================================================
// 2. Configuration Constants
// ============================================================================
// Quantization Schemes (Must match Python utils.py)
#define NONE 0
#define DYNAMIC 1
#define STATIC 2

// Granularity
#define PER_TENSOR 1
#define PER_CHANNEL 2

// ============================================================================
// 3. Data Access Macros (Float & Int32)
// ============================================================================
#define act_write_float(array, index, value) ((array)[index] = (value))
#define act_read_float(array, index)         ((array)[index])

#define par_read_float(array, index)         (dmc_pgm_read_float((array) + (index)))
#define par_read_int32(array, index)         ((int32_t)dmc_pgm_read_dword((array) + (index)))

// ============================================================================
// 4. Activation Functions (Integer Optimized)
// ============================================================================
/* * Branchless implementations where possible to optimize pipeline usage.
 */

// Floating Point
inline float relu(float val) { return (val < 0) ? 0 : val; }
inline float relux(float val, float x) { return (val < 0) ? 0 : (val > x) ? x : val; }
inline float relu6(float val) { return relux(val, 6.0f); }

// Integer (32-bit Accumulators)
inline int32_t relu(int32_t val) { return (val < 0) ? 0 : val; }
inline int32_t relux(int32_t val, int32_t x) { return (val < 0) ? 0 : (val > x) ? x : val; }
inline int32_t relu6(int32_t val) { return relux(val, 6); }

// Quantized Clamping Helpers (Int8)
inline int8_t relu_zero_point(int8_t val, int8_t zp) { 
    return (val < zp) ? zp : val; 
}
inline int8_t relu6_zero_point(int8_t val, int8_t zp, int8_t sp) { 
    return (val < zp) ? zp : (val > sp) ? sp : val; 
}

// ============================================================================
// 5. Bit-Unpacking Logic
// ============================================================================
/*
 * Implements "Optimized Bit-Unpacking" (Algorithm 2 & 3).
 * Uses bitwise shifts and masks to extract sub-byte weights on the fly.
 *
 * Terms:
 * - par_read: Read from Parameter array (Flash/PROGMEM)
 * - act_read: Read from Activation array (RAM)
 * - act_write: Write to Activation array (RAM)
 */

#if !defined(QUANTIZATION_BITWIDTH) || QUANTIZATION_BITWIDTH == 8
    // ------------------------------------------------------------------------
    // 8-BIT (Standard)
    // ------------------------------------------------------------------------
    // To clamp to int8 range after operations
    #define clampb(x) (((x) < -128) ? -128 : ((x) > 127) ? 127 : (x))

    // Direct Access
    #define par_read_packed_intb(arr, idx) ((int8_t)dmc_pgm_read_byte((arr) + (idx)))
    #define act_read_packed_intb(arr, idx) ((int8_t)(arr)[(idx)])
    #define act_write_packed_intb(arr, idx, val) ((arr)[(idx)] = (val))

#elif QUANTIZATION_BITWIDTH == 4
// ------------------------------------------------------------------------
    // 4-BIT (2 weights per byte)
    // ------------------------------------------------------------------------
    // To clamp to int4 range after operations
    #define clampb(x) (((x) < -8) ? -8 : ((x) > 7) ? 7 : (x))

    // x / 2 is replaced by x >> 1
    #define shifting_divisor 1
    // Mask for 4 bits (0x0F)
    #define MASK 0b00001111   // (1 << bitwidth - 1)
    #define position_in_byte(index) ((uint8_t)((index) & 0b1) << 2)
    // #define byte(index) (index) >> shifting_divisor

    /*
     * Algorithm 2 Step: Extract Raw Bits
     * 1. Access byte: packed_array[index >> 1]
     * 2. Mask relevant bits: & (MASK << shift)
     * 3. Shift to LSB: >> shift
     */

    #define par_read_unsigned_packed_intb(packed_array, index) \
        ((dmc_pgm_read_byte((packed_array) + ((index) >> shifting_divisor)) & (MASK << position_in_byte(index))) >> position_in_byte(index))

    #define act_read_unsigned_packed_intb(packed_array, index) \
        (((packed_array)[(index) >> shifting_divisor]) & (MASK << position_in_byte(index))) >> position_in_byte(index)

    /*
     * Sign Extension
     * To interpret a 4-bit value as signed int8:
     * 1. Shift Left to move sign bit to MSB (<< 4).
     * 2. Arithmetic Shift Right to propagate sign (>> 4).
     */
    #define par_read_signed_packed_intb(packed_array, index) \
            ((int8_t)(par_read_unsigned_packed_intb(packed_array, index) << 4) >> 4)

    #define act_read_signed_packed_intb(packed_array, index) \
            ((int8_t)(act_read_unsigned_packed_intb(packed_array, index) << 4) >> 4)

    // Getter
    #define par_read_packed_intb(packed_array, index) par_read_signed_packed_intb(packed_array, index)
    #define act_read_packed_intb(packed_array, index) act_read_signed_packed_intb(packed_array, index)

    // Setter
    #define act_write_packed_intb(packed_array, index, expression) (packed_array)[(index) >> shifting_divisor] = ((packed_array)[(index) >> shifting_divisor] & ~(MASK << position_in_byte(index))) | ((expression & MASK) << position_in_byte(index)); 

#elif QUANTIZATION_BITWIDTH == 2
    // ------------------------------------------------------------------------
    // 2-BIT (4 weights per byte)
    // ------------------------------------------------------------------------
    // To clamp to int2 range after operations
    #define clampb(x) (((x) < -2) ? -2 : ((x) > 1) ? 1 : (x))

    // x / 4 is replaced by x >> 2
    #define shifting_divisor 2 
    
    // Mask for 2 bits (0x03)
    #define MASK 0b00000011   // (1 << bitwidth - 1)

    // Calculates shift amount: (index % 4) * 2 becomes ((index & 3) << 1)
    #define position_in_byte(index) ((uint8_t)((index) & 0b11) << 1)

    #define par_read_unsigned_packed_intb(packed_array, index) \
        ((dmc_pgm_read_byte((packed_array) + ((index) >> shifting_divisor)) & (MASK << position_in_byte(index))) >> position_in_byte(index))

    #define act_read_unsigned_packed_intb(packed_array, index) \
        (((packed_array)[(index) >> shifting_divisor]) & (MASK << position_in_byte(index))) >> position_in_byte(index)




    /*
     * Appendix Algorithm 4: Sign Extension for 2-bit
     * 1. Shift Left by (8 - 2) = 6
     * 2. Arithmetic Shift Right by 6
     */

    #define par_read_signed_packed_intb(packed_array, index) \
            ((int8_t)(par_read_unsigned_packed_intb(packed_array, index) << 6) >> 6)

    #define act_read_signed_packed_intb(packed_array, index) \
            ((int8_t)(act_read_unsigned_packed_intb(packed_array, index) << 6) >> 6)


    #define act_read_packed_intb(packed_array, index) act_read_signed_packed_intb(packed_array, index)


    #define par_read_packed_intb(packed_array, index) par_read_signed_packed_intb(packed_array, index)

    #define act_write_packed_intb(packed_array, index, expression) (packed_array)[(index) >> shifting_divisor] = ((packed_array)[(index) >> shifting_divisor] & ~(MASK << position_in_byte(index))) | ((expression & MASK) << position_in_byte(index)); 

#endif //QUANTIZATION_BITWIDTH

#endif //DMC_DEINE_H
