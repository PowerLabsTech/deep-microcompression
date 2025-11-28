/*
 * @file define.h
 * @brief Core definitions for the DMC Bare-Metal Inference Engine.
 *
 * This file implements the low-level "Hardware-Aware" operations described in
 *
 * Key components:
 * 1. Integer-based Activation Functions (ReLU/ReLU6).
 * 2. Bit-Unpacking Macros for 4-bit and 2-bit weights.
 * 3. Sign Extension Logic.
 */
#include <stdint.h>


// Configuration Constants (Must match Python utils.py)
#define NONE 0
#define DYNAMIC 1
#define STATIC 2

#define PER_TENSOR 1
#define PER_CHANNEL 2


// Activation Functions (Integer Optimized)
/*
 * Standard ReLU: y = max(0, x)
 * Implemented with ternary operators to compile into conditional moves (CMOV)
 * on supported architectures, avoiding branching overhead.
 */
inline float relu(float val) { return (val < 0) ? 0 : val;}
inline int32_t relu(int32_t val) { return (val < 0) ? 0 : val;}

// Helpers for generic clamping
inline int8_t relu_zero_point(int8_t val, int8_t zero_point) { return (val < zero_point) ? zero_point : val;}
inline int8_t relu6_zero_point(int8_t val, int8_t zero_point, int8_t six_point) { return (val < zero_point) ? zero_point : (val > six_point) ? six_point : val;}

inline float relux(float val, float x) { return (val < 0) ? 0 : (val > x) ? x : val;}
inline int32_t relux(int32_t val, int32_t x) { return (val < 0) ? 0 : (val > x) ? x : val;}

inline float relu6(float val) { return relux(val, 6.);}
inline int32_t relu6(int32_t val) { return relux(val, 6);}


// Bit-Unpacking Logic (Section III-C)
/*
 * These macros implement "Optimized Bit-Unpacking"
 *
 * Instead of computationally expensive division/modulo operations, we use
 * compile-time constants for bitwise shifts and masks.
 *
 * Constraints:
 * - QUANTIZATION_BITWIDTH must be defined during compilation (e.g., -DQUANTIZATION_BITWIDTH=4).
 * - Defaults to 8-bit (Standard) if undefined.
 */
#if !defined(QUANTIZATION_BITWIDTH) || QUANTIZATION_BITWIDTH == 8
    // Standard 8-bit access (No packing)
    #define get_packed_value(packed_array, index) (packed_array[index])
    #define set_packed_value(packed_array, index, expression) (packed_array[index] = expression)
#elif QUANTIZATION_BITWIDTH == 4
    // 4-Bit Unpacking (2 weights per byte)

    // x / 2 is replaced by x >> 1
    #define shifting_divisor 1
    // Mask for 4 bits (0x0F)
    #define MASK 0b00001111   // (1 << bitwidth - 1)
    #define position_in_byte(index) (((index) & 0b1) << 2)
    // #define byte(index) (index) >> shifting_divisor

    /*
     * Algorithm 2 Step: Extract Raw Bits
     * 1. Access byte: packed_array[index >> 1]
     * 2. Mask relevant bits: & (MASK << shift)
     * 3. Shift to LSB: >> shift
     */
    #define get_unsigned_packed_value(packed_array, index) \
        ((packed_array[(index) >> shifting_divisor] & (MASK << position_in_byte(index))) >> position_in_byte(index))

    /*
     * Sign Extension
     * To interpret a 4-bit value as signed int8:
     * 1. Shift Left to move sign bit to MSB (<< 4).
     * 2. Arithmetic Shift Right to propagate sign (>> 4).
     */
    #define get_signed_packed_value(packed_array, index) \
        ((int8_t)(get_unsigned_packed_value(packed_array, index) << 4) >> 4)

    // Default getter relies on signed unpacking for weights
    #define get_packed_value(packed_array, index) get_signed_packed_value(packed_array, index)
    // #define get_packed_value(packed_array, index) ((packed_array[(index) >> shifting_divisor] >> position_in_byte(index)) & MASK)
    
    // Setter
    #define set_packed_value(packed_array, index, expression) packed_array[(index) >> shifting_divisor] = (packed_array[(index) >> shifting_divisor] & ~(MASK << position_in_byte(index))) | ((expression & MASK) << position_in_byte(index)); 
    // #define set_packed_value(packed_array, index, expression) packed_array[(index) >> shifting_divisor] = ((packed_array[(index) >> shifting_divisor] | (expression & MASK) << position_in_byte(index)) & (expression | ~MASK) << position_in_byte(index)))
    // #define set_packed_value(packed_array, index, expression) packed_array[(index) >> shifting_divisor] = ((packed_array[(index) >> shifting_divisor] << position_in_byte(index)) | (expression & MASK))

    #elif QUANTIZATION_BITWIDTH == 2
    // 2-Bit Unpacking (4 weights per byte)

    // x / 4 is replaced by x >> 2
    #define shifting_divisor 2 
    
    // Mask for 2 bits (0x03)
    #define MASK 0b00000011   // (1 << bitwidth - 1)

    // Calculates shift amount: (index % 4) * 2 becomes ((index & 3) << 1)
    #define position_in_byte(index) (((index) & 0b11) << 1)

    #define get_unsigned_packed_value(packed_array, index) ((packed_array[(index) >> shifting_divisor] & (MASK << position_in_byte(index))) >> position_in_byte(index))

    /*
     * Appendix Algorithm 4: Sign Extension for 2-bit
     * 1. Shift Left by (8 - 2) = 6
     * 2. Arithmetic Shift Right by 6
     */
    #define get_signed_packed_value(packed_array, index) ((int8_t)(get_unsigned_packed_value(packed_array, index) << 6) >> 6)
    #define get_packed_value(packed_array, index) get_signed_packed_value(packed_array, index)

    #define set_packed_value(packed_array, index, expression) packed_array[(index) >> shifting_divisor] = (packed_array[(index) >> shifting_divisor] & ~(MASK << position_in_byte(index))) | ((expression & MASK) << position_in_byte(index)); 

#endif
