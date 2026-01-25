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

#include "quantization_property.h"
#include <stdint.h>
#include <math.h> 

// ============================================================================
// Platform Abstraction Layer
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
// Quantization Configuration Constants
// ============================================================================


// ============================================================================
// Data Access Macros (Float & Int32)
// ============================================================================
#define activation_write_float(array, index, value) ((array)[index] = (value))
#define activation_read_float(array, index)         ((array)[index])

#define parameter_read_float(array, index)         (dmc_pgm_read_float((array) + (index)))
#define parameter_read_int32(array, index)         ((int32_t)dmc_pgm_read_dword((array) + (index)))


// ============================================================================
// Activation Functions (Integer Optimized)
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
// Bit-Unpacking Logic
// ============================================================================
/*
 * Implements "Optimized Bit-Unpacking" (Algorithm 2 & 3).
 * Uses bitwise shifts and masks to extract sub-byte weights on the fly.
 *
 * Terms:
 * - parameter_read: Read from Parameter array (Flash/PROGMEM)
 * - activation_read: Read from Activation array (RAM)
 * - activation_write: Write to Activation array (RAM)
 */

    // ------------------------------------------------------------------------
    // 8-BIT (Standard)
    // ------------------------------------------------------------------------
    // To clamp to int8 range after operations
    #define DATA_PER_BYTE_int8 1
    // #define clamp_int8(x) (((x) < -128) ? -128 : ((x) > 127) ? 127 : (x))

    inline int8_t clamp_int8(int32_t x) {
        return (((x) < -128) ? -128 : ((x) > 127) ? 127 : (x));
    }
    // Direct Access
    inline int8_t parameter_read_packed_int8(const int8_t* array, uint32_t index) {
        return ((int8_t)dmc_pgm_read_byte(array + index));
    }

    inline int8_t activation_read_packed_int8(int8_t* array, uint32_t index) {
        return ((int8_t)array[index]);
    }

    inline void activation_write_packed_int8(int8_t* array, uint32_t index, int8_t expression) {
        (array[index] = expression);
    }


// ------------------------------------------------------------------------
    // 4-BIT (2 weights per byte)
    // ------------------------------------------------------------------------
    // To clamp to int4 range after operations
    #define DATA_PER_BYTE_int4 2
    // #define clamp_int4(x) (((x) < -8) ? -8 : ((x) > 7) ? 7 : (x))

    inline int8_t clamp_int4(int32_t x) {
        return (((x) < -8) ? -8 : ((x) > 7) ? 7 : (x));
    }

    // x / 2 is replaced by x >> 1
    #define shifting_divisor_int4 1
    // Mask for 4 bits (0x0F)
    #define MASK_int4 0b00001111   // (1 << bitwidth - 1)
    #define position_in_byte_int4(index) ((uint8_t)((index) & 0b1) << 2)
    // #define byte(index) (index) >> shifting_divisor

    /*
     * Algorithm 2 Step: Extract Raw Bits
     * 1. Access byte: packed_array[index >> 1]
     * 2. Mask relevant bits: & (MASK << shift)
     * 3. Shift to LSB: >> shift
     */

    #define parameter_read_unsigned_packed_int4(packed_array, index) \
        ((dmc_pgm_read_byte((packed_array) + ((index) >> shifting_divisor_int4)) & (MASK_int4 << position_in_byte_int4(index))) >> position_in_byte_int4(index))

    #define activation_read_unsigned_packed_int4(packed_array, index) \
        (((packed_array)[(index) >> shifting_divisor_int4]) & (MASK_int4 << position_in_byte_int4(index))) >> position_in_byte_int4(index)

    /*
     * Sign Extension
     * To interpret a 4-bit value as signed int8:
     * 1. Shift Left to move sign bit to MSB (<< 4).
     * 2. Arithmetic Shift Right to propagate sign (>> 4).
     */
    #define parameter_read_signed_packed_int4(packed_array, index) \
            ((int8_t)(parameter_read_unsigned_packed_int4(packed_array, index) << 4) >> 4)

    #define activation_read_signed_packed_int4(packed_array, index) \
            ((int8_t)(activation_read_unsigned_packed_int4(packed_array, index) << 4) >> 4)

    inline int8_t parameter_read_packed_int4(const int8_t* packed_array, uint32_t index) {
        return parameter_read_signed_packed_int4(packed_array, index);
    }

    inline int8_t activation_read_packed_int4(int8_t* packed_array, uint32_t index) {
        return activation_read_signed_packed_int4(packed_array, index);
    }

    inline void activation_write_packed_int4(int8_t* packed_array, uint32_t index, int8_t expression) {
        (packed_array)[(index) >> shifting_divisor_int4] = ((packed_array)[(index) >> shifting_divisor_int4] & ~(MASK_int4 << position_in_byte_int4(index))) | ((expression & MASK_int4) << position_in_byte_int4(index)); 
    }



    // ------------------------------------------------------------------------
    // 2-BIT (4 weights per byte)
    // ------------------------------------------------------------------------
    // To clamp to int2 range after operations
    #define DATA_PER_BYTE_int2 4
    // #define clamp_int2(x) (((x) < -2) ? -2 : ((x) > 1) ? 1 : (x))

    inline int8_t clamp_int2(int32_t x) {
        return (((x) < -2) ? -2 : ((x) > 1) ? 1 : (x));
    }
    // x / 4 is replaced by x >> 2
    #define shifting_divisor_int2 2 
    
    // Mask for 2 bits (0x03)
    #define MASK_int2 0b00000011   // (1 << bitwidth - 1)

    // Calculates shift amount: (index % 4) * 2 becomes ((index & 3) << 1)
    #define position_in_byte_int2(index) ((uint8_t)((index) & 0b11) << 1)

    #define parameter_read_unsigned_packed_int2(packed_array, index) \
        ((dmc_pgm_read_byte((packed_array) + ((index) >> shifting_divisor_int2)) & (MASK_int2 << position_in_byte_int2(index))) >> position_in_byte_int2(index))

    #define activation_read_unsigned_packed_int2(packed_array, index) \
        (((packed_array)[(index) >> shifting_divisor_int2]) & (MASK_int2 << position_in_byte_int2(index))) >> position_in_byte_int2(index)

    /*
     * Appendix Algorithm 4: Sign Extension for 2-bit
     * 1. Shift Left by (8 - 2) = 6
     * 2. Arithmetic Shift Right by 6
     */

    #define parameter_read_signed_packed_int2(packed_array, index) \
            ((int8_t)(parameter_read_unsigned_packed_int2(packed_array, index) << 6) >> 6)

    #define activation_read_signed_packed_int2(packed_array, index) \
            ((int8_t)(activation_read_unsigned_packed_int2(packed_array, index) << 6) >> 6)

    inline int8_t parameter_read_packed_int2(const int8_t* packed_array, uint32_t index) {
        return parameter_read_signed_packed_int2(packed_array, index);
    }

    inline int8_t activation_read_packed_int2(int8_t* packed_array, uint32_t index) {
        return activation_read_signed_packed_int2(packed_array, index);
    }

    inline void activation_write_packed_int2(int8_t* packed_array, uint32_t index, int8_t expression) {
        (packed_array)[(index) >> shifting_divisor_int2] = ((packed_array)[(index) >> shifting_divisor_int2] & ~(MASK_int2 << position_in_byte_int2(index))) | ((expression & MASK_int2) << position_in_byte_int2(index)); 
    }


// ============================================================================
// Quantization Abstracition funtions
// ============================================================================
inline uint8_t get_activation_bitwidth(uint8_t quantize_property) {
    return GET_ACTIVATION_BITWIDTH(quantize_property);
}

inline uint8_t get_parameter_bitwidth(uint8_t quantize_property) {
    return GET_PARAMETER_BITWIDTH(quantize_property);
}

inline uint8_t get_granularity(uint8_t quantize_property) {
    return GET_GRANULARITY(quantize_property);
}

inline void get_activation_write_packed_intb(uint8_t quantize_property,  void (**activation_write_packed_intb_addr) (int8_t*, uint32_t, int8_t)) {
    switch (get_activation_bitwidth(quantize_property)) {
        case INT8:
            *activation_write_packed_intb_addr = activation_write_packed_int8;
            break;
        case INT4:
            *activation_write_packed_intb_addr = activation_write_packed_int4;
            break;
        case INT2:
            *activation_write_packed_intb_addr = activation_write_packed_int2;
            break;
    }
}


inline void get_activation_read_packed_intb(uint8_t quantize_property, int8_t (**activation_read_packed_intb_addr) (int8_t*, uint32_t)) {
    switch (get_activation_bitwidth(quantize_property)) {
        case INT8:
            *activation_read_packed_intb_addr = activation_read_packed_int8;
            break;
        case INT4:
            *activation_read_packed_intb_addr = activation_read_packed_int4;
            break;
        case INT2:
            *activation_read_packed_intb_addr = activation_read_packed_int2;
            break;
    }
}


inline void get_parameter_read_packed_intb(uint8_t quantize_property, int8_t (**parameter_read_packed_intb_addr) (const int8_t*, uint32_t)) {
    switch (get_parameter_bitwidth(quantize_property)) {
        case INT8:
            *parameter_read_packed_intb_addr = parameter_read_packed_int8;
            break;
        case INT4:
            *parameter_read_packed_intb_addr = parameter_read_packed_int4;
            break;
        case INT2:
            *parameter_read_packed_intb_addr = parameter_read_packed_int2;
            break;
    }
}


inline void get_activation_clamp_intb(uint8_t quantize_property, int8_t (**clamp_intb_addr) (int32_t)) {
    switch (get_activation_bitwidth(quantize_property)) {
        case INT8:
            *clamp_intb_addr = clamp_int8;
            break;
        case INT4:
            *clamp_intb_addr = clamp_int4;
            break;
        case INT2:
            *clamp_intb_addr = clamp_int2;
            break;
    }
}


inline uint8_t get_activation_data_per_byte(uint8_t quantize_property) {
    switch (get_activation_bitwidth(quantize_property)) {
        case INT8:
            return DATA_PER_BYTE_int8;
        case INT4:
            return DATA_PER_BYTE_int4;
        case INT2:
            return DATA_PER_BYTE_int2;
    }
    return 0;
} 

#endif //DMC_DEINE_H
