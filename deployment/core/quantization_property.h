#ifndef QUANTIZATION_PROPERTIES_H
#define QUANTIZATION_PROPERTIES_H

// ============================================================================
// 1. Bitfield Layout (Quantization Property Encoding)
// ----------------------------------------------------------------------------
// Layout (LSB → MSB):
// [ parameter_bitwidth | activation_bitwidth | granularity | input_activation_bitwidth ]
//
// Bits:
//   parameter_bitwidth         : bits [1:0]
//   activation_bitwidth        : bits [3:2]
//   granularity                : bit  [4]
//   (reserved)                 : bit  [5]
//   input_activation_bitwidth  : bits [7:6]
//
// Example:
//   PER_CHANNEL + IA=8 + ACT=8 + PARAM=4
//   (3 << 6) | (1 << 4) | (3 << 2) | (2 << 0)
// ============================================================================


// ============================================================================
// 2. Bit Shifts
// ============================================================================
#define PARAMETER_BITWIDTH_SHIFT        0
#define ACTIVATION_BITWIDTH_SHIFT       2
#define GRANULARITY_SHIFT               4
#define INPUT_ACTIVATION_BITWIDTH_SHIFT 6


// ============================================================================
// 3. Granularity Encoding
// ============================================================================
#define PER_TENSOR   0b0
#define PER_CHANNEL  0b1


// ============================================================================
// 4. Bitwidth Encoding
// ----------------------------------------------------------------------------
// Aligned with packed-int encoding
//   BITWIDTH_8 = 0b11
//   BITWIDTH_4 = 0b10
//   BITWIDTH_2 = 0b01
//   INT1 = 0b00  (used as "absent")
// ============================================================================

#define BITWIDTH_8 0b11
#define BITWIDTH_4 0b10
#define BITWIDTH_2 0b01
#define BITWIDTH_1 0b00


// ============================================================================
// 5. Canonical Quantization Property Constructors
// ----------------------------------------------------------------------------
// QPROP    — no input-activation field (passthrough layers, parameter-only).
// QPROP_IA — full encoding including input_activation_bitwidth (Conv2d, Linear).
// ============================================================================
#define QPROP(granularity, activation_bw, parameter_bw) \
    (((granularity) << GRANULARITY_SHIFT) |             \
     ((activation_bw) << ACTIVATION_BITWIDTH_SHIFT) |   \
     ((parameter_bw) << PARAMETER_BITWIDTH_SHIFT))

#define QPROP_IA(input_activation_bw, granularity, activation_bw, parameter_bw) \
    (((input_activation_bw) << INPUT_ACTIVATION_BITWIDTH_SHIFT) | \
     QPROP(granularity, activation_bw, parameter_bw))


// ============================================================================
// 6. Predefined Quantization Configurations
// ----------------------------------------------------------------------------
// Naming:
//   <GRANULARITY>_<ACTIVATION?>_<PARAMETER?>
// Rules:
//   - Activation comes before parameter
//   - Omit absent fields
// ============================================================================


// ----------------------------------------------------------------------------
// Per-Tensor: activation + parameter
// ----------------------------------------------------------------------------
#define PER_TENSOR_A8_P8  QPROP(PER_TENSOR, BITWIDTH_8, BITWIDTH_8)
#define PER_TENSOR_A8_P4  QPROP(PER_TENSOR, BITWIDTH_8, BITWIDTH_4)
#define PER_TENSOR_A8_P2  QPROP(PER_TENSOR, BITWIDTH_8, BITWIDTH_2)

#define PER_TENSOR_A4_P8  QPROP(PER_TENSOR, BITWIDTH_4, BITWIDTH_8)
#define PER_TENSOR_A4_P4  QPROP(PER_TENSOR, BITWIDTH_4, BITWIDTH_4)
#define PER_TENSOR_A4_P2  QPROP(PER_TENSOR, BITWIDTH_4, BITWIDTH_2)

#define PER_TENSOR_A2_P8  QPROP(PER_TENSOR, BITWIDTH_2, BITWIDTH_8)
#define PER_TENSOR_A2_P4  QPROP(PER_TENSOR, BITWIDTH_2, BITWIDTH_4)
#define PER_TENSOR_A2_P2  QPROP(PER_TENSOR, BITWIDTH_2, BITWIDTH_2)


// ----------------------------------------------------------------------------
// Per-Tensor: activation only
// ----------------------------------------------------------------------------
#define PER_TENSOR_A8     QPROP(PER_TENSOR, BITWIDTH_8, BITWIDTH_1)
#define PER_TENSOR_A4     QPROP(PER_TENSOR, BITWIDTH_4, BITWIDTH_1)
#define PER_TENSOR_A2     QPROP(PER_TENSOR, BITWIDTH_2, BITWIDTH_1)


// ----------------------------------------------------------------------------
// Per-Tensor: parameter only
// ----------------------------------------------------------------------------
#define PER_TENSOR_P8     QPROP(PER_TENSOR, BITWIDTH_1, BITWIDTH_8)
#define PER_TENSOR_P4     QPROP(PER_TENSOR, BITWIDTH_1, BITWIDTH_4)
#define PER_TENSOR_P2     QPROP(PER_TENSOR, BITWIDTH_1, BITWIDTH_2)


// ----------------------------------------------------------------------------
// Per-Channel: activation + parameter
// ----------------------------------------------------------------------------
#define PER_CHANNEL_A8_P8 QPROP(PER_CHANNEL, BITWIDTH_8, BITWIDTH_8)
#define PER_CHANNEL_A8_P4 QPROP(PER_CHANNEL, BITWIDTH_8, BITWIDTH_4)
#define PER_CHANNEL_A8_P2 QPROP(PER_CHANNEL, BITWIDTH_8, BITWIDTH_2)

#define PER_CHANNEL_A4_P8 QPROP(PER_CHANNEL, BITWIDTH_4, BITWIDTH_8)
#define PER_CHANNEL_A4_P4 QPROP(PER_CHANNEL, BITWIDTH_4, BITWIDTH_4)
#define PER_CHANNEL_A4_P2 QPROP(PER_CHANNEL, BITWIDTH_4, BITWIDTH_2)

#define PER_CHANNEL_A2_P8 QPROP(PER_CHANNEL, BITWIDTH_2, BITWIDTH_8)
#define PER_CHANNEL_A2_P4 QPROP(PER_CHANNEL, BITWIDTH_2, BITWIDTH_4)
#define PER_CHANNEL_A2_P2 QPROP(PER_CHANNEL, BITWIDTH_2, BITWIDTH_2)


// ----------------------------------------------------------------------------
// Per-Channel: activation only
// ----------------------------------------------------------------------------
#define PER_CHANNEL_A8    QPROP(PER_CHANNEL, BITWIDTH_8, BITWIDTH_1)
#define PER_CHANNEL_A4    QPROP(PER_CHANNEL, BITWIDTH_4, BITWIDTH_1)
#define PER_CHANNEL_A2    QPROP(PER_CHANNEL, BITWIDTH_2, BITWIDTH_1)


// ----------------------------------------------------------------------------
// Per-Channel: parameter only
// ----------------------------------------------------------------------------
#define PER_CHANNEL_P8    QPROP(PER_CHANNEL, BITWIDTH_1, BITWIDTH_8)
#define PER_CHANNEL_P4    QPROP(PER_CHANNEL, BITWIDTH_1, BITWIDTH_4)
#define PER_CHANNEL_P2    QPROP(PER_CHANNEL, BITWIDTH_1, BITWIDTH_2)


// ----------------------------------------------------------------------------
// Global shorthand (per-tensor defaults)
// ----------------------------------------------------------------------------
#define A8 QPROP(PER_TENSOR, BITWIDTH_8, BITWIDTH_1)
#define A4 QPROP(PER_TENSOR, BITWIDTH_4, BITWIDTH_1)
#define A2 QPROP(PER_TENSOR, BITWIDTH_2, BITWIDTH_1)

#define P8 QPROP(PER_TENSOR, BITWIDTH_1, BITWIDTH_8)
#define P4 QPROP(PER_TENSOR, BITWIDTH_1, BITWIDTH_4)
#define P2 QPROP(PER_TENSOR, BITWIDTH_1, BITWIDTH_2)


// ----------------------------------------------------------------------------
// Per-Tensor: IA8 input activation
// ----------------------------------------------------------------------------
#define IA8_PER_TENSOR_A8_P8  QPROP_IA(BITWIDTH_8, PER_TENSOR, BITWIDTH_8, BITWIDTH_8)
#define IA8_PER_TENSOR_A8_P4  QPROP_IA(BITWIDTH_8, PER_TENSOR, BITWIDTH_8, BITWIDTH_4)
#define IA8_PER_TENSOR_A8_P2  QPROP_IA(BITWIDTH_8, PER_TENSOR, BITWIDTH_8, BITWIDTH_2)
#define IA8_PER_TENSOR_A4_P8  QPROP_IA(BITWIDTH_8, PER_TENSOR, BITWIDTH_4, BITWIDTH_8)
#define IA8_PER_TENSOR_A4_P4  QPROP_IA(BITWIDTH_8, PER_TENSOR, BITWIDTH_4, BITWIDTH_4)
#define IA8_PER_TENSOR_A4_P2  QPROP_IA(BITWIDTH_8, PER_TENSOR, BITWIDTH_4, BITWIDTH_2)
#define IA8_PER_TENSOR_A2_P8  QPROP_IA(BITWIDTH_8, PER_TENSOR, BITWIDTH_2, BITWIDTH_8)
#define IA8_PER_TENSOR_A2_P4  QPROP_IA(BITWIDTH_8, PER_TENSOR, BITWIDTH_2, BITWIDTH_4)
#define IA8_PER_TENSOR_A2_P2  QPROP_IA(BITWIDTH_8, PER_TENSOR, BITWIDTH_2, BITWIDTH_2)

// ----------------------------------------------------------------------------
// Per-Channel: IA8 input activation
// ----------------------------------------------------------------------------
#define IA8_PER_CHANNEL_A8_P8 QPROP_IA(BITWIDTH_8, PER_CHANNEL, BITWIDTH_8, BITWIDTH_8)
#define IA8_PER_CHANNEL_A8_P4 QPROP_IA(BITWIDTH_8, PER_CHANNEL, BITWIDTH_8, BITWIDTH_4)
#define IA8_PER_CHANNEL_A8_P2 QPROP_IA(BITWIDTH_8, PER_CHANNEL, BITWIDTH_8, BITWIDTH_2)
#define IA8_PER_CHANNEL_A4_P8 QPROP_IA(BITWIDTH_8, PER_CHANNEL, BITWIDTH_4, BITWIDTH_8)
#define IA8_PER_CHANNEL_A4_P4 QPROP_IA(BITWIDTH_8, PER_CHANNEL, BITWIDTH_4, BITWIDTH_4)
#define IA8_PER_CHANNEL_A4_P2 QPROP_IA(BITWIDTH_8, PER_CHANNEL, BITWIDTH_4, BITWIDTH_2)
#define IA8_PER_CHANNEL_A2_P8 QPROP_IA(BITWIDTH_8, PER_CHANNEL, BITWIDTH_2, BITWIDTH_8)
#define IA8_PER_CHANNEL_A2_P4 QPROP_IA(BITWIDTH_8, PER_CHANNEL, BITWIDTH_2, BITWIDTH_4)
#define IA8_PER_CHANNEL_A2_P2 QPROP_IA(BITWIDTH_8, PER_CHANNEL, BITWIDTH_2, BITWIDTH_2)

// ----------------------------------------------------------------------------
// Per-Tensor: IA4 input activation
// ----------------------------------------------------------------------------
#define IA4_PER_TENSOR_A8_P8  QPROP_IA(BITWIDTH_4, PER_TENSOR, BITWIDTH_8, BITWIDTH_8)
#define IA4_PER_TENSOR_A8_P4  QPROP_IA(BITWIDTH_4, PER_TENSOR, BITWIDTH_8, BITWIDTH_4)
#define IA4_PER_TENSOR_A8_P2  QPROP_IA(BITWIDTH_4, PER_TENSOR, BITWIDTH_8, BITWIDTH_2)
#define IA4_PER_TENSOR_A4_P8  QPROP_IA(BITWIDTH_4, PER_TENSOR, BITWIDTH_4, BITWIDTH_8)
#define IA4_PER_TENSOR_A4_P4  QPROP_IA(BITWIDTH_4, PER_TENSOR, BITWIDTH_4, BITWIDTH_4)
#define IA4_PER_TENSOR_A4_P2  QPROP_IA(BITWIDTH_4, PER_TENSOR, BITWIDTH_4, BITWIDTH_2)
#define IA4_PER_TENSOR_A2_P8  QPROP_IA(BITWIDTH_4, PER_TENSOR, BITWIDTH_2, BITWIDTH_8)
#define IA4_PER_TENSOR_A2_P4  QPROP_IA(BITWIDTH_4, PER_TENSOR, BITWIDTH_2, BITWIDTH_4)
#define IA4_PER_TENSOR_A2_P2  QPROP_IA(BITWIDTH_4, PER_TENSOR, BITWIDTH_2, BITWIDTH_2)

// ----------------------------------------------------------------------------
// Per-Channel: IA4 input activation
// ----------------------------------------------------------------------------
#define IA4_PER_CHANNEL_A8_P8 QPROP_IA(BITWIDTH_4, PER_CHANNEL, BITWIDTH_8, BITWIDTH_8)
#define IA4_PER_CHANNEL_A8_P4 QPROP_IA(BITWIDTH_4, PER_CHANNEL, BITWIDTH_8, BITWIDTH_4)
#define IA4_PER_CHANNEL_A8_P2 QPROP_IA(BITWIDTH_4, PER_CHANNEL, BITWIDTH_8, BITWIDTH_2)
#define IA4_PER_CHANNEL_A4_P8 QPROP_IA(BITWIDTH_4, PER_CHANNEL, BITWIDTH_4, BITWIDTH_8)
#define IA4_PER_CHANNEL_A4_P4 QPROP_IA(BITWIDTH_4, PER_CHANNEL, BITWIDTH_4, BITWIDTH_4)
#define IA4_PER_CHANNEL_A4_P2 QPROP_IA(BITWIDTH_4, PER_CHANNEL, BITWIDTH_4, BITWIDTH_2)
#define IA4_PER_CHANNEL_A2_P8 QPROP_IA(BITWIDTH_4, PER_CHANNEL, BITWIDTH_2, BITWIDTH_8)
#define IA4_PER_CHANNEL_A2_P4 QPROP_IA(BITWIDTH_4, PER_CHANNEL, BITWIDTH_2, BITWIDTH_4)
#define IA4_PER_CHANNEL_A2_P2 QPROP_IA(BITWIDTH_4, PER_CHANNEL, BITWIDTH_2, BITWIDTH_2)

// ----------------------------------------------------------------------------
// Per-Tensor: IA2 input activation
// ----------------------------------------------------------------------------
#define IA2_PER_TENSOR_A8_P8  QPROP_IA(BITWIDTH_2, PER_TENSOR, BITWIDTH_8, BITWIDTH_8)
#define IA2_PER_TENSOR_A8_P4  QPROP_IA(BITWIDTH_2, PER_TENSOR, BITWIDTH_8, BITWIDTH_4)
#define IA2_PER_TENSOR_A8_P2  QPROP_IA(BITWIDTH_2, PER_TENSOR, BITWIDTH_8, BITWIDTH_2)
#define IA2_PER_TENSOR_A4_P8  QPROP_IA(BITWIDTH_2, PER_TENSOR, BITWIDTH_4, BITWIDTH_8)
#define IA2_PER_TENSOR_A4_P4  QPROP_IA(BITWIDTH_2, PER_TENSOR, BITWIDTH_4, BITWIDTH_4)
#define IA2_PER_TENSOR_A4_P2  QPROP_IA(BITWIDTH_2, PER_TENSOR, BITWIDTH_4, BITWIDTH_2)
#define IA2_PER_TENSOR_A2_P8  QPROP_IA(BITWIDTH_2, PER_TENSOR, BITWIDTH_2, BITWIDTH_8)
#define IA2_PER_TENSOR_A2_P4  QPROP_IA(BITWIDTH_2, PER_TENSOR, BITWIDTH_2, BITWIDTH_4)
#define IA2_PER_TENSOR_A2_P2  QPROP_IA(BITWIDTH_2, PER_TENSOR, BITWIDTH_2, BITWIDTH_2)

// ----------------------------------------------------------------------------
// Per-Channel: IA2 input activation
// ----------------------------------------------------------------------------
#define IA2_PER_CHANNEL_A8_P8 QPROP_IA(BITWIDTH_2, PER_CHANNEL, BITWIDTH_8, BITWIDTH_8)
#define IA2_PER_CHANNEL_A8_P4 QPROP_IA(BITWIDTH_2, PER_CHANNEL, BITWIDTH_8, BITWIDTH_4)
#define IA2_PER_CHANNEL_A8_P2 QPROP_IA(BITWIDTH_2, PER_CHANNEL, BITWIDTH_8, BITWIDTH_2)
#define IA2_PER_CHANNEL_A4_P8 QPROP_IA(BITWIDTH_2, PER_CHANNEL, BITWIDTH_4, BITWIDTH_8)
#define IA2_PER_CHANNEL_A4_P4 QPROP_IA(BITWIDTH_2, PER_CHANNEL, BITWIDTH_4, BITWIDTH_4)
#define IA2_PER_CHANNEL_A4_P2 QPROP_IA(BITWIDTH_2, PER_CHANNEL, BITWIDTH_4, BITWIDTH_2)
#define IA2_PER_CHANNEL_A2_P8 QPROP_IA(BITWIDTH_2, PER_CHANNEL, BITWIDTH_2, BITWIDTH_8)
#define IA2_PER_CHANNEL_A2_P4 QPROP_IA(BITWIDTH_2, PER_CHANNEL, BITWIDTH_2, BITWIDTH_4)
#define IA2_PER_CHANNEL_A2_P2 QPROP_IA(BITWIDTH_2, PER_CHANNEL, BITWIDTH_2, BITWIDTH_2)


// ============================================================================
// 7. Bitfield Extraction Helpers
// ============================================================================
#define GET_GRANULARITY(qprop) \
    (((qprop) >> GRANULARITY_SHIFT) & 0x1)

#define GET_PARAMETER_BITWIDTH(qprop) \
    (((qprop) >> PARAMETER_BITWIDTH_SHIFT) & 0x3)

#define GET_ACTIVATION_BITWIDTH(qprop) \
    (((qprop) >> ACTIVATION_BITWIDTH_SHIFT) & 0x3)

#define GET_INPUT_ACTIVATION_BITWIDTH(qprop) \
    (((qprop) >> INPUT_ACTIVATION_BITWIDTH_SHIFT) & 0x3)

// ============================================================================
// 8. Semantic Helpers
// ============================================================================
#define IS_PER_CHANNEL(qprop) (GET_GRANULARITY(qprop) == PER_CHANNEL)
#define IS_PER_TENSOR(qprop)  (GET_GRANULARITY(qprop) == PER_TENSOR)

#define PARAM_IS_INT8(qprop)  (GET_PARAMETER_BITWIDTH(qprop) == BITWIDTH_8)
#define PARAM_IS_INT4(qprop)  (GET_PARAMETER_BITWIDTH(qprop) == BITWIDTH_4)
#define PARAM_IS_INT2(qprop)  (GET_PARAMETER_BITWIDTH(qprop) == BITWIDTH_2)

#define ACT_IS_INT8(qprop)    (GET_ACTIVATION_BITWIDTH(qprop) == BITWIDTH_8)
#define ACT_IS_INT4(qprop)    (GET_ACTIVATION_BITWIDTH(qprop) == BITWIDTH_4)
#define ACT_IS_INT2(qprop)    (GET_ACTIVATION_BITWIDTH(qprop) == BITWIDTH_2)


#endif // QUANTIZATION_PROPERTIES_H
