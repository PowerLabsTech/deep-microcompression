cmake_minimum_required(VERSION 3.28)

project(deep-microcompression)


add_library(deep_microcompression)

target_sources(deep_microcompression 
            PRIVATE
                models/sequential.cpp
                layers/activation.cpp
                layers/conv.cpp
                layers/batchnorm.cpp
                layers/flatten.cpp
                layers/fused_layers.cpp
                layers/layer.cpp
                layers/linear.cpp
                layers/pad.cpp
                layers/pooling.cpp
                
            PUBLIC
                FILE_SET HEADERS
                FILES
                    deep_microcompression.h
)


# -------------------------------------------------------------
# COMPRESSION_FORMAT option
# -------------------------------------------------------------
# Format: <Scheme><Granularity><Bitwidth>
#   Scheme:
#       N  = NONE
#       ST = STATIC
#       DT = DYNAMIC
#
#   Granularity:
#       T = PER_TENSOR
#       C = PER_CHANNEL
#
#   Bitwidth:
#       2,4,8
#
# Examples:
#   NONE   → No quantization
#   ST8    → Static, per-tensor, 8-bit
#   ST4    → Static, per-tensor, 4-bit
#   ST2    → Static, per-tensor, 2-bit
#   DT8    → Dynamic, per-tensor, 8-bit
#   SC4    → Static, per-channel, 4-bit
#
option(COMPRESSION_FORMAT "Compression format (NONE, ST8, ST4, ST2, DT8, SC4)" "NONE")

message(STATUS "Selected COMPRESSION_FORMAT=${COMPRESSION_FORMAT}")

# -------------------------------------------------------------
# Map COMPRESSION_FORMAT → compile definitions
# -------------------------------------------------------------
if(COMPRESSION_FORMAT STREQUAL "NONE" OR COMPRESSION_FORMAT STREQUAL "N")
    set(QUANT_SCHEME       "NONE")
    set(QUANT_GRANULARITY  "NONE")
    set(QUANT_BITWIDTH     "0")

elseif(COMPRESSION_FORMAT STREQUAL "ST8")
    set(QUANT_SCHEME       "STATIC")
    set(QUANT_GRANULARITY  "PER_TENSOR")
    set(QUANT_BITWIDTH     "8")

elseif(COMPRESSION_FORMAT STREQUAL "ST4")
    set(QUANT_SCHEME       "STATIC")
    set(QUANT_GRANULARITY  "PER_TENSOR")
    set(QUANT_BITWIDTH     "4")

elseif(COMPRESSION_FORMAT STREQUAL "ST2")
    set(QUANT_SCHEME       "STATIC")
    set(QUANT_GRANULARITY  "PER_TENSOR")
    set(QUANT_BITWIDTH     "2")

elseif(COMPRESSION_FORMAT STREQUAL "DT8")
    set(QUANT_SCHEME       "DYNAMIC")
    set(QUANT_GRANULARITY  "PER_TENSOR")
    set(QUANT_BITWIDTH     "8")

elseif(COMPRESSION_FORMAT STREQUAL "SC4")
    set(QUANT_SCHEME       "STATIC")
    set(QUANT_GRANULARITY  "PER_CHANNEL")
    set(QUANT_BITWIDTH     "4")

else()
    message(FATAL_ERROR "Unknown COMPRESSION_FORMAT: '${COMPRESSION_FORMAT}'")
endif()


target_compile_definitions(deep_microcompression 
    PUBLIC
        QUANTIZATION_SCHEME=${QUANT_SCHEME}
        QUANTIZATION_GRANULARITY=${QUANT_GRANULARITY}
        QUANTIZATION_BITWIDTH=${QUANT_BITWIDTH}
)
message(STATUS "Quantization configuration:")
message(STATUS "  SCHEME      = ${QUANT_SCHEME}")
message(STATUS "  GRANULARITY = ${QUANT_GRANULARITY}")
message(STATUS "  BITWIDTH    = ${QUANT_BITWIDTH}")
