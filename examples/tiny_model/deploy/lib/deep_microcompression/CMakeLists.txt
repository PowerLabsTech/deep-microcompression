cmake_minimum_required(VERSION 3.28)

project(deep-microcompression)

add_library(deep_microcompression)

target_sources(deep_microcompression 
            PUBLIC
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