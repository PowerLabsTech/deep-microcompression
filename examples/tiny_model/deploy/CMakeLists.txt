cmake_minimum_required(VERSION 3.28)

project(TINY_MODEL_EXAMPLE)

add_executable(main)

target_sources(main 
        PRIVATE
            src/main.cpp
            src/tiny_model_def.cpp
            src/tiny_model_params.cpp

        PRIVATE
            FILE_SET header
            TYPE HEADERS
            BASE_DIRS
                include
)   


add_subdirectory(lib/deep_microcompression)

target_link_libraries(main
    PRIVATE 
        deep_microcompression
)