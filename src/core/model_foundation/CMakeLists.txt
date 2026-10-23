# Host-only model primitives shared by model packages and portable tooling.
# Keep accelerator runtimes out of this target so metadata/tokenization tools
# never inherit an architecture-specific kernel graph.
add_library(gufo_model_foundation STATIC
  ${PROJECT_SOURCE_DIR}/src/core/crypto/sha256.cpp
  ${PROJECT_SOURCE_DIR}/src/core/gguf_reader.cpp
  ${PROJECT_SOURCE_DIR}/src/core/gguf_identity.cpp
  ${PROJECT_SOURCE_DIR}/src/core/json_constraint.cpp
  ${PROJECT_SOURCE_DIR}/src/core/json_schema_lexeme.cpp
  ${PROJECT_SOURCE_DIR}/src/core/json_schema_regex.cpp
  ${PROJECT_SOURCE_DIR}/src/core/sampling.cpp
  ${PROJECT_SOURCE_DIR}/src/core/quant/ggml_dequant.cpp)
target_include_directories(gufo_model_foundation PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_model_foundation PUBLIC
  Threads::Threads OpenSSL::Crypto)

add_library(gufo_image_foundation STATIC
  ${PROJECT_SOURCE_DIR}/src/core/image.cpp)
target_include_directories(gufo_image_foundation PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_image_foundation PUBLIC
  PNG::PNG JPEG::JPEG ${GUFO_WEBP_TARGET} CURL::libcurl)
