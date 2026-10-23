# Portable Qwen tokenizer and prompt renderer.  This target deliberately has
# no HIP dependency and is usable by cross-architecture data tools.
add_library(gufo_qwen_tokenization STATIC
  ${PROJECT_SOURCE_DIR}/src/models/qwen/tokenizer.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/chat_template.cpp)
target_include_directories(gufo_qwen_tokenization PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen_tokenization PUBLIC
  gufo_model_foundation ICU::uc ICU::i18n)
