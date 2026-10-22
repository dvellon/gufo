#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <hip/hip_runtime_api.h>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace {

constexpr std::string_view kFormat = "gufo.flashmap.exact-target-generation.v1";
constexpr std::size_t kMaxJsonLineBytes = 16 * 1024 * 1024;
constexpr std::size_t kMaxConfigBytes = 64 * 1024;
constexpr std::uint64_t kMaxResumeBytes = 512ULL * 1024 * 1024;

std::string ReadBounded(const std::filesystem::path& path, std::size_t limit) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open " + path.string());
  const auto end = input.tellg();
  if (end < 0 || static_cast<std::uint64_t>(end) > limit)
    throw std::runtime_error("file exceeds its byte envelope: " +
                             path.string());
  std::string result(static_cast<std::size_t>(end), '\0');
  input.seekg(0);
  if (!result.empty() && !input.read(result.data(), result.size()))
    throw std::runtime_error("cannot read " + path.string());
  return result;
}

bool IsLowerHex(std::string_view value, std::size_t digits) {
  if (value.size() != digits)
    return false;
  return std::ranges::all_of(value, [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

gufo::json::Value CanonicalValue(const gufo::json::Value& value) {
  if (value.is_array()) {
    auto result = gufo::json::Value::array();
    for (const auto& item : value.items())
      result.push_back(CanonicalValue(item));
    return result;
  }
  if (value.is_object()) {
    std::vector<const gufo::json::Value::Member*> members;
    members.reserve(value.members().size());
    for (const auto& member : value.members())
      members.push_back(&member);
    std::ranges::sort(members, [](const auto* left, const auto* right) {
      return left->first < right->first;
    });
    auto result = gufo::json::Value::object();
    for (const auto* member : members)
      result.append_member(member->first, CanonicalValue(member->second));
    return result;
  }
  return value;
}

std::string CanonicalJson(const gufo::json::Value& value) {
  return CanonicalValue(value).dump();
}

void RequireKeys(const gufo::json::Value& value,
                 std::span<const std::string_view> keys,
                 std::string_view context) {
  if (!value.is_object() || value.members().size() != keys.size())
    throw std::runtime_error(std::string(context) + " has the wrong fields");
  std::unordered_set<std::string> actual;
  for (const auto& [key, unused] : value.members()) {
    (void)unused;
    if (!actual.insert(key).second)
      throw std::runtime_error(std::string(context) + " repeats field " + key);
  }
  for (const auto key : keys) {
    if (!actual.contains(std::string(key)))
      throw std::runtime_error(std::string(context) + " is missing field " +
                               std::string(key));
  }
}

void RequirePortableMetadata(const gufo::json::Value& value,
                             std::string_view context) {
  if (value.is_number())
    throw std::runtime_error(std::string(context) +
                             " contains a non-portable number");
  if (value.is_array()) {
    for (const auto& item : value.items())
      RequirePortableMetadata(item, context);
  } else if (value.is_object()) {
    std::unordered_set<std::string> keys;
    for (const auto& [key, item] : value.members()) {
      if (!keys.insert(key).second)
        throw std::runtime_error(std::string(context) + " repeats key " + key);
      RequirePortableMetadata(item, context);
    }
  }
}

std::size_t ExactSize(const gufo::json::Value* value, std::string_view field) {
  if (value == nullptr || !value->is_number())
    throw std::runtime_error(std::string(field) + " must be an integer");
  const double number = value->as_double(-1.0);
  if (number < 1.0 || number > 1'000'000.0 ||
      number != static_cast<double>(static_cast<std::size_t>(number)))
    throw std::runtime_error(std::string(field) + " is outside its envelope");
  return static_cast<std::size_t>(number);
}

struct GenerationConfig {
  std::uint32_t max_context{};
  std::uint32_t max_response_tokens{};
  gufo::tokenization::ChatTemplateOptions chat;
  std::string chat_template_sha256;
  std::string sha256;
};

GenerationConfig ParseGenerationConfig(const std::filesystem::path& path,
                                       std::string_view reviewed_sha256) {
  if (!IsLowerHex(reviewed_sha256, 64))
    throw std::runtime_error(
        "reviewed generation configuration identity is not lowercase SHA-256");
  const std::string bytes = ReadBounded(path, kMaxConfigBytes);
  if (bytes.empty() || bytes.back() != '\n')
    throw std::runtime_error(
        "generation configuration must be newline-terminated canonical JSON");
  const std::string actual = gufo::crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()});
  if (actual != reviewed_sha256)
    throw std::runtime_error("generation configuration identity differs");
  const auto value =
      gufo::json::parse(std::string_view(bytes).substr(0, bytes.size() - 1));
  constexpr std::array<std::string_view, 10> keys{"add_generation_prompt",
                                                  "chat_template_sha256",
                                                  "enable_thinking",
                                                  "format",
                                                  "max_context",
                                                  "max_response_tokens",
                                                  "preserve_thinking",
                                                  "reasoning_effort",
                                                  "sampling",
                                                  "stop_at_eos"};
  RequireKeys(value, keys, "generation configuration");
  if (CanonicalJson(value) + "\n" != bytes)
    throw std::runtime_error("generation configuration is not canonical JSON");
  const auto* format = value.find("format");
  const auto* template_sha = value.find("chat_template_sha256");
  const auto* add_prompt = value.find("add_generation_prompt");
  const auto* thinking = value.find("enable_thinking");
  const auto* preserve = value.find("preserve_thinking");
  const auto* effort = value.find("reasoning_effort");
  const auto* sampling = value.find("sampling");
  const auto* stop = value.find("stop_at_eos");
  if (format == nullptr || !format->is_string() || format->str() != kFormat ||
      template_sha == nullptr || !template_sha->is_string() ||
      !IsLowerHex(template_sha->str(), 64) || add_prompt == nullptr ||
      !add_prompt->is_bool() || !add_prompt->as_bool() || thinking == nullptr ||
      !thinking->is_bool() || preserve == nullptr || !preserve->is_bool() ||
      !preserve->as_bool() || effort == nullptr || !effort->is_string() ||
      sampling == nullptr || !sampling->is_string() ||
      sampling->str() != "greedy" || stop == nullptr || !stop->is_bool() ||
      !stop->as_bool()) {
    throw std::runtime_error(
        "generation configuration does not describe exact greedy generation");
  }
  GenerationConfig result;
  result.max_context = static_cast<std::uint32_t>(
      ExactSize(value.find("max_context"), "max_context"));
  result.max_response_tokens = static_cast<std::uint32_t>(
      ExactSize(value.find("max_response_tokens"), "max_response_tokens"));
  if (result.max_response_tokens >= result.max_context)
    throw std::runtime_error("response budget must be smaller than context");
  result.chat.add_generation_prompt = true;
  result.chat.enable_thinking = thinking->as_bool();
  result.chat.preserve_thinking = true;
  if (effort->str() == "low")
    result.chat.reasoning_effort =
        gufo::tokenization::QwenReasoningEffort::kLow;
  else if (effort->str() == "medium")
    result.chat.reasoning_effort =
        gufo::tokenization::QwenReasoningEffort::kMedium;
  else if (effort->str() == "xhigh")
    result.chat.reasoning_effort =
        gufo::tokenization::QwenReasoningEffort::kXHigh;
  else
    throw std::runtime_error("reasoning_effort is not low, medium or xhigh");
  result.chat.max_output_bytes =
      gufo::tokenization::RenderedPromptBoundBytes(result.max_context);
  result.chat_template_sha256 = template_sha->str();
  result.sha256 = actual;
  return result;
}

struct PromptRecord {
  std::string id;
  std::string source_id;
  std::vector<gufo::tokenization::ChatMessage> messages;
  gufo::json::Value metadata;
  std::string canonical_sha256;
};

gufo::tokenization::ChatRole ParseRole(std::string_view role) {
  if (role == "system")
    return gufo::tokenization::ChatRole::kSystem;
  if (role == "developer")
    return gufo::tokenization::ChatRole::kDeveloper;
  if (role == "user")
    return gufo::tokenization::ChatRole::kUser;
  throw std::runtime_error(
      "prompt messages support only system, developer and user roles");
}

PromptRecord ParsePromptRecord(const gufo::json::Value& value) {
  constexpr std::array<std::string_view, 4> keys{"id", "messages", "metadata",
                                                 "source_id"};
  RequireKeys(value, keys, "prompt source row");
  const auto* id = value.find("id");
  const auto* source_id = value.find("source_id");
  const auto* messages = value.find("messages");
  const auto* metadata = value.find("metadata");
  if (id == nullptr || !id->is_string() || id->str().empty() ||
      source_id == nullptr || !source_id->is_string() ||
      source_id->str().empty())
    throw std::runtime_error("prompt source IDs must be non-empty strings");
  if (messages == nullptr || !messages->is_array() || messages->items().empty())
    throw std::runtime_error(
        "prompt source messages must be a non-empty array");
  if (metadata == nullptr || !metadata->is_object())
    throw std::runtime_error("prompt source metadata must be an object");
  RequirePortableMetadata(*metadata, "prompt source metadata");
  PromptRecord result{.id = id->str(),
                      .source_id = source_id->str(),
                      .metadata = CanonicalValue(*metadata)};
  constexpr std::array<std::string_view, 2> message_keys{"content", "role"};
  for (const auto& item : messages->items()) {
    RequireKeys(item, message_keys, "prompt message");
    const auto* role = item.find("role");
    const auto* content = item.find("content");
    if (role == nullptr || !role->is_string() || content == nullptr ||
        !content->is_string() || content->str().empty())
      throw std::runtime_error("prompt message role/content is invalid");
    result.messages.emplace_back(ParseRole(role->str()), content->str());
  }
  if (result.messages.back().role != gufo::tokenization::ChatRole::kUser)
    throw std::runtime_error("the final prompt message must have user role");
  const std::string canonical = CanonicalJson(value);
  result.canonical_sha256 = gufo::crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(canonical.data()),
       canonical.size()});
  return result;
}

std::vector<std::int32_t> RenderPrompt(
    const gufo::tokenization::QwenTokenizer& tokenizer,
    std::size_t vocabulary_size,
                                       const PromptRecord& record,
                                       const GenerationConfig& config) {
  std::string error;
  const auto tokens = gufo::tokenization::QwenChatTemplate::RenderAndTokenize(
      tokenizer, record.messages, config.chat, &error);
  if (!tokens || tokens->empty())
    throw std::runtime_error("chat prompt rendering failed: " + error);
  if (tokens->size() + config.max_response_tokens > config.max_context)
    throw std::runtime_error("prompt and response budget exceed max_context");
  std::vector<std::int32_t> result(tokens->begin(), tokens->end());
  for (const auto token : result) {
    if (token < 0 || static_cast<std::size_t>(token) >= vocabulary_size)
      throw std::runtime_error("chat template emitted an invalid token");
  }
  return result;
}

std::vector<std::int32_t> RenderPrompt(const qfn::Model& model,
                                       const PromptRecord& record,
                                       const GenerationConfig& config) {
  return RenderPrompt(model.tokenizer(), model.VocabSize(), record, config);
}

gufo::json::Value TokenArray(std::span<const std::int32_t> tokens) {
  auto result = gufo::json::Value::array();
  for (const auto token : tokens)
    result.push_back(token);
  return result;
}

gufo::json::Value OutputMetadata(const PromptRecord& source,
                                 std::string_view source_file_sha256,
                                 std::string_view termination) {
  auto result = gufo::json::Value::object();
  result["prompt_source_file_sha256"] = std::string(source_file_sha256);
  result["prompt_source_record_sha256"] = source.canonical_sha256;
  result["termination"] = std::string(termination);
  result["user_metadata"] = source.metadata;
  return result;
}

std::string OutputLine(
    const PromptRecord& source, std::span<const std::int32_t> prompt,
    std::span<const std::int32_t> response, std::string_view target_sha256,
    std::string_view generator_sha256, std::string_view producer_git,
    std::string_view generation_config_sha256,
    std::string_view source_file_sha256, std::string_view termination) {
  auto value = gufo::json::Value::object();
  value["id"] = source.id;
  value["source_id"] = source.source_id;
  value["prompt_tokens"] = TokenArray(prompt);
  value["response_tokens"] = TokenArray(response);
  value["target_sha256"] = std::string(target_sha256);
  value["generator_sha256"] = std::string(generator_sha256);
  value["generator_git_commit"] = std::string(producer_git);
  value["generation_config_sha256"] = std::string(generation_config_sha256);
  value["metadata"] = OutputMetadata(source, source_file_sha256, termination);
  return CanonicalJson(value);
}

std::vector<std::int32_t> ParseTokenArray(const gufo::json::Value* value,
                                          const qfn::Model& model,
                                          std::string_view field) {
  if (value == nullptr || !value->is_array() || value->items().empty())
    throw std::runtime_error(std::string(field) + " is not a non-empty array");
  std::vector<std::int32_t> result;
  result.reserve(value->items().size());
  for (const auto& item : value->items()) {
    const double number = item.as_double(-1.0);
    if (!item.is_number() || !std::isfinite(number) || number < 0.0 ||
        number >
            static_cast<double>(std::numeric_limits<std::uint32_t>::max()) ||
        number != static_cast<double>(static_cast<std::uint32_t>(number)) ||
        number >= model.VocabSize())
      throw std::runtime_error(std::string(field) + " has an invalid token");
    result.push_back(static_cast<std::int32_t>(number));
  }
  return result;
}

void ValidateExistingLine(std::string_view line, const PromptRecord& source,
                          std::span<const std::int32_t> prompt,
                          const qfn::Model& model,
                          const GenerationConfig& config,
                          std::string_view target_sha256,
                          std::string_view generator_sha256,
                          std::string_view producer_git,
                          std::string_view source_file_sha256) {
  const auto value = gufo::json::parse(line);
  constexpr std::array<std::string_view, 9> keys{"generation_config_sha256",
                                                 "generator_git_commit",
                                                 "generator_sha256",
                                                 "id",
                                                 "metadata",
                                                 "prompt_tokens",
                                                 "response_tokens",
                                                 "source_id",
                                                 "target_sha256"};
  RequireKeys(value, keys, "partial response row");
  if (CanonicalJson(value) != line)
    throw std::runtime_error("partial response row is not canonical JSON");
  const auto exact_string = [&](std::string_view key,
                                std::string_view expected) {
    const auto* field = value.find(std::string(key));
    return field != nullptr && field->is_string() && field->str() == expected;
  };
  if (!exact_string("id", source.id) ||
      !exact_string("source_id", source.source_id) ||
      !exact_string("target_sha256", target_sha256) ||
      !exact_string("generator_sha256", generator_sha256) ||
      !exact_string("generator_git_commit", producer_git) ||
      !exact_string("generation_config_sha256", config.sha256))
    throw std::runtime_error("partial response row identity differs");
  const auto prompt_tokens =
      ParseTokenArray(value.find("prompt_tokens"), model, "prompt_tokens");
  const auto response_tokens =
      ParseTokenArray(value.find("response_tokens"), model, "response_tokens");
  if (!std::ranges::equal(prompt_tokens, prompt) ||
      response_tokens.size() > config.max_response_tokens)
    throw std::runtime_error("partial response token sequence differs");
  const auto* metadata = value.find("metadata");
  const std::string_view termination =
      response_tokens.size() == config.max_response_tokens ? "length" : "stop";
  if (metadata == nullptr ||
      CanonicalJson(*metadata) != CanonicalJson(OutputMetadata(
                                      source, source_file_sha256, termination)))
    throw std::runtime_error("partial response source binding differs");
}

std::filesystem::path ExecutablePath() {
#ifdef _WIN32
  std::wstring buffer(32768, L'\0');
  const DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
                                          static_cast<DWORD>(buffer.size()));
  if (length == 0 || length == buffer.size())
    throw std::runtime_error("cannot resolve generator executable");
  buffer.resize(length);
  return std::filesystem::path(buffer);
#else
  std::error_code error;
  const auto path = std::filesystem::read_symlink("/proc/self/exe", error);
  if (error || path.empty())
    throw std::runtime_error("cannot resolve generator executable");
  return path;
#endif
}

class DurableOutput {
public:
  DurableOutput(const std::filesystem::path& path, bool append) {
#ifdef _WIN32
    int descriptor = -1;
    const int flags =
        _O_BINARY | _O_WRONLY | (append ? _O_APPEND : (_O_CREAT | _O_EXCL));
    const errno_t status = _wsopen_s(&descriptor, path.c_str(), flags,
                                     _SH_DENYRW, _S_IREAD | _S_IWRITE);
    if (status != 0 || descriptor < 0)
      throw std::runtime_error("cannot open partial response export");
    file_ = _fdopen(descriptor, append ? "ab" : "wb");
    if (file_ == nullptr) {
      _close(descriptor);
      throw std::runtime_error("cannot open partial response export stream");
    }
#else
    const int flags = O_WRONLY | O_CLOEXEC | O_NOFOLLOW |
                      (append ? O_APPEND : (O_CREAT | O_EXCL));
    const int descriptor = ::open(path.c_str(), flags, 0600);
    if (descriptor < 0)
      throw std::runtime_error("cannot open partial response export");
    file_ = ::fdopen(descriptor, append ? "ab" : "wb");
    if (file_ == nullptr) {
      ::close(descriptor);
      throw std::runtime_error("cannot open partial response export stream");
    }
#endif
  }

  ~DurableOutput() {
    if (file_ != nullptr)
      std::fclose(file_);
  }

  void Commit(std::string_view line) {
    if (std::fwrite(line.data(), 1, line.size(), file_) != line.size() ||
        std::fputc('\n', file_) == EOF || std::fflush(file_) != 0)
      throw std::runtime_error("response export write failed");
#ifdef _WIN32
    if (_commit(_fileno(file_)) != 0)
      throw std::runtime_error("response export durable commit failed");
#else
    if (::fsync(::fileno(file_)) != 0)
      throw std::runtime_error("response export durable commit failed");
#endif
  }

  void Close() {
    if (std::fclose(file_) != 0) {
      file_ = nullptr;
      throw std::runtime_error("response export close failed");
    }
    file_ = nullptr;
  }

private:
  std::FILE* file_{nullptr};
};

void PublishNoReplace(const std::filesystem::path& partial,
                      const std::filesystem::path& output) {
#ifdef _WIN32
  if (!MoveFileExW(partial.c_str(), output.c_str(), MOVEFILE_WRITE_THROUGH))
    throw std::runtime_error("non-overwriting response publication failed");
#else
  if (::link(partial.c_str(), output.c_str()) != 0)
    throw std::runtime_error("non-overwriting response publication failed");
  const auto directory = output.parent_path().empty()
                             ? std::filesystem::path(".")
                             : output.parent_path();
  const int descriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
  if (descriptor < 0 || ::fsync(descriptor) != 0) {
    if (descriptor >= 0)
      ::close(descriptor);
    throw std::runtime_error("response directory publication flush failed");
  }
  ::close(descriptor);
  if (::unlink(partial.c_str()) != 0)
    throw std::runtime_error("published response but could not unlink partial");
#endif
}

struct Arguments {
  std::filesystem::path model;
  std::filesystem::path input;
  std::filesystem::path output;
  std::filesystem::path generation_config;
  std::string input_sha256;
  std::string target_sha256;
  std::string generator_sha256;
  std::string generation_config_sha256;
  std::string producer_git;
  std::size_t expected_records{};
  bool resume{false};
};

Arguments ParseArguments(int argc, char** argv) {
  Arguments args;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--resume") {
      args.resume = true;
      continue;
    }
    if (index + 1 >= argc)
      throw std::runtime_error("missing option value");
    const std::string value = argv[++index];
    if (option == "--model")
      args.model = value;
    else if (option == "--input")
      args.input = value;
    else if (option == "--output")
      args.output = value;
    else if (option == "--generation-config")
      args.generation_config = value;
    else if (option == "--generation-config-sha256")
      args.generation_config_sha256 = value;
    else if (option == "--input-sha256")
      args.input_sha256 = value;
    else if (option == "--target-sha256")
      args.target_sha256 = value;
    else if (option == "--generator-sha256")
      args.generator_sha256 = value;
    else if (option == "--producer-git")
      args.producer_git = value;
    else if (option == "--expected-records")
      args.expected_records = std::stoull(value);
    else
      throw std::runtime_error("unknown option: " + std::string(option));
  }
  if (args.model.empty() || args.input.empty() || args.output.empty() ||
      args.generation_config.empty() || args.generation_config_sha256.empty() ||
      args.input_sha256.empty() || args.target_sha256.empty() ||
      args.generator_sha256.empty() || args.producer_git.empty() ||
      args.expected_records == 0)
    throw std::runtime_error(
        "requires --model GGUF --input PROMPTS.jsonl --output RESPONSES.jsonl "
        "--generation-config CONFIG.json --generation-config-sha256 SHA256 "
        "--input-sha256 SHA256 --target-sha256 SHA256 "
        "--generator-sha256 SHA256 --producer-git COMMIT "
        "--expected-records N [--resume]");
  if (!IsLowerHex(args.input_sha256, 64) ||
      !IsLowerHex(args.target_sha256, 64) ||
      !IsLowerHex(args.generator_sha256, 64) ||
      !IsLowerHex(args.producer_git, 40))
    throw std::runtime_error(
        "reviewed identities must be lowercase hexadecimal");
  return args;
}

bool EndsWithNewline(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    return false;
  const auto size = input.tellg();
  if (size <= 0)
    return false;
  input.seekg(-1, std::ios::end);
  return input.get() == '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 7 && std::string_view(argv[1]) == "--self-test-output" &&
        std::string_view(argv[3]) == "--generation-config" &&
        std::string_view(argv[5]) == "--generation-config-sha256") {
      const std::filesystem::path directory = argv[2];
      if (!std::filesystem::is_directory(directory))
        throw std::runtime_error(
            "self-test output must be an existing directory");
      const GenerationConfig config = ParseGenerationConfig(argv[4], argv[6]);
      const std::filesystem::path output =
          directory / "flashmap-responses-selftest.jsonl";
      const std::filesystem::path partial =
          directory / "flashmap-responses-selftest.jsonl.partial";
      if (std::filesystem::exists(output) ||
          std::filesystem::is_symlink(output) ||
          std::filesystem::exists(partial) ||
          std::filesystem::is_symlink(partial))
        throw std::runtime_error("refusing to overwrite self-test output");
      auto metadata = gufo::json::Value::object();
      metadata["kind"] = "synthetic";
      PromptRecord record{.id = "synthetic-sample",
                          .source_id = "synthetic-source",
                          .metadata = std::move(metadata),
                          .canonical_sha256 =
                              "333333333333333333333333333333333333333333333333"
                              "3333333333333333"};
      constexpr std::array<std::int32_t, 2> prompt{1, 2};
      constexpr std::array<std::int32_t, 1> response{3};
      constexpr std::string_view target =
          "1111111111111111111111111111111111111111111111111111111111111111";
      constexpr std::string_view git =
          "2222222222222222222222222222222222222222";
      constexpr std::string_view source =
          "4444444444444444444444444444444444444444444444444444444444444444";
      const std::string generator =
          gufo::crypto::Sha256FileHex(ExecutablePath());
      DurableOutput sink(partial, false);
      sink.Commit(OutputLine(record, prompt, response, target, generator, git,
                             config.sha256, source, "stop"));
      sink.Close();
      PublishNoReplace(partial, output);
      std::printf(
          "FLASHMAP_RESPONSES_SELFTEST_WRITTEN target=%s generator=%s "
          "producer_git=%s generation_config=%s sha256=%s output=%s\n",
          std::string(target).c_str(), generator.c_str(),
          std::string(git).c_str(), config.sha256.c_str(),
          gufo::crypto::Sha256FileHex(output).c_str(), output.string().c_str());
      return 0;
    }
    const Arguments args = ParseArguments(argc, argv);
    const auto partial =
        std::filesystem::path(args.output.string() + ".partial");
    if (std::filesystem::exists(args.output) ||
        std::filesystem::is_symlink(args.output))
      throw std::runtime_error("refusing to overwrite response export");
    if (std::filesystem::is_symlink(partial))
      throw std::runtime_error("partial response export must not be a symlink");
    const bool partial_exists = std::filesystem::exists(partial);
    if (partial_exists != args.resume)
      throw std::runtime_error(
          args.resume ? "resume requires an existing partial export"
                      : "partial export exists; use --resume");
    if (!EndsWithNewline(args.input))
      throw std::runtime_error(
          "prompt source must be non-empty and newline-terminated");
    if (partial_exists && std::filesystem::file_size(partial) != 0 &&
        !EndsWithNewline(partial))
      throw std::runtime_error(
          "partial response export is not newline-terminated");

    const GenerationConfig config = ParseGenerationConfig(
        args.generation_config, args.generation_config_sha256);
    const std::string source_file_sha256 =
        gufo::crypto::Sha256FileHex(args.input);
    const std::string generator_sha256 =
        gufo::crypto::Sha256FileHex(ExecutablePath());
    if (source_file_sha256 != args.input_sha256)
      throw std::runtime_error("prompt source identity differs");
    if (generator_sha256 != args.generator_sha256)
      throw std::runtime_error("generator executable identity differs");

    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(args.model, &error);
    if (!reader)
      throw std::runtime_error("model metadata load failed: " + error);
    auto chat_template =
        gufo::tokenization::QwenChatTemplate::CreateFromGguf(*reader, &error);
    if (!chat_template)
      throw std::runtime_error("chat template load failed: " + error);
    if (chat_template->GetTemplateSha256() != config.chat_template_sha256)
      throw std::runtime_error(
          "model chat template differs from reviewed config");
    auto tokenizer =
        gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
    if (!tokenizer)
      throw std::runtime_error("model tokenizer load failed: " + error);

    // Prove the whole immutable corpus fits before HIP initialization, model
    // upload, or the first durable response row. Generation repeats the same
    // checks so this early pass cannot weaken the per-record contract.
    {
      std::ifstream preflight(args.input, std::ios::binary);
      if (!preflight)
        throw std::runtime_error("cannot open prompt source for preflight");
      std::unordered_set<std::string> ids;
      std::size_t records = 0;
      std::size_t max_prompt_tokens = 0;
      std::string max_prompt_id;
      std::string line;
      while (std::getline(preflight, line)) {
        if (line.empty() || line.size() > kMaxJsonLineBytes)
          throw std::runtime_error(
              "prompt source preflight row is empty or too large");
        PromptRecord record = ParsePromptRecord(gufo::json::parse(line));
        if (!ids.insert(record.id).second)
          throw std::runtime_error("prompt source repeats id " + record.id);
        const auto tokens = RenderPrompt(*tokenizer, tokenizer->GetVocabSize(),
                                         record, config);
        if (tokens.size() > max_prompt_tokens) {
          max_prompt_tokens = tokens.size();
          max_prompt_id = record.id;
        }
        ++records;
      }
      if (!preflight.eof())
        throw std::runtime_error("prompt source preflight read failed");
      if (records != args.expected_records)
        throw std::runtime_error(
            "prompt source preflight record count differs from expected");
      if (gufo::crypto::Sha256FileHex(args.input) != source_file_sha256)
        throw std::runtime_error("prompt source changed during preflight");
      std::fprintf(stderr,
                   "FLASHMAP_PROMPTS_PREFLIGHT_OK records=%zu max_prompt=%zu "
                   "required_context=%zu max_id=%s\n",
                   records, max_prompt_tokens,
                   max_prompt_tokens + config.max_response_tokens,
                   max_prompt_id.c_str());
    }
    tokenizer.reset();
    reader.reset();

    if (const hipError_t status = hipInit(0); status != hipSuccess)
      throw std::runtime_error(std::string("HIP initialization failed: ") +
                               hipGetErrorString(status));
    qfn::ModelOptions model_options;
    model_options.max_context = config.max_context;
    auto model = qfn::Model::Load(args.model.string(), model_options, &error);
    if (!model)
      throw std::runtime_error("model load failed: " + error);
    const std::string target_sha256 = model->ArtifactIdentityHex();
    if (target_sha256 != args.target_sha256)
      throw std::runtime_error("target artifact identity differs");

    std::ifstream source(args.input, std::ios::binary);
    if (!source)
      throw std::runtime_error("cannot open prompt source");
    std::vector<std::string> existing_lines;
    if (args.resume) {
      if (std::filesystem::file_size(partial) > kMaxResumeBytes)
        throw std::runtime_error("partial response export exceeds 512 MiB");
      std::ifstream existing(partial, std::ios::binary);
      if (!existing)
        throw std::runtime_error("cannot open partial response export");
      std::string line;
      while (std::getline(existing, line)) {
        if (line.empty() || line.size() > kMaxJsonLineBytes)
          throw std::runtime_error(
              "partial response row is empty or too large");
        existing_lines.push_back(std::move(line));
        if (existing_lines.size() > args.expected_records)
          throw std::runtime_error("partial response export has extra rows");
      }
      if (!existing.eof())
        throw std::runtime_error("partial response export read failed");
    }
    DurableOutput output(partial, args.resume);
    auto session = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, config.max_context, &error);
    if (!session)
      throw std::runtime_error("session creation failed: " + error);

    std::unordered_set<std::string> ids;
    std::size_t records = 0;
    std::size_t resumed = 0;
    std::size_t prompt_token_total = 0;
    std::size_t response_token_total = 0;
    std::string source_line;
    while (std::getline(source, source_line)) {
      if (source_line.empty() || source_line.size() > kMaxJsonLineBytes)
        throw std::runtime_error("prompt source row is empty or too large");
      PromptRecord record = ParsePromptRecord(gufo::json::parse(source_line));
      if (!ids.insert(record.id).second)
        throw std::runtime_error("prompt source repeats id " + record.id);
      const std::vector<std::int32_t> prompt =
          RenderPrompt(*model, record, config);
      prompt_token_total += prompt.size();

      if (records < existing_lines.size()) {
        ValidateExistingLine(existing_lines[records], record, prompt, *model,
                             config, target_sha256, generator_sha256,
                             args.producer_git, source_file_sha256);
        const auto value = gufo::json::parse(existing_lines[records]);
        response_token_total += value.find("response_tokens")->items().size();
        ++resumed;
        ++records;
        continue;
      }
      session->Reset();
      if (!session->Sync(prompt, &error))
        throw std::runtime_error("target prefill failed: " + error);
      const std::vector<gufo::sampling::TokenId> sampling_prompt(prompt.begin(),
                                                                 prompt.end());
      gufo::sampling::SamplerState sampler({}, sampling_prompt);
      std::vector<std::int32_t> response;
      while (response.size() < config.max_response_tokens) {
        qfn::Session::DecodeResult decoded;
        if (!session->DecodeStep(config.max_response_tokens - response.size(),
                                 sampler, &decoded, &error, true))
          throw std::runtime_error("target decode failed: " + error);
        response.insert(response.end(), decoded.tokens.begin(),
                        decoded.tokens.end());
        if (decoded.stop)
          break;
        if (decoded.tokens.empty())
          throw std::runtime_error("target decode made no progress");
      }
      if (response.empty())
        throw std::runtime_error("target produced no response tokens");
      const std::string line = OutputLine(
          record, prompt, response, target_sha256, generator_sha256,
          args.producer_git, config.sha256, source_file_sha256,
          response.size() == config.max_response_tokens ? "length" : "stop");
      if (line.size() > kMaxJsonLineBytes)
        throw std::runtime_error("response export row exceeds 16 MiB");
      output.Commit(line);
      response_token_total += response.size();
      ++records;
      std::fprintf(stderr,
                   "FLASHMAP_RESPONSE_COMMITTED ordinal=%zu prompt=%zu "
                   "response=%zu\n",
                   records - 1, prompt.size(), response.size());
    }
    if (!source.eof())
      throw std::runtime_error("prompt source read failed");
    if (resumed != existing_lines.size())
      throw std::runtime_error("partial response export has extra rows");
    if (records != args.expected_records)
      throw std::runtime_error(
          "prompt source record count differs from expected");
    if (gufo::crypto::Sha256FileHex(args.input) != source_file_sha256)
      throw std::runtime_error("prompt source changed during generation");
    output.Close();
    PublishNoReplace(partial, args.output);
    const std::string output_sha256 = gufo::crypto::Sha256FileHex(args.output);
    std::printf(
        "FLASHMAP_RESPONSES_OK records=%zu resumed=%zu prompt_tokens=%zu "
        "response_tokens=%zu target=%s generator=%s producer_git=%s "
        "generation_config=%s source=%s sha256=%s output=%s\n",
        records, resumed, prompt_token_total, response_token_total,
        target_sha256.c_str(), generator_sha256.c_str(),
        args.producer_git.c_str(), config.sha256.c_str(),
        source_file_sha256.c_str(), output_sha256.c_str(),
        args.output.string().c_str());
    return 0;
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "FLASHMAP_RESPONSES_FAILED: %s\n", exception.what());
    return 1;
  }
}
