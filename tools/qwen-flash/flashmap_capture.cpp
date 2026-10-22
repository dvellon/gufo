#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
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
#include "src/core/json.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace {

constexpr std::array<std::uint32_t, 6> kLayerTaps{1, 12, 23, 34, 45, 47};
constexpr std::array<std::uint8_t, 8> kFileMagic{'G', 'F', 'M', 'T', 'R', 'C',
                                                 '2', 0};
constexpr std::array<std::uint8_t, 8> kRecordMagic{'G', 'F', 'M', 'R', 'E', 'C',
                                                   '2', 0};
constexpr std::array<std::uint8_t, 8> kCommitMagic{'G', 'F', 'M', 'E', 'N', 'D',
                                                   '2', 0};
constexpr std::string_view kIdentityScheme = "gguf-sha256-v1";
constexpr std::string_view kFeatureAbi =
    "gufo.flashmap.feature.v1\n"
    "architecture=qwen4exp\n"
    "alignment=post-input-token\n"
    "tap-index=zero-based-completed-layer\n"
    "taps=1,12,23,34,45,47\n"
    "aux-count=5\n"
    "final-tap=47\n"
    "readout=target-final-hyperconnection-mixer\n"
    "hidden=2560\n"
    "capture=f32\n"
    "storage=bf16-rne-le\n"
    "layout=token,tap,hidden\n"
    "target-only=true\n";
constexpr std::size_t kFileHeaderBytes = 320;
constexpr std::size_t kRecordHeaderBytes = 192;
constexpr std::size_t kRecordTrailerBytes = 64;
constexpr std::size_t kAlignment = 64;
constexpr std::size_t kMaxMetadataBytes = 64 * 1024;
constexpr std::size_t kMaxJsonLineBytes = 16 * 1024 * 1024;

void PutU16(std::span<std::uint8_t> out, std::size_t offset,
            std::uint16_t value) {
  if (offset + 2 > out.size())
    throw std::logic_error("trace U16 write exceeds buffer");
  out[offset] = static_cast<std::uint8_t>(value);
  out[offset + 1] = static_cast<std::uint8_t>(value >> 8);
}

void PutU32(std::span<std::uint8_t> out, std::size_t offset,
            std::uint32_t value) {
  if (offset + 4 > out.size())
    throw std::logic_error("trace U32 write exceeds buffer");
  for (unsigned shift = 0; shift != 32; shift += 8)
    out[offset++] = static_cast<std::uint8_t>(value >> shift);
}

void PutU64(std::span<std::uint8_t> out, std::size_t offset,
            std::uint64_t value) {
  if (offset + 8 > out.size())
    throw std::logic_error("trace U64 write exceeds buffer");
  for (unsigned shift = 0; shift != 64; shift += 8)
    out[offset++] = static_cast<std::uint8_t>(value >> shift);
}

std::array<std::uint8_t, 32> HashBytes(std::span<const std::uint8_t> bytes) {
  gufo::crypto::Sha256Hasher hash;
  hash.Update(bytes);
  return hash.Finish();
}

std::array<std::uint8_t, 32> HashString(std::string_view value) {
  return HashBytes({reinterpret_cast<const std::uint8_t*>(value.data()),
                    value.size()});
}

std::array<std::uint8_t, 32> DecodeSha256(std::string_view value) {
  if (value.size() != 64)
    throw std::runtime_error("identity is not SHA-256");
  std::array<std::uint8_t, 32> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    const auto digit = [](char c) -> int {
      if (c >= '0' && c <= '9')
        return c - '0';
      if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
      if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
      return -1;
    };
    const int high = digit(value[index * 2]);
    const int low = digit(value[index * 2 + 1]);
    if (high < 0 || low < 0)
      throw std::runtime_error("identity is not hexadecimal");
    result[index] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return result;
}

std::array<std::uint8_t, 20> DecodeGitCommit(std::string_view value) {
  if (value.size() != 40)
    throw std::runtime_error("producer Git commit is not 40 hexadecimal digits");
  std::array<std::uint8_t, 20> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    const auto digit = [](char c) -> int {
      if (c >= '0' && c <= '9')
        return c - '0';
      if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
      if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
      return -1;
    };
    const int high = digit(value[index * 2]);
    const int low = digit(value[index * 2 + 1]);
    if (high < 0 || low < 0)
      throw std::runtime_error("producer Git commit is not hexadecimal");
    result[index] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return result;
}

std::string Hex(std::span<const std::uint8_t> bytes) {
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result(bytes.size() * 2, '0');
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    result[index * 2] = digits[bytes[index] >> 4];
    result[index * 2 + 1] = digits[bytes[index] & 15];
  }
  return result;
}

std::size_t CheckedAdd(std::size_t left, std::size_t right) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::runtime_error("trace size overflows host address space");
  return left + right;
}

std::size_t CheckedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::runtime_error("trace size overflows host address space");
  return left * right;
}

std::size_t Align(std::size_t value) {
  return CheckedAdd(value, kAlignment - 1) & ~(kAlignment - 1);
}

std::uint16_t FloatToBf16(float value) {
  if (!std::isfinite(value))
    throw std::runtime_error("non-finite target feature");
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
  const auto result = static_cast<std::uint16_t>(rounded >> 16);
  if ((result & 0x7F80u) == 0x7F80u)
    throw std::runtime_error("target feature overflows finite BF16");
  return result;
}

gufo::json::Value CanonicalValue(const gufo::json::Value& value) {
  // Gufo's minimal JSON value stores every number as a double, so it cannot
  // preserve whether the source used an integer, a decimal, or negative zero.
  // Python's independent validator must reproduce the canonical metadata
  // bytes exactly. Keep metadata in their shared lossless JSON subset instead
  // of silently publishing a shard that another implementation rejects.
  if (value.is_number())
    throw std::runtime_error("numeric trace metadata is not portable");
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

std::filesystem::path ExecutablePath() {
#ifdef _WIN32
  std::wstring buffer(32768, L'\0');
  const DWORD length =
      GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
  if (length == 0 || length == buffer.size())
    throw std::runtime_error("cannot resolve producer executable");
  buffer.resize(length);
  return std::filesystem::path(buffer);
#else
  std::error_code error;
  const auto path = std::filesystem::read_symlink("/proc/self/exe", error);
  if (error || path.empty())
    throw std::runtime_error("cannot resolve producer executable");
  return path;
#endif
}

class TraceSink {
public:
  explicit TraceSink(const std::filesystem::path& path) {
#ifdef _WIN32
    int descriptor = -1;
    const errno_t status =
        _wsopen_s(&descriptor, path.c_str(),
                  _O_BINARY | _O_CREAT | _O_EXCL | _O_WRONLY, _SH_DENYRW,
                  _S_IREAD | _S_IWRITE);
    if (status != 0 || descriptor < 0)
      throw std::runtime_error("cannot exclusively create partial trace");
    file_ = _fdopen(descriptor, "wb");
    if (file_ == nullptr) {
      _close(descriptor);
      throw std::runtime_error("cannot open partial trace stream");
    }
#else
    const int descriptor =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
               0600);
    if (descriptor < 0)
      throw std::runtime_error("cannot exclusively create partial trace");
    file_ = ::fdopen(descriptor, "wb");
    if (file_ == nullptr) {
      ::close(descriptor);
      throw std::runtime_error("cannot open partial trace stream");
    }
#endif
  }

  TraceSink(const TraceSink&) = delete;
  TraceSink& operator=(const TraceSink&) = delete;

  ~TraceSink() {
    if (file_ != nullptr)
      std::fclose(file_);
  }

  void Write(std::span<const std::uint8_t> bytes,
             gufo::crypto::Sha256Hasher* record = nullptr) {
    std::size_t written = 0;
    while (written < bytes.size()) {
      const std::size_t count =
          std::fwrite(bytes.data() + written, 1, bytes.size() - written, file_);
      if (count == 0)
        throw std::runtime_error("trace write failed");
      written += count;
    }
    file_hash_.Update(bytes);
    if (record != nullptr)
      record->Update(bytes);
  }

  void WriteZeros(std::size_t count, gufo::crypto::Sha256Hasher* record) {
    constexpr std::array<std::uint8_t, kAlignment> zeros{};
    while (count != 0) {
      const std::size_t chunk = std::min(count, zeros.size());
      Write(std::span<const std::uint8_t>(zeros).first(chunk), record);
      count -= chunk;
    }
  }

  void Durable() {
    if (std::fflush(file_) != 0)
      throw std::runtime_error("trace stdio flush failed");
#ifdef _WIN32
    if (_commit(_fileno(file_)) != 0)
      throw std::runtime_error("trace durable commit failed");
#else
    if (::fsync(::fileno(file_)) != 0)
      throw std::runtime_error("trace durable commit failed");
#endif
  }

  std::array<std::uint8_t, 32> FinishHash() { return file_hash_.Finish(); }

  void Close() {
    Durable();
    if (std::fclose(file_) != 0) {
      file_ = nullptr;
      throw std::runtime_error("trace close failed");
    }
    file_ = nullptr;
  }

private:
  std::FILE* file_{nullptr};
  gufo::crypto::Sha256Hasher file_hash_;
};

void PublishNoReplace(const std::filesystem::path& partial,
                      const std::filesystem::path& output) {
#ifdef _WIN32
  if (!MoveFileExW(partial.c_str(), output.c_str(), MOVEFILE_WRITE_THROUGH))
    throw std::runtime_error("non-overwriting trace publication failed");
#else
  if (::link(partial.c_str(), output.c_str()) != 0)
    throw std::runtime_error("non-overwriting trace publication failed");
  const auto directory = output.parent_path().empty()
                             ? std::filesystem::path(".")
                             : output.parent_path();
  const int descriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
  if (descriptor < 0 || ::fsync(descriptor) != 0) {
    if (descriptor >= 0)
      ::close(descriptor);
    throw std::runtime_error("trace directory publication flush failed");
  }
  ::close(descriptor);
  if (::unlink(partial.c_str()) != 0)
    throw std::runtime_error("published trace but could not unlink partial");
#endif
}

std::array<std::uint8_t, 16> MakeUuid() {
  std::array<std::uint8_t, 16> result{};
  std::random_device random;
  for (auto& byte : result)
    byte = static_cast<std::uint8_t>(random());
  result[6] = static_cast<std::uint8_t>((result[6] & 0x0F) | 0x40);
  result[8] = static_cast<std::uint8_t>((result[8] & 0x3F) | 0x80);
  return result;
}

std::vector<std::uint8_t> MakeFileHeader(
    std::uint32_t hidden, std::uint32_t max_context, std::uint32_t vocab_size,
    std::string_view target_identity,
    const std::array<std::uint8_t, 32>& producer_identity,
    const std::array<std::uint8_t, 20>& producer_git_commit,
    std::uint64_t shard_ordinal,
    const std::array<std::uint8_t, 16>& shard_uuid) {
  std::vector<std::uint8_t> bytes(kFileHeaderBytes);
  std::ranges::copy(kFileMagic, bytes.begin());
  PutU16(bytes, 8, 2);
  PutU16(bytes, 10, 0);
  PutU32(bytes, 12, kFileHeaderBytes);
  PutU32(bytes, 16, 1);  // BF16 round-to-nearest-even.
  PutU32(bytes, 20, 1);  // U32LE tokens.
  PutU32(bytes, 24, 1);  // BF16-RNE-LE features.
  PutU32(bytes, 28, hidden);
  PutU32(bytes, 32, static_cast<std::uint32_t>(kLayerTaps.size()));
  PutU32(bytes, 36, max_context);
  PutU32(bytes, 40, vocab_size);
  for (std::size_t index = 0; index < kLayerTaps.size(); ++index)
    PutU32(bytes, 48 + index * 4, kLayerTaps[index]);
  std::memcpy(bytes.data() + 80, kIdentityScheme.data(), kIdentityScheme.size());
  const auto target = DecodeSha256(target_identity);
  std::ranges::copy(target, bytes.begin() + 96);
  const auto abi = HashString(kFeatureAbi);
  std::ranges::copy(abi, bytes.begin() + 128);
  std::ranges::copy(producer_identity, bytes.begin() + 160);
  std::ranges::copy(producer_git_commit, bytes.begin() + 192);
  PutU64(bytes, 216, shard_ordinal);
  const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  PutU64(bytes, 224, static_cast<std::uint64_t>(now));
  std::ranges::copy(shard_uuid, bytes.begin() + 232);
  const auto digest =
      HashBytes(std::span<const std::uint8_t>(bytes).first(288));
  std::ranges::copy(digest, bytes.begin() + 288);
  return bytes;
}

struct InputRecord {
  std::string id;
  std::string source_id;
  std::vector<std::int32_t> tokens;
  std::string metadata;
  std::uint32_t flags{};
};

InputRecord ParseRecord(const gufo::json::Value& value, const qfn::Model& model,
                        std::uint32_t max_context) {
  if (!value.is_object())
    throw std::runtime_error("each JSONL row must be an object");
  const auto* id = value.find("id");
  const auto* source_id = value.find("source_id");
  const auto* text = value.find("text");
  const auto* token_values = value.find("tokens");
  const auto* metadata = value.find("metadata");
  if (id == nullptr || !id->is_string() || id->str().empty())
    throw std::runtime_error("JSONL row requires non-empty string id");
  if (source_id == nullptr || !source_id->is_string() ||
      source_id->str().empty())
    throw std::runtime_error("JSONL row requires non-empty string source_id");
  if ((text != nullptr) == (token_values != nullptr))
    throw std::runtime_error("JSONL row requires exactly one of text or tokens");
  InputRecord result{.id = id->str(), .source_id = source_id->str()};
  if (text != nullptr) {
    if (!text->is_string() || text->str().empty())
      throw std::runtime_error("text must be a non-empty string");
    result.tokens = model.Tokenize(text->str());
  } else {
    if (!token_values->is_array() || token_values->items().empty())
      throw std::runtime_error("tokens must be a non-empty array");
    result.tokens.reserve(token_values->items().size());
    for (const auto& token : token_values->items()) {
      const double numeric = token.as_double(-1.0);
      if (!token.is_number() || !std::isfinite(numeric) || numeric < 0.0 ||
          std::floor(numeric) != numeric || numeric >= model.VocabSize())
        throw std::runtime_error("token is outside the target vocabulary");
      result.tokens.push_back(static_cast<std::int32_t>(numeric));
    }
  }
  if (result.tokens.empty() || result.tokens.size() > max_context)
    throw std::runtime_error("tokenized record exceeds capture envelope");
  for (std::int32_t token : result.tokens) {
    if (token < 0 || static_cast<std::uint32_t>(token) >= model.VocabSize())
      throw std::runtime_error("token is outside the target vocabulary");
  }
  if (metadata != nullptr) {
    if (!metadata->is_object())
      throw std::runtime_error("metadata must be an object");
    result.metadata = CanonicalValue(*metadata).dump();
    if (result.metadata.size() > kMaxMetadataBytes)
      throw std::runtime_error("metadata exceeds 64 KiB");
  }
  return result;
}

std::vector<std::uint8_t> SerializeTokens(
    std::span<const std::int32_t> tokens) {
  std::vector<std::uint8_t> bytes(tokens.size() * sizeof(std::uint32_t));
  for (std::size_t index = 0; index < tokens.size(); ++index)
    PutU32(bytes, index * 4, static_cast<std::uint32_t>(tokens[index]));
  return bytes;
}

void WriteRecord(TraceSink* sink, const InputRecord& input,
                 std::span<const float> features, std::uint64_t ordinal) {
  const std::size_t expected = CheckedMultiply(
      CheckedMultiply(input.tokens.size(), kLayerTaps.size()), 2560);
  if (features.size() != expected)
    throw std::runtime_error("capture returned the wrong feature shape");
  const auto token_bytes = SerializeTokens(input.tokens);
  const std::size_t metadata_offset = kRecordHeaderBytes;
  const std::size_t token_offset =
      Align(CheckedAdd(metadata_offset, input.metadata.size()));
  const std::size_t feature_offset =
      Align(CheckedAdd(token_offset, token_bytes.size()));
  const std::size_t feature_bytes =
      CheckedMultiply(features.size(), sizeof(std::uint16_t));
  const std::size_t trailer_offset =
      Align(CheckedAdd(feature_offset, feature_bytes));
  const std::size_t total_bytes =
      CheckedAdd(trailer_offset, kRecordTrailerBytes);
  const auto sample_digest = HashString(input.id);
  const auto source_digest = HashString(input.source_id);
  const auto token_digest = HashBytes(token_bytes);
  std::vector<std::uint8_t> header(kRecordHeaderBytes);
  std::ranges::copy(kRecordMagic, header.begin());
  PutU16(header, 8, 2);
  PutU16(header, 10, 0);
  PutU32(header, 12, kRecordHeaderBytes);
  PutU64(header, 16, total_bytes);
  PutU64(header, 24, ordinal);
  PutU32(header, 32, static_cast<std::uint32_t>(input.tokens.size()));
  PutU32(header, 36, static_cast<std::uint32_t>(input.metadata.size()));
  PutU64(header, 40, metadata_offset);
  PutU64(header, 48, token_offset);
  PutU64(header, 56, feature_offset);
  PutU64(header, 64, trailer_offset);
  PutU64(header, 72, token_bytes.size());
  PutU64(header, 80, feature_bytes);
  std::ranges::copy(sample_digest, header.begin() + 88);
  std::ranges::copy(token_digest, header.begin() + 120);
  std::ranges::copy(source_digest, header.begin() + 152);
  PutU32(header, 184, input.flags);

  gufo::crypto::Sha256Hasher record_hash;
  sink->Write(header, &record_hash);
  sink->Write({reinterpret_cast<const std::uint8_t*>(input.metadata.data()),
               input.metadata.size()},
              &record_hash);
  sink->WriteZeros(token_offset - metadata_offset - input.metadata.size(),
                   &record_hash);
  sink->Write(token_bytes, &record_hash);
  sink->WriteZeros(feature_offset - token_offset - token_bytes.size(),
                   &record_hash);

  constexpr std::size_t kChunkValues = 1u << 16;
  std::vector<std::uint8_t> chunk;
  chunk.reserve(kChunkValues * sizeof(std::uint16_t));
  for (float feature : features) {
    const std::uint16_t bf16 = FloatToBf16(feature);
    chunk.push_back(static_cast<std::uint8_t>(bf16));
    chunk.push_back(static_cast<std::uint8_t>(bf16 >> 8));
    if (chunk.size() == kChunkValues * sizeof(std::uint16_t)) {
      sink->Write(chunk, &record_hash);
      chunk.clear();
    }
  }
  if (!chunk.empty())
    sink->Write(chunk, &record_hash);
  sink->WriteZeros(trailer_offset - feature_offset - feature_bytes,
                   &record_hash);

  std::vector<std::uint8_t> trailer(kRecordTrailerBytes);
  const auto digest = record_hash.Finish();
  std::ranges::copy(digest, trailer.begin());
  PutU64(trailer, 32, total_bytes);
  PutU64(trailer, 40, ordinal);
  PutU32(trailer, 48, static_cast<std::uint32_t>(input.tokens.size()));
  PutU32(trailer, 52, input.flags);
  std::ranges::copy(kCommitMagic, trailer.begin() + 56);
  sink->Write(std::span<const std::uint8_t>(trailer).first(56));
  sink->Write(std::span<const std::uint8_t>(trailer).subspan(56));
  sink->Durable();
}

struct Arguments {
  std::filesystem::path model;
  std::filesystem::path input;
  std::filesystem::path output;
  std::string producer_git;
  std::uint32_t max_context{4096};
  std::uint64_t shard_ordinal{};
};

Arguments ParseArguments(int argc, char** argv) {
  Arguments args;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (index + 1 >= argc)
      throw std::runtime_error("missing option value");
    const std::string value = argv[++index];
    if (option == "--model")
      args.model = value;
    else if (option == "--input")
      args.input = value;
    else if (option == "--output")
      args.output = value;
    else if (option == "--producer-git")
      args.producer_git = value;
    else if (option == "--max-context")
      args.max_context = static_cast<std::uint32_t>(std::stoul(value));
    else if (option == "--shard-ordinal")
      args.shard_ordinal = std::stoull(value);
    else
      throw std::runtime_error("unknown option: " + std::string(option));
  }
  if (args.model.empty() || args.input.empty() || args.output.empty() ||
      args.producer_git.empty() || args.max_context == 0)
    throw std::runtime_error(
        "requires --model GGUF --input RECORDS.jsonl --output DIRECTORY "
        "--producer-git COMMIT [--max-context N] [--shard-ordinal N]");
  (void)DecodeGitCommit(args.producer_git);
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path partial;
  try {
    if (argc == 3 && std::string_view(argv[1]) == "--self-test-output") {
      const std::filesystem::path directory = argv[2];
      if (!std::filesystem::is_directory(directory))
        throw std::runtime_error("self-test output must be an existing directory");
      const auto shard_uuid = MakeUuid();
      partial = directory / ("shard-" + Hex(shard_uuid) + ".gfm.partial");
      const auto producer =
          DecodeSha256(gufo::crypto::Sha256FileHex(ExecutablePath()));
      std::array<std::uint8_t, 20> producer_git{};
      producer_git.fill(0x22);
      constexpr std::string_view target_identity =
          "1111111111111111111111111111111111111111111111111111111111111111";
      TraceSink sink(partial);
      sink.Write(MakeFileHeader(2560, 8, 128, target_identity, producer,
                                producer_git, 0, shard_uuid));
      sink.Durable();
      InputRecord record{.id = "synthetic-sample",
                         .source_id = "synthetic-source",
                         .tokens = {1, 2, 3},
                         .metadata = "{\"kind\":\"synthetic\"}"};
      std::vector<float> features(3 * kLayerTaps.size() * 2560);
      for (std::size_t index = 0; index < features.size(); ++index)
        features[index] = static_cast<float>(static_cast<int>(index % 31) - 15) /
                          16.0F;
      WriteRecord(&sink, record, features, 0);
      const auto file_digest = sink.FinishHash();
      sink.Close();
      const auto output =
          directory / ("sha256-" + Hex(file_digest) + ".gfm");
      PublishNoReplace(partial, output);
      std::printf(
          "FLASHMAP_TRACE_SELFTEST_WRITTEN target=%s abi=%s producer=%s "
          "sha256=%s output=%s\n",
          std::string(target_identity).c_str(), Hex(HashString(kFeatureAbi)).c_str(),
          Hex(producer).c_str(), Hex(file_digest).c_str(),
          output.string().c_str());
      return 0;
    }
    const Arguments args = ParseArguments(argc, argv);
    if (!std::filesystem::is_directory(args.output))
      throw std::runtime_error("output must be an existing directory");
    const auto shard_uuid = MakeUuid();
    partial = args.output /
              ("shard-" + Hex(shard_uuid) + ".gfm.partial");
    std::ifstream input(args.input, std::ios::binary);
    if (!input)
      throw std::runtime_error("cannot open prompt JSONL");
    if (const hipError_t status = hipInit(0); status != hipSuccess)
      throw std::runtime_error(std::string("HIP initialization failed: ") +
                               hipGetErrorString(status));
    std::string error;
    qfn::ModelOptions options;
    options.max_context = args.max_context;
    auto model = qfn::Model::Load(args.model.string(), options, &error);
    if (!model)
      throw std::runtime_error("model load failed: " + error);
    const std::string identity = model->ArtifactIdentityHex();
    const auto producer =
        DecodeSha256(gufo::crypto::Sha256FileHex(ExecutablePath()));
    const auto producer_git = DecodeGitCommit(args.producer_git);
    TraceSink sink(partial);
    sink.Write(MakeFileHeader(model->config().hidden_size, args.max_context,
                              model->VocabSize(), identity, producer,
                              producer_git,
                              args.shard_ordinal, shard_uuid));
    sink.Durable();

    std::uint64_t records = 0;
    std::uint64_t total_tokens = 0;
    std::unordered_set<std::string> sample_ids;
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty())
        continue;
      if (line.size() > kMaxJsonLineBytes)
        throw std::runtime_error("JSONL row exceeds 16 MiB");
      const InputRecord record =
          ParseRecord(gufo::json::parse(line), *model, args.max_context);
      const std::string sample_digest = Hex(HashString(record.id));
      if (!sample_ids.insert(sample_digest).second)
        throw std::runtime_error("duplicate sample id in trace shard");
      std::vector<float> features;
      if (!model->CaptureFlashMapFeatures(record.tokens, kLayerTaps, &features,
                                          &error))
        throw std::runtime_error("feature capture failed: " + error);
      WriteRecord(&sink, record, features, records);
      ++records;
      total_tokens += record.tokens.size();
      std::fprintf(stderr,
                   "FLASHMAP_RECORD_COMMITTED ordinal=%llu tokens=%zu\n",
                   static_cast<unsigned long long>(records - 1),
                   record.tokens.size());
    }
    if (!input.eof())
      throw std::runtime_error("prompt JSONL read failed");
    if (records == 0)
      throw std::runtime_error("prompt JSONL contained no records");
    const auto file_digest = sink.FinishHash();
    sink.Close();
    const auto output =
        args.output / ("sha256-" + Hex(file_digest) + ".gfm");
    PublishNoReplace(partial, output);
    std::printf(
        "FLASHMAP_TRACE_OK records=%llu tokens=%llu target=%s abi=%s "
        "producer=%s producer_git=%s sha256=%s output=%s\n",
        static_cast<unsigned long long>(records),
        static_cast<unsigned long long>(total_tokens), identity.c_str(),
        Hex(HashString(kFeatureAbi)).c_str(), Hex(producer).c_str(),
        args.producer_git.c_str(), Hex(file_digest).c_str(),
        output.string().c_str());
    return 0;
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "FLASHMAP_TRACE_FAILED: %s partial=%s\n",
                 exception.what(), partial.string().c_str());
    return 1;
  }
}
