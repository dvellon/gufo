// FlashMap v2 exact-response generator.
//
// The v1 implementation owns the reviewed row parser, rendering, identity
// validation and durable publication primitives.  Include it into this tool's
// translation unit with its entry point renamed so v2 cannot silently fork
// those contracts while adding immutable shards and fixed decode cohorts.
#define main gufo_flashmap_generate_v1_entry
#include "flashmap_generate.cpp"
#undef main

#include <memory>
#include <sstream>

namespace {

constexpr std::string_view kPackageFormat =
    "gufo.flashmap.exact-target-shards.v2";

struct Shard {
  std::size_t index{};
  std::size_t start{};
  std::size_t end{};
  std::string prompt_file;
  std::string prompt_sha256;
  std::string response_file;
};

struct Package {
  std::size_t decode_width{};
  std::size_t expected_records{};
  std::size_t max_response_tokens{};
  std::size_t shard_records{};
  std::string generation_config_sha256;
  std::string generator_git_commit;
  std::string generator_sha256;
  std::string runtime_identity;
  std::string source_sha256;
  std::string target_sha256;
  std::vector<Shard> shards;
};

struct V2Arguments {
  std::filesystem::path model;
  std::filesystem::path manifest;
  std::filesystem::path responses;
  std::filesystem::path generation_config;
  std::string manifest_sha256;
  std::filesystem::path runtime_manifest;
  bool resume{false};
};

std::string ExactString(const gufo::json::Value& value,
                        std::string_view field) {
  const auto* member = value.find(std::string(field));
  if (member == nullptr || !member->is_string())
    throw std::runtime_error(std::string(field) + " must be a string");
  return member->str();
}

std::size_t ExactIndex(const gufo::json::Value& value, std::string_view field,
                       bool allow_zero = false) {
  const auto* member = value.find(std::string(field));
  if (member == nullptr || !member->is_number())
    throw std::runtime_error(std::string(field) + " must be an integer");
  const double number = member->as_double(-1.0);
  const double minimum = allow_zero ? 0.0 : 1.0;
  if (!std::isfinite(number) || number < minimum || number > 1'000'000.0 ||
      number != static_cast<double>(static_cast<std::size_t>(number)))
    throw std::runtime_error(std::string(field) + " is outside its envelope");
  return static_cast<std::size_t>(number);
}

bool IsSimpleFilename(std::string_view value) {
  if (value.empty() || value == "." || value == "..")
    return false;
  return value.find('/') == std::string_view::npos &&
         value.find('\\') == std::string_view::npos &&
         value.find(':') == std::string_view::npos;
}

Package ParsePackage(const std::filesystem::path& path,
                     std::string_view reviewed_sha256) {
  if (!IsLowerHex(reviewed_sha256, 64))
    throw std::runtime_error("manifest identity is not lowercase SHA-256");
  const std::string bytes = ReadBounded(path, 4 * 1024 * 1024);
  if (bytes.empty() || bytes.back() != '\n')
    throw std::runtime_error("manifest must be newline-terminated");
  const std::string actual = gufo::crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()});
  if (actual != reviewed_sha256)
    throw std::runtime_error("manifest identity differs");
  const auto value =
      gufo::json::parse(std::string_view(bytes).substr(0, bytes.size() - 1));
  constexpr std::array<std::string_view, 12> keys{"decode_width",
                                                  "expected_records",
                                                  "format",
                                                  "generation_config_sha256",
                                                  "generator_git_commit",
                                                  "generator_sha256",
                                                  "max_response_tokens",
                                                  "runtime_identity",
                                                  "shard_records",
                                                  "shards",
                                                  "source_sha256",
                                                  "target_sha256"};
  // The explicit count catches future accidental additions as well as
  // missing fields.  Keep this local rather than relaxing RequireKeys.
  RequireKeys(value, keys, "v2 package manifest");
  if (CanonicalJson(value) + "\n" != bytes)
    throw std::runtime_error("manifest is not canonical JSON");
  if (ExactString(value, "format") != kPackageFormat)
    throw std::runtime_error("manifest format differs");

  Package result;
  result.decode_width = ExactIndex(value, "decode_width");
  result.expected_records = ExactIndex(value, "expected_records");
  result.max_response_tokens = ExactIndex(value, "max_response_tokens");
  result.shard_records = ExactIndex(value, "shard_records");
  result.generation_config_sha256 =
      ExactString(value, "generation_config_sha256");
  result.generator_git_commit = ExactString(value, "generator_git_commit");
  result.generator_sha256 = ExactString(value, "generator_sha256");
  result.runtime_identity = ExactString(value, "runtime_identity");
  result.source_sha256 = ExactString(value, "source_sha256");
  result.target_sha256 = ExactString(value, "target_sha256");
  if (result.decode_width > 8 ||
      result.shard_records % result.decode_width != 0)
    throw std::runtime_error("manifest decode geometry is invalid");
  for (const auto* identity :
       {&result.generation_config_sha256, &result.generator_sha256,
        &result.runtime_identity, &result.source_sha256,
        &result.target_sha256}) {
    if (!IsLowerHex(*identity, 64))
      throw std::runtime_error("manifest contains a non-SHA-256 identity");
  }
  if (!IsLowerHex(result.generator_git_commit, 40))
    throw std::runtime_error("manifest producer commit is invalid");

  const auto* shards = value.find("shards");
  if (shards == nullptr || !shards->is_array() || shards->items().empty())
    throw std::runtime_error("manifest shards must be a non-empty array");
  constexpr std::array<std::string_view, 6> shard_keys{
      "end", "index", "prompt_file", "prompt_sha256", "response_file", "start"};
  std::size_t cursor = 0;
  for (const auto& item : shards->items()) {
    RequireKeys(item, shard_keys, "v2 package shard");
    Shard shard;
    shard.index = ExactIndex(item, "index", true);
    shard.start = ExactIndex(item, "start", true);
    shard.end = ExactIndex(item, "end");
    shard.prompt_file = ExactString(item, "prompt_file");
    shard.prompt_sha256 = ExactString(item, "prompt_sha256");
    shard.response_file = ExactString(item, "response_file");
    const std::string expected_prompt = "prompts-" + [&] {
      char buffer[6]{};
      std::snprintf(buffer, sizeof(buffer), "%05zu", result.shards.size());
      return std::string(buffer);
    }() + ".jsonl";
    const std::string expected_response = "responses-" + [&] {
      char buffer[6]{};
      std::snprintf(buffer, sizeof(buffer), "%05zu", result.shards.size());
      return std::string(buffer);
    }() + ".jsonl";
    if (shard.index != result.shards.size() || shard.start != cursor ||
        shard.end != std::min(result.expected_records,
                              shard.start + result.shard_records) ||
        shard.end <= shard.start || !IsSimpleFilename(shard.prompt_file) ||
        !IsSimpleFilename(shard.response_file) ||
        shard.prompt_file != expected_prompt ||
        shard.response_file != expected_response ||
        !IsLowerHex(shard.prompt_sha256, 64))
      throw std::runtime_error("manifest shard geometry or identity differs");
    cursor = shard.end;
    result.shards.push_back(std::move(shard));
  }
  if (cursor != result.expected_records)
    throw std::runtime_error("manifest shards do not cover expected_records");
  return result;
}

V2Arguments ParseV2Arguments(int argc, char** argv) {
  V2Arguments args;
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
    else if (option == "--package-manifest")
      args.manifest = value;
    else if (option == "--package-manifest-sha256")
      args.manifest_sha256 = value;
    else if (option == "--responses-dir")
      args.responses = value;
    else if (option == "--generation-config")
      args.generation_config = value;
    else if (option == "--runtime-manifest")
      args.runtime_manifest = value;
    else
      throw std::runtime_error("unknown option: " + std::string(option));
  }
  if (args.model.empty() || args.manifest.empty() || args.responses.empty() ||
      args.generation_config.empty() || args.manifest_sha256.empty() ||
      args.runtime_manifest.empty())
    throw std::runtime_error(
        "requires --model GGUF --package-manifest MANIFEST.json "
        "--package-manifest-sha256 SHA256 --responses-dir DIR "
        "--generation-config CONFIG.json --runtime-manifest RUNTIME.json "
        "[--resume]");
  if (!IsLowerHex(args.manifest_sha256, 64))
    throw std::runtime_error("reviewed identities must be lowercase SHA-256");
  return args;
}

std::vector<std::string> ReadResponseLines(const std::filesystem::path& path,
                                           std::size_t limit) {
  if (std::filesystem::file_size(path) > kMaxResumeBytes)
    throw std::runtime_error("response shard exceeds 512 MiB");
  if (std::filesystem::file_size(path) != 0 && !EndsWithNewline(path))
    throw std::runtime_error("response shard is not newline-terminated");
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open response shard");
  std::vector<std::string> result;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.size() > kMaxJsonLineBytes)
      throw std::runtime_error("response shard row is empty or too large");
    result.push_back(std::move(line));
    if (result.size() > limit)
      throw std::runtime_error("response shard has extra rows");
  }
  if (!input.eof())
    throw std::runtime_error("response shard read failed");
  return result;
}

struct PreparedRecord {
  PromptRecord source;
  std::vector<std::int32_t> prompt;
};

std::vector<PromptRecord> ReadPromptShard(const std::filesystem::path& path,
                                          const Shard& shard,
                                          std::unordered_set<std::string>* ids,
                                          gufo::crypto::Sha256Hasher* source) {
  if (std::filesystem::is_symlink(path))
    throw std::runtime_error("prompt shard must not be a symlink");
  const std::string bytes = ReadBounded(path, kMaxResumeBytes);
  if (bytes.empty() || bytes.back() != '\n' ||
      gufo::crypto::Sha256Hex(
          {reinterpret_cast<const std::uint8_t*>(bytes.data()),
           bytes.size()}) != shard.prompt_sha256)
    throw std::runtime_error("prompt shard identity or framing differs");
  if (source != nullptr)
    source->Update(
        {reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()});
  std::istringstream input(bytes);
  std::vector<PromptRecord> records;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.size() > kMaxJsonLineBytes)
      throw std::runtime_error("prompt shard row is empty or too large");
    PromptRecord record = ParsePromptRecord(gufo::json::parse(line));
    if (!ids->insert(record.id).second)
      throw std::runtime_error("prompt package repeats id " + record.id);
    records.push_back(std::move(record));
  }
  if (!input.eof() || records.size() != shard.end - shard.start)
    throw std::runtime_error("prompt shard row count differs");
  return records;
}

std::vector<std::string> GenerateCohort(
    const std::vector<PreparedRecord>& cohort,
    std::vector<std::unique_ptr<qfn::Session>>* sessions,
    const GenerationConfig& config, const Package& package,
    std::string* error) {
  const std::size_t width = cohort.size();
  std::vector<std::unique_ptr<gufo::sampling::SamplerState>> samplers;
  std::vector<std::vector<std::int32_t>> responses(width);
  std::vector<bool> stopped(width, false);
  samplers.reserve(width);
  for (std::size_t i = 0; i < width; ++i) {
    auto& session = *(*sessions)[i];
    session.Reset();
    if (!session.Sync(cohort[i].prompt, error))
      throw std::runtime_error("target prefill failed: " + *error);
    const std::vector<gufo::sampling::TokenId> prompt(cohort[i].prompt.begin(),
                                                      cohort[i].prompt.end());
    samplers.push_back(std::make_unique<gufo::sampling::SamplerState>(
        gufo::sampling::SamplingConfig{}, prompt));
  }
  for (;;) {
    std::vector<qfn::Session::DecodeResult> decoded(width);
    std::vector<qfn::Session::BatchOutcome> outcomes(width);
    std::vector<qfn::Session::DecodeRequest> requests;
    std::vector<std::size_t> active;
    for (std::size_t i = 0; i < width; ++i) {
      if (stopped[i])
        continue;
      const std::size_t remaining =
          config.max_response_tokens - responses[i].size();
      if (remaining == 0) {
        stopped[i] = true;
        continue;
      }
      active.push_back(i);
      requests.push_back({(*sessions)[i].get(), remaining, samplers[i].get(),
                          &decoded[i], true, &outcomes[i]});
    }
    if (requests.empty())
      break;
    std::string batch_error;
    if (!qfn::Session::DecodeBatch(requests, &batch_error)) {
      for (const auto i : active) {
        if (!outcomes[i].completed && !outcomes[i].error.empty())
          batch_error = outcomes[i].error;
      }
      throw std::runtime_error("target batch decode failed: " + batch_error);
    }
    for (const auto i : active) {
      responses[i].insert(responses[i].end(), decoded[i].tokens.begin(),
                          decoded[i].tokens.end());
      if (responses[i].size() > config.max_response_tokens)
        throw std::runtime_error("target exceeded response budget");
      if (decoded[i].tokens.empty() && !decoded[i].stop)
        throw std::runtime_error("target batch decode made no progress");
      stopped[i] =
          decoded[i].stop || responses[i].size() == config.max_response_tokens;
    }
  }
  std::vector<std::string> lines;
  lines.reserve(width);
  for (std::size_t i = 0; i < width; ++i) {
    if (responses[i].empty())
      throw std::runtime_error("target produced no response tokens");
    lines.push_back(OutputLine(
        cohort[i].source, cohort[i].prompt, responses[i], package.target_sha256,
        package.generator_sha256, package.generator_git_commit,
        package.generation_config_sha256, package.source_sha256,
        responses[i].size() == config.max_response_tokens ? "length" : "stop"));
  }
  return lines;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const V2Arguments args = ParseV2Arguments(argc, argv);
    if (std::filesystem::is_symlink(args.manifest))
      throw std::runtime_error("package manifest must not be a symlink");
    if (args.manifest.filename() != "manifest.json")
      throw std::runtime_error(
          "package manifest filename must be manifest.json");
    const Package package = ParsePackage(args.manifest, args.manifest_sha256);
    if (std::filesystem::is_symlink(args.runtime_manifest) ||
        gufo::crypto::Sha256FileHex(args.runtime_manifest) !=
            package.runtime_identity)
      throw std::runtime_error("runtime identity differs from package");
    const GenerationConfig config = ParseGenerationConfig(
        args.generation_config, package.generation_config_sha256);
    if (config.max_response_tokens != package.max_response_tokens)
      throw std::runtime_error("response budget differs from package");
    if (gufo::crypto::Sha256FileHex(ExecutablePath()) !=
        package.generator_sha256)
      throw std::runtime_error("generator executable identity differs");
    if (!std::filesystem::is_directory(args.responses) ||
        std::filesystem::is_symlink(args.responses))
      throw std::runtime_error(
          "responses directory must be an existing real directory");
    const auto package_directory = args.manifest.parent_path();
    std::unordered_set<std::string> allowed_package_files{"manifest.json"};
    for (const auto& shard : package.shards)
      allowed_package_files.insert(shard.prompt_file);
    for (const auto& entry :
         std::filesystem::directory_iterator(package_directory)) {
      if (entry.is_symlink() || !entry.is_regular_file() ||
          !allowed_package_files.contains(entry.path().filename().string()))
        throw std::runtime_error("package directory has an unexpected entry");
    }
    std::unordered_set<std::string> allowed_response_files;
    for (const auto& shard : package.shards) {
      allowed_response_files.insert(shard.response_file);
      allowed_response_files.insert(shard.response_file + ".partial");
    }
    for (const auto& entry :
         std::filesystem::directory_iterator(args.responses)) {
      if (entry.is_symlink() || !entry.is_regular_file() ||
          !allowed_response_files.contains(entry.path().filename().string()))
        throw std::runtime_error("responses directory has an unexpected entry");
    }

    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(args.model, &error);
    if (!reader)
      throw std::runtime_error("model metadata load failed: " + error);
    auto chat_template =
        gufo::tokenization::QwenChatTemplate::CreateFromGguf(*reader, &error);
    if (!chat_template ||
        chat_template->GetTemplateSha256() != config.chat_template_sha256)
      throw std::runtime_error("model chat template differs from package");
    auto tokenizer =
        gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
    if (!tokenizer)
      throw std::runtime_error("model tokenizer load failed: " + error);

    // The complete package is validated before HIP or durable output.  This
    // proves the fixed schedule and context envelope cannot change mid-run.
    std::unordered_set<std::string> ids;
    std::size_t preflight_records = 0;
    gufo::crypto::Sha256Hasher source_hasher;
    for (const auto& shard : package.shards) {
      const auto records = ReadPromptShard(
          package_directory / shard.prompt_file, shard, &ids, &source_hasher);
      for (const auto& record : records) {
        (void)RenderPrompt(*tokenizer, tokenizer->GetVocabSize(), record,
                           config);
        ++preflight_records;
      }
    }
    if (preflight_records != package.expected_records)
      throw std::runtime_error("package preflight record count differs");
    if (source_hasher.FinishHex() != package.source_sha256)
      throw std::runtime_error(
          "concatenated prompt shards differ from source identity");
    tokenizer.reset();
    reader.reset();

    if (const hipError_t status = hipInit(0); status != hipSuccess)
      throw std::runtime_error(std::string("HIP initialization failed: ") +
                               hipGetErrorString(status));
    qfn::ModelOptions options;
    options.max_context = config.max_context;
    auto model = qfn::Model::Load(args.model.string(), options, &error);
    if (!model)
      throw std::runtime_error("model load failed: " + error);
    if (model->ArtifactIdentityHex() != package.target_sha256)
      throw std::runtime_error("target artifact identity differs");

    std::vector<std::unique_ptr<qfn::Session>> sessions;
    sessions.reserve(package.decode_width);
    for (std::size_t i = 0; i < package.decode_width; ++i) {
      auto session = model->CreateSession(
          gufo::core::SessionMode::kAutoregressive, config.max_context, &error);
      if (!session)
        throw std::runtime_error("session creation failed: " + error);
      sessions.push_back(std::move(session));
    }

    std::size_t generated = 0;
    std::size_t resumed = 0;
    ids.clear();
    for (const auto& shard : package.shards) {
      const auto records = ReadPromptShard(
          package_directory / shard.prompt_file, shard, &ids, nullptr);
      const auto output = args.responses / shard.response_file;
      const auto partial = std::filesystem::path(output.string() + ".partial");
      if (std::filesystem::is_symlink(output) ||
          std::filesystem::is_symlink(partial) ||
          (std::filesystem::exists(output) && std::filesystem::exists(partial)))
        throw std::runtime_error("response shard state is ambiguous");
      if (!args.resume &&
          (std::filesystem::exists(output) || std::filesystem::exists(partial)))
        throw std::runtime_error("response shard exists; use --resume");

      const bool complete = std::filesystem::exists(output);
      const bool has_partial = std::filesystem::exists(partial);
      std::vector<std::string> existing;
      if (complete || has_partial) {
        existing =
            ReadResponseLines(complete ? output : partial, records.size());
        if (complete && existing.size() != records.size())
          throw std::runtime_error("published response shard is incomplete");
      }
      std::vector<PreparedRecord> prepared;
      prepared.reserve(records.size());
      for (std::size_t i = 0; i < records.size(); ++i) {
        PreparedRecord item{records[i],
                            RenderPrompt(*model, records[i], config)};
        if (i < existing.size())
          ValidateExistingLine(
              existing[i], item.source, item.prompt, *model, config,
              package.target_sha256, package.generator_sha256,
              package.generator_git_commit, package.source_sha256);
        prepared.push_back(std::move(item));
      }
      resumed += existing.size();
      if (complete)
        continue;

      DurableOutput sink(partial, has_partial);
      const std::size_t first_cohort =
          (existing.size() / package.decode_width) * package.decode_width;
      for (std::size_t start = first_cohort; start < prepared.size();
           start += package.decode_width) {
        const std::size_t end =
            std::min(prepared.size(), start + package.decode_width);
        std::vector<PreparedRecord> cohort(prepared.begin() + start,
                                           prepared.begin() + end);
        const auto lines =
            GenerateCohort(cohort, &sessions, config, package, &error);
        for (std::size_t i = 0; i < lines.size(); ++i) {
          const std::size_t local = start + i;
          if (local < existing.size()) {
            if (lines[i] != existing[local])
              throw std::runtime_error(
                  "resumed cohort differs from durable prefix");
            continue;
          }
          if (lines[i].size() > kMaxJsonLineBytes)
            throw std::runtime_error("response row exceeds 16 MiB");
          sink.Commit(lines[i]);
          ++generated;
          std::fprintf(stderr,
                       "FLASHMAP_V2_RESPONSE_COMMITTED ordinal=%zu "
                       "cohort_start=%zu decode_width=%zu\n",
                       shard.start + local, shard.start + start, cohort.size());
        }
      }
      sink.Close();
      PublishNoReplace(partial, output);
    }
    std::printf(
        "FLASHMAP_V2_RESPONSES_OK records=%zu resumed=%zu generated=%zu "
        "decode_width=%zu runtime=%s manifest=%s responses=%s\n",
        package.expected_records, resumed, generated, package.decode_width,
        package.runtime_identity.c_str(), args.manifest_sha256.c_str(),
        args.responses.string().c_str());
    return 0;
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "FLASHMAP_V2_RESPONSES_FAILED: %s\n",
                 exception.what());
    return 1;
  }
}
