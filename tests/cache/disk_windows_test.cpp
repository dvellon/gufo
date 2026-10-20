#include <fcntl.h>
#include <sys/random.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>

#include "src/cache/disk.hpp"

using namespace gufo::cache;
namespace fs = std::filesystem;
namespace {

struct Directory {
  fs::path path;
  Directory() {
    std::array<unsigned char, 8> nonce{};
    assert(getrandom(nonce.data(), nonce.size(), 0) == nonce.size());
    constexpr char hex[] = "0123456789abcdef";
    std::string suffix;
    for (const auto byte : nonce) {
      suffix += hex[byte >> 4];
      suffix += hex[byte & 15];
    }
    path = fs::temp_directory_path() / ("gufo-disk-windows-" + suffix);
    assert(fs::create_directory(path));
  }
  ~Directory() { fs::remove_all(path); }
};

DiskFileId Id(unsigned value) {
  DiskFileId id{};
  id[0] = value & 255;
  id[1] = (value >> 8) & 255;
  return id;
}

void Write(const fs::path& path, std::string_view bytes) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(bytes.data(), bytes.size());
  assert(stream.good());
}

std::string Read(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

void NoReplaceRename() {
  Directory directory;
  fs::create_directory(directory.path / "source");
  fs::create_directory(directory.path / "target");
  Write(directory.path / "source/item", "source");
  Write(directory.path / "target/item", "target");
  const int source = open((directory.path / "source").c_str(),
                          O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  const int target = open((directory.path / "target").c_str(),
                          O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  assert(source >= 0 && target >= 0);
  errno = 0;
  const int collision = gufo_renameat_noreplace(source, "item", target, "item");
  if (collision >= 0 || errno != EEXIST) {
    std::cerr << "no-replace collision result=" << collision
              << " errno=" << errno << '\n';
  }
  assert(collision < 0);
  assert(errno == EEXIST && Read(directory.path / "source/item") == "source" &&
         Read(directory.path / "target/item") == "target");
  fs::remove(directory.path / "target/item");
  errno = 0;
  const int moved = gufo_renameat_noreplace(source, "item", target, "item");
  if (moved != 0) {
    std::cerr << "no-replace move result=" << moved << " errno=" << errno
              << " source-exists=" << fs::exists(directory.path / "source/item")
              << " target-exists=" << fs::exists(directory.path / "target/item")
              << '\n';
  }
  assert(moved == 0);
  assert(!fs::exists(directory.path / "source/item") &&
         Read(directory.path / "target/item") == "source");
  assert(close(source) == 0 && close(target) == 0);
}

DiskManifest Manifest(std::span<const std::uint8_t> chunk,
                      std::span<const std::uint8_t> tail,
                      std::span<const std::uint8_t> state) {
  DiskManifest manifest;
  manifest.checkpoint = {7};
  manifest.lineage = {3};
  manifest.compatibility.fill(0x42);
  manifest.input = {1, 2, 3};
  manifest.tokens = {17, 18, 19};
  manifest.components.push_back(
      {{{1}, 4, ComponentKind::kAppendRows, 4, 2, 0},
       {{1}, 3},
       {{Id(1), chunk.size(), DiskChecksum(chunk)}},
       DiskPayload{Id(2), tail.size(), DiskChecksum(tail)},
       {}});
  manifest.components.push_back(
      {{{2}, 5, ComponentKind::kPrivateState, 0, 0, state.size()},
       {{2}, 3},
       {},
       {},
       DiskPayload{Id(3), state.size(), DiskChecksum(state)}});
  return manifest;
}

void StoreRoundTrip() {
  Directory directory;
  ResourceLedger ledger{{1ULL << 20, 1ULL << 20, 0, 0}};
  const std::array<std::uint8_t, 8> chunk{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<std::uint8_t, 4> tail{9, 10, 11, 12};
  const std::array<std::uint8_t, 16> state{13, 14, 15, 16};
  const auto manifest = Manifest(chunk, tail, state);
  const std::array<DiskWriteBuffer, 3> writes{{
      {false, Id(1), chunk},
      {true, Id(2), tail},
      {true, Id(3), state},
  }};
  {
    DiskStore store(ledger, directory.path, 1ULL << 20);
    store.Publish(Id(7), manifest, writes);
    bool locked = false;
    try {
      DiskStore second(ledger, directory.path, 1ULL << 20);
    } catch (const std::exception&) {
      locked = true;
    }
    assert(locked);
  }
  {
    DiskStore store(ledger, directory.path, 1ULL << 20);
    assert(store.Entries().size() == 1);
    auto snapshot = store.Open({7});
    assert(snapshot);
    const auto payload = *snapshot->Manifest().components[1].private_state;
    std::array<std::uint8_t, 7> staging{};
    std::array<std::uint8_t, state.size()> restored{};
    const auto stats = store.ReadPayload(
        *snapshot, true, payload, staging,
        [&](std::uint64_t offset, std::span<const std::uint8_t> bytes) {
          std::copy(bytes.begin(), bytes.end(), restored.begin() + offset);
        });
    assert(restored == state && stats.bytes == state.size());
    snapshot.reset();
    assert(store.Retire({7}));
    store.ReclaimOrphans();
    store.WaitForReclamation();
    assert(store.Entries().empty());
  }
  assert(ledger.Snapshot().total_bytes == 0);
}

}  // namespace

int main() {
  std::array<unsigned char, 32> first{}, second{};
  assert(getrandom(first.data(), first.size(), 0) == first.size());
  assert(getrandom(second.data(), second.size(), 0) == second.size());
  assert(first != second);
  NoReplaceRename();
  StoreRoundTrip();
  std::cout
      << "Windows RNG, no-replace rename, scan and disk round-trip passed\n";
}
