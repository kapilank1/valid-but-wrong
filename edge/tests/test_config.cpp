// Config parser tests: grammar (SPEC section 2), accessors, errors, path resolution, and the real falcon9.cfg.
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS  // std::getenv
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"

using namespace vbw;
namespace fs = std::filesystem;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    ++g_checks;                                                                  \
    if (!(cond)) {                                                               \
      ++g_failures;                                                              \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
    }                                                                            \
  } while (0)

// Runs fn, expecting std::runtime_error whose message contains every needle.
template <class F>
static bool throws_with(F fn, std::initializer_list<const char*> needles) {
  try {
    fn();
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    for (const char* n : needles) {
      if (msg.find(n) == std::string::npos) {
        std::fprintf(stderr, "  error message '%s' lacks '%s'\n", msg.c_str(), n);
        return false;
      }
    }
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

static fs::path write_temp(const std::string& name, const std::string& text) {
  fs::path dir = fs::temp_directory_path() / "vbw_test_config";
  fs::create_directories(dir);
  fs::path p = dir / name;
  std::ofstream out(p, std::ios::binary);
  out << text;
  return p;
}

static bool same_file(const std::string& a, const fs::path& b) {
  return fs::path(a).lexically_normal() == b.lexically_normal();
}

static void test_grammar() {
  const std::string text =
      "\xEF\xBB\xBF# leading comment with a BOM before it\r\n"
      "schema falcon9\r\n"
      "schema_id 1   # trailing comment\n"
      "rate_hz 10.0\n"
      "\n"
      "   \t  \n"
      "channels speed altitude vz vx downrange accel angle q\n"
      "l1.threshold.b 1e-3\n"
      "l1.threshold.a 0.5\n"
      "l1.thresholdx 7\n"
      "l1.params.x nan inf -inf 1.5 -0.0 infinity\n"
      "\tindented   2   3  \n"
      "mixed 1 abc\n"
      "hexval 0x10\n"
      "plus +2.5\n"
      "empty.key\n"
      "tight#comment right after a token is part of the token\n"
      "l2.onnx ../../models/l2_falcon9.onnx\n"
      "model.abs " +
#ifdef _WIN32
      std::string("C:\\abs\\model.onnx") +
#else
      std::string("/abs/model.onnx") +
#endif
      "\n"
      "# final comment without newline";
  fs::path p = write_temp("grammar.cfg", text);
  Config c = Config::load(p.string());

  CHECK(same_file(c.base_dir, p.parent_path()));
  CHECK(c.words("schema").size() == 1 && c.words("schema")[0] == "falcon9");
  CHECK(c.num("schema_id") == 1.0);
  CHECK(c.num("rate_hz") == 10.0);
  CHECK(c.words("channels").size() == 8 && c.words("channels")[7] == "q");
  CHECK(c.num("l1.threshold.b") == 1e-3);
  CHECK(c.num("l1.threshold.a") == 0.5);

  const auto& x = c.vec("l1.params.x");
  CHECK(x.size() == 6);
  if (x.size() == 6) {
    CHECK(std::isnan(x[0]));
    CHECK(std::isinf(x[1]) && x[1] > 0);
    CHECK(std::isinf(x[2]) && x[2] < 0);
    CHECK(x[3] == 1.5);
    CHECK(x[4] == 0.0 && std::signbit(x[4]));
    CHECK(std::isinf(x[5]) && x[5] > 0);
  }
  CHECK(c.vec("indented").size() == 2 && c.vec("indented")[1] == 3.0);
  // A line with any non-number is all strings.
  CHECK(c.words("mixed").size() == 2 && c.words("mixed")[0] == "1" && c.words("mixed")[1] == "abc");
  CHECK(!c.nums.count("mixed"));
  CHECK(c.num("hexval") == 16.0);   // strtod hex
  CHECK(c.num("plus") == 2.5);
  CHECK(c.has("empty.key") && c.vec("empty.key").empty());
  CHECK(c.has("tight#comment"));
  CHECK(c.words("tight#comment").size() == 9);
  CHECK(!c.has("final"));
  CHECK(c.lines.at("schema") == 2 && c.lines.at("schema_id") == 3 && c.lines.at("channels") == 7);

  // Exact round trip of Python repr() floats.
  Config r = Config::parse("a 0.1 122.85879959105988 2.363173197256789 1e-06 5e-324 1.7976931348623157e+308\n", "x.cfg");
  const auto& a = r.vec("a");
  CHECK(a.size() == 6);
  if (a.size() == 6) {
    CHECK(a[0] == 0.1);
    CHECK(a[1] == 122.85879959105988);
    CHECK(a[2] == 2.363173197256789);
    CHECK(a[3] == 1e-06);
    CHECK(a[4] == 5e-324);
    CHECK(a[5] == 1.7976931348623157e+308);
  }

  // num_or
  CHECK(c.num_or("missing.key", 42.0) == 42.0);
  CHECK(c.num_or("rate_hz", 42.0) == 10.0);

  // keys_with_prefix: sorted, nums and strs merged, prefix is literal ("l1.thresholdx" excluded).
  auto keys = c.keys_with_prefix("l1.threshold.");
  CHECK(keys.size() == 2 && keys[0] == "l1.threshold.a" && keys[1] == "l1.threshold.b");
  auto sk = c.keys_with_prefix("s");
  CHECK(sk.size() == 2 && sk[0] == "schema" && sk[1] == "schema_id");
  CHECK(c.keys_with_prefix("zzz").empty());
  CHECK(c.keys_with_prefix("").size() == c.nums.size() + c.strs.size());

  // Paths resolve against the config file's directory; absolute paths are kept.
  CHECK(same_file(c.path("l2.onnx"), p.parent_path() / ".." / ".." / "models" / "l2_falcon9.onnx"));
  CHECK(fs::path(c.path("model.abs")).is_absolute());
  Config cwd = Config::parse("m model.onnx\n", "falcon9.cfg");   // config in the current directory
  CHECK(cwd.base_dir == ".");
  CHECK(same_file(cwd.path("m"), fs::path("model.onnx")));

  // Accessor errors name the key (and the line when the key exists).
  CHECK(throws_with([&] { c.vec("nope"); }, {"missing key 'nope'", "grammar.cfg"}));
  CHECK(throws_with([&] { c.num("nope"); }, {"missing key 'nope'"}));
  CHECK(throws_with([&] { c.words("nope"); }, {"missing key 'nope'"}));
  CHECK(throws_with([&] { c.path("nope"); }, {"missing key 'nope'"}));
  CHECK(throws_with([&] { c.vec("schema"); }, {"'schema'", "not numeric", ":2"}));
  CHECK(throws_with([&] { c.words("rate_hz"); }, {"'rate_hz'", "numeric", ":4"}));
  CHECK(throws_with([&] { c.num("l1.params.x"); }, {"'l1.params.x'", "6 values", "exactly 1"}));
  CHECK(throws_with([&] { c.num("empty.key"); }, {"0 values"}));
  CHECK(throws_with([&] { c.num_or("l1.params.x", 1.0); }, {"6 values"}));
}

static void test_errors() {
  CHECK(throws_with([] { Config::parse("a 1\nb 2\n\na 3\n", "dup.cfg"); },
                    {"dup.cfg:4", "duplicate key 'a'", "line 1"}));
  // A duplicate across numeric and string lines is still a duplicate.
  CHECK(throws_with([] { Config::parse("a 1\na x\n", "dup2.cfg"); }, {"dup2.cfg:2", "duplicate key 'a'"}));
  CHECK(throws_with([] { Config::parse("ok 1\n# c\n  3.5 4\n", "num.cfg"); }, {"num.cfg:3", "'3.5'", "number"}));
  CHECK(throws_with([] { Config::load("definitely/not/here.cfg"); }, {"definitely/not/here.cfg", "cannot open"}));
  // Empty and comment-only files are valid and empty.
  Config e = Config::parse("", "e.cfg");
  CHECK(e.nums.empty() && e.strs.empty());
  Config e2 = Config::parse("# only\n\n   # comments\n", "e2.cfg");
  CHECK(e2.nums.empty() && e2.strs.empty());
}

static fs::path find_repo_file(const std::string& rel) {
  if (fs::exists(rel)) return fs::path(rel);
  if (const char* root = std::getenv("VBW_REPO_ROOT")) {
    fs::path p = fs::path(root) / rel;
    if (fs::exists(p)) return p;
  }
#ifdef VBW_REPO_ROOT
  {
    fs::path p = fs::path(VBW_REPO_ROOT) / rel;
    if (fs::exists(p)) return p;
  }
#endif
  return {};
}

static void test_falcon9_cfg() {
  fs::path p = find_repo_file("edge/config/falcon9.cfg");
  if (p.empty()) {
    std::printf("test_config: SKIP falcon9.cfg (edge/config/falcon9.cfg not found)\n");
    return;
  }
  try {
    Config c = Config::load(p.string());
    CHECK(c.words("schema").at(0) == "falcon9");
    CHECK(c.num("schema_id") == 1.0);
    CHECK(c.num("rate_hz") > 0);
    const auto& ch = c.words("channels");
    CHECK(ch.size() == 8);
    if (ch.size() == 8) CHECK(ch[0] == "speed" && ch[1] == "altitude" && ch[7] == "q");
    for (const char* k : {"l2.mean", "l2.std", "l2.dstd", "l2.res_median", "l2.res_mad"}) {
      CHECK(c.vec(k).size() == 8);
    }
    CHECK(c.num("l2.window") == 20.0);
    const auto& rules = c.words("l1.rules");
    CHECK(!rules.empty());
    for (const auto& r : rules) {
      CHECK(c.has("l1.threshold." + r));
      CHECK(c.has("l1.lookback." + r));
      if (c.has("l1.threshold." + r)) CHECK(c.num("l1.threshold." + r) > 0);
    }
    CHECK(c.keys_with_prefix("l1.threshold.").size() == rules.size());
    CHECK(c.num("alert.tau") > 0 && c.num("alert.tau") < 1);
    CHECK(c.num("alert.k") >= 1 && c.num("alert.k") <= c.num("alert.n"));
    if (c.has("l2.onnx")) {
      const std::string model = c.path("l2.onnx");
      CHECK(fs::exists(model));
      std::printf("test_config: falcon9.cfg ok (%zu rules, l2.onnx -> %s%s)\n", rules.size(), model.c_str(),
                  fs::exists(model) ? "" : " MISSING");
    } else {
      std::printf("test_config: falcon9.cfg ok (%zu rules, no l2.onnx)\n", rules.size());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL falcon9.cfg: %s\n", e.what());
    ++g_failures;
  }
}

int main() {
  try {
    test_grammar();
    test_errors();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL unexpected exception: %s\n", e.what());
    ++g_failures;
  }
  test_falcon9_cfg();
  if (g_failures) {
    std::fprintf(stderr, "test_config: %d of %d checks FAILED\n", g_failures, g_checks);
    return 1;
  }
  std::printf("test_config: all %d checks passed\n", g_checks);
  return 0;
}
