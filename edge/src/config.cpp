#include "config.hpp"

#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace vbw {
namespace {

namespace fs = std::filesystem;

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f'; }

// Full-token number parse with strtod semantics. std::from_chars handles the common case exactly (correctly
// rounded, locale-independent); strtod covers what from_chars rejects (a leading '+', hex floats, nan(...)).
bool parse_number(const std::string& tok, double& out) {
  if (tok.empty()) return false;
  const char* first = tok.data();
  const char* last = first + tok.size();
  auto [ptr, ec] = std::from_chars(first, last, out);
  if (ec == std::errc() && ptr == last) return true;
  // Out of range (from_chars leaves `out` untouched) or a form from_chars rejects: let strtod decide.
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(first, &end);
  if (end != last || end == first) return false;
  out = v;  // ERANGE still yields the IEEE value (+-HUGE_VAL or a denormal), which is what we want
  return true;
}

std::string where(const std::string& file, int line) {
  std::ostringstream os;
  os << "config " << file << ":" << line << ": ";
  return os.str();
}

std::string where_key(const Config& c, const std::string& key) {
  std::ostringstream os;
  os << "config " << c.file;
  auto it = c.lines.find(key);
  if (it != c.lines.end()) os << ":" << it->second;
  os << ": ";
  return os.str();
}

}  // namespace

Config Config::parse(const std::string& text, const std::string& path) {
  Config cfg;
  cfg.file = path;
  fs::path parent = fs::path(path).parent_path();
  cfg.base_dir = parent.empty() ? std::string(".") : parent.string();

  size_t pos = 0;
  int line_no = 0;
  // Skip a UTF-8 byte order mark.
  if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
      static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) {
    pos = 3;
  }
  std::vector<std::string> toks;
  while (pos <= text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    ++line_no;
    toks.clear();
    size_t i = pos;
    while (i < eol) {
      while (i < eol && is_space(text[i])) ++i;
      if (i >= eol) break;
      if (text[i] == '#') break;  // comment to end of line
      size_t j = i;
      while (j < eol && !is_space(text[j])) ++j;
      toks.emplace_back(text, i, j - i);
      i = j;
    }
    pos = eol + 1;
    if (toks.empty()) continue;

    const std::string& key = toks[0];
    double dummy = 0;
    if (parse_number(key, dummy)) {
      throw std::runtime_error(where(path, line_no) + "key '" + key +
                               "' is a number (missing key, or a value list wrapped onto a new line?)");
    }
    auto prev = cfg.lines.find(key);
    if (prev != cfg.lines.end()) {
      throw std::runtime_error(where(path, line_no) + "duplicate key '" + key + "' (first defined on line " +
                               std::to_string(prev->second) + ")");
    }

    std::vector<double> values;
    values.reserve(toks.size() - 1);
    bool numeric = true;
    for (size_t k = 1; k < toks.size(); ++k) {
      double v = 0;
      if (!parse_number(toks[k], v)) {
        numeric = false;
        break;
      }
      values.push_back(v);
    }
    cfg.lines.emplace(key, line_no);
    if (numeric) {
      cfg.nums.emplace(key, std::move(values));
    } else {
      cfg.strs.emplace(key, std::vector<std::string>(toks.begin() + 1, toks.end()));
    }
  }
  return cfg;
}

Config Config::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("config " + path + ": cannot open file");
  std::ostringstream ss;
  ss << in.rdbuf();
  if (in.bad()) throw std::runtime_error("config " + path + ": read error");
  return parse(ss.str(), path);
}

bool Config::has(const std::string& key) const { return nums.count(key) != 0 || strs.count(key) != 0; }

const std::vector<double>& Config::vec(const std::string& key) const {
  auto it = nums.find(key);
  if (it != nums.end()) return it->second;
  if (strs.count(key)) throw std::runtime_error(where_key(*this, key) + "key '" + key + "' is not numeric");
  throw std::runtime_error("config " + file + ": missing key '" + key + "'");
}

double Config::num(const std::string& key) const {
  const auto& v = vec(key);
  if (v.size() != 1) {
    throw std::runtime_error(where_key(*this, key) + "key '" + key + "' has " + std::to_string(v.size()) +
                             " values, expected exactly 1");
  }
  return v[0];
}

double Config::num_or(const std::string& key, double fallback) const {
  if (!has(key)) return fallback;
  return num(key);
}

const std::vector<std::string>& Config::words(const std::string& key) const {
  auto it = strs.find(key);
  if (it != strs.end()) return it->second;
  if (nums.count(key)) {
    throw std::runtime_error(where_key(*this, key) + "key '" + key + "' is numeric, expected words");
  }
  throw std::runtime_error("config " + file + ": missing key '" + key + "'");
}

std::string Config::path(const std::string& key) const {
  const auto& w = words(key);
  fs::path p(w.at(0));
  if (p.is_absolute()) return p.lexically_normal().string();
  return (fs::path(base_dir) / p).lexically_normal().string();
}

std::vector<std::string> Config::keys_with_prefix(const std::string& prefix) const {
  std::vector<std::string> out;
  auto a = nums.lower_bound(prefix);
  auto b = strs.lower_bound(prefix);
  auto starts = [&](const std::string& k) { return k.compare(0, prefix.size(), prefix) == 0; };
  // Merge the two sorted ranges.
  while (true) {
    const bool a_ok = a != nums.end() && starts(a->first);
    const bool b_ok = b != strs.end() && starts(b->first);
    if (!a_ok && !b_ok) break;
    if (a_ok && (!b_ok || a->first < b->first)) {
      out.push_back(a->first);
      ++a;
    } else {
      out.push_back(b->first);
      ++b;
    }
  }
  return out;
}

}  // namespace vbw
