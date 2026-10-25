// Edge config file (edge/config/<schema>.cfg), written by scripts/export_edge_config.py.
//
// Grammar (SPEC section 2): one entry per line, `key value value ...`. A `#` at the start of a token begins a
// comment that runs to the end of the line; blank lines are ignored; CRLF and a UTF-8 BOM are tolerated. A line
// whose values all parse as numbers (strtod rules: decimal, exponent, hex, `nan`, `inf`, `-inf`, `infinity`) is
// numeric and goes to `nums`; otherwise all of its values are kept as strings in `strs`. A key with no values is
// numeric with an empty vector. Keys are unique. Paths are relative to the config file's directory.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace vbw {

struct Config {
  std::map<std::string, std::vector<double>> nums;
  std::map<std::string, std::vector<std::string>> strs;
  std::string base_dir;                                      // directory of the config file
  std::string file;                                          // the path the config was loaded from (for errors)
  std::map<std::string, int> lines;                          // key -> 1-based line number (for errors)

  static Config load(const std::string& path);               // throws std::runtime_error with line number
  // Parses config text as if it were read from `path` (base_dir = directory of `path`). Used by load().
  static Config parse(const std::string& text, const std::string& path);

  bool has(const std::string& key) const;
  const std::vector<double>& vec(const std::string& key) const;          // throws if missing / not numeric
  double num(const std::string& key) const;                              // exactly one value
  double num_or(const std::string& key, double fallback) const;
  const std::vector<std::string>& words(const std::string& key) const;   // throws if missing / numeric
  std::string path(const std::string& key) const;            // words(key)[0] resolved against base_dir
  std::vector<std::string> keys_with_prefix(const std::string& prefix) const;   // sorted, nums and strs
};

}  // namespace vbw
