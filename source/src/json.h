// Minimal JSON reader/writer for the engine protocol and safetensors headers.
#pragma once
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace json {
struct Value {
  enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;
  const Value& operator[](const std::string& k) const {
    static const Value null;
    for (auto& kv : obj) if (kv.first == k) return kv.second;
    return null;
  }
};

struct Parser {
  const char* p; const char* e;
  void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
  [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json: ") + m); }
  std::string string() {
    if (*p != '"') fail("expected string");
    p++;
    std::string s;
    while (p < e && *p != '"') {
      if (*p == '\\') {
        p++;
        if (p >= e) fail("bad escape");
        char c = *p++;
        switch (c) {
          case 'n': s += '\n'; break; case 't': s += '\t'; break; case 'r': s += '\r'; break;
          case 'b': s += '\b'; break; case 'f': s += '\f'; break;
          case 'u': {
            if (e - p < 4) fail("bad \\u");
            unsigned cp = strtoul(std::string(p, 4).c_str(), nullptr, 16); p += 4;
            if (cp < 0x80) s += (char)cp;
            else if (cp < 0x800) { s += (char)(0xc0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3f)); }
            else { s += (char)(0xe0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3f)); s += (char)(0x80 | (cp & 0x3f)); }
            break;
          }
          default: s += c;
        }
      } else s += *p++;
    }
    if (p >= e) fail("unterminated string");
    p++;
    return s;
  }
  Value value() {
    ws();
    if (p >= e) fail("unexpected end");
    Value v;
    char c = *p;
    if (c == '{') {
      v.kind = Value::Obj; p++; ws();
      if (*p == '}') { p++; return v; }
      while (true) {
        ws(); std::string k = string(); ws();
        if (*p != ':') fail("expected :");
        p++;
        v.obj.emplace_back(k, value());
        ws();
        if (*p == ',') { p++; continue; }
        if (*p == '}') { p++; return v; }
        fail("expected , or }");
      }
    } else if (c == '[') {
      v.kind = Value::Arr; p++; ws();
      if (*p == ']') { p++; return v; }
      while (true) {
        v.arr.push_back(value());
        ws();
        if (*p == ',') { p++; continue; }
        if (*p == ']') { p++; return v; }
        fail("expected , or ]");
      }
    } else if (c == '"') {
      v.kind = Value::Str; v.str = string();
    } else if (c == 't' && e - p >= 4 && !strncmp(p, "true", 4)) { v.kind = Value::Bool; v.b = true; p += 4; }
    else if (c == 'f' && e - p >= 5 && !strncmp(p, "false", 5)) { v.kind = Value::Bool; v.b = false; p += 5; }
    else if (c == 'n' && e - p >= 4 && !strncmp(p, "null", 4)) { p += 4; }
    else {
      char* end;
      v.num = strtod(p, &end);
      if (end == p) fail("bad value");
      v.kind = Value::Num; p = end;
    }
    return v;
  }
};

inline Value parse(const std::string& s) {
  Parser ps{s.data(), s.data() + s.size()};
  return ps.value();
}

inline std::string quote(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n";
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char)c;
  }
  return o + "\"";
}

inline std::string dump(const Value& v) {
  switch (v.kind) {
    case Value::Null: return "null";
    case Value::Bool: return v.b ? "true" : "false";
    case Value::Num: {
      char b[64];
      if (v.num == (double)(long long)v.num) snprintf(b, sizeof b, "%lld", (long long)v.num);
      else snprintf(b, sizeof b, "%.17g", v.num);
      return b;
    }
    case Value::Str: return quote(v.str);
    case Value::Arr: { std::string s = "["; for (size_t i = 0; i < v.arr.size(); i++) { if (i) s += ","; s += dump(v.arr[i]); } return s + "]"; }
    case Value::Obj: { std::string s = "{"; for (size_t i = 0; i < v.obj.size(); i++) { if (i) s += ","; s += quote(v.obj[i].first) + ":" + dump(v.obj[i].second); } return s + "}"; }
  }
  return "null";
}
}  // namespace json
