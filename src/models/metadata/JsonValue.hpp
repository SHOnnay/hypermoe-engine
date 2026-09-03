#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace hypermoe::models::metadata {

class MetadataError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct JsonNumber {
    std::string text;
};

class JsonValue {
public:
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue, std::less<>>;
    using Storage = std::variant<std::monostate, bool, JsonNumber, std::string,
                                 Array, Object>;

    JsonValue() = default;
    explicit JsonValue(bool value);
    explicit JsonValue(JsonNumber value);
    explicit JsonValue(std::string value);
    explicit JsonValue(Array value);
    explicit JsonValue(Object value);

    [[nodiscard]] bool isNull() const noexcept;
    [[nodiscard]] bool isBool() const noexcept;
    [[nodiscard]] bool isNumber() const noexcept;
    [[nodiscard]] bool isString() const noexcept;
    [[nodiscard]] bool isArray() const noexcept;
    [[nodiscard]] bool isObject() const noexcept;

    [[nodiscard]] bool asBool() const;
    [[nodiscard]] double asDouble() const;
    [[nodiscard]] std::uint64_t asUInt64() const;
    [[nodiscard]] const std::string& asString() const;
    [[nodiscard]] const Array& asArray() const;
    [[nodiscard]] const Object& asObject() const;
    [[nodiscard]] const JsonValue& require(std::string_view key) const;
    [[nodiscard]] const JsonValue* find(std::string_view key) const noexcept;

private:
    Storage value_;
};

[[nodiscard]] JsonValue parseJson(std::string_view text);
[[nodiscard]] JsonValue parseJsonFile(const std::filesystem::path& path,
                                      std::uintmax_t maximumBytes =
                                          64ULL * 1024ULL * 1024ULL);

} // namespace hypermoe::models::metadata
