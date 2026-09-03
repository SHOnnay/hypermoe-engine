#include "models/metadata/JsonValue.hpp"

#include <charconv>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <utility>

namespace hypermoe::models::metadata {
namespace {

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    JsonValue parse() {
        auto result = parseValue(0);
        skipWhitespace();
        if (position_ != text_.size()) fail("unexpected trailing input");
        return result;
    }

private:
    [[noreturn]] void fail(std::string_view message) const {
        throw MetadataError(std::string(message) + " at byte " +
                            std::to_string(position_));
    }

    void skipWhitespace() {
        while (position_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[position_])) != 0) {
            ++position_;
        }
    }

    bool consume(char expected) {
        skipWhitespace();
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    JsonValue parseValue(std::size_t depth) {
        if (depth > 128) fail("JSON nesting limit exceeded");
        skipWhitespace();
        if (position_ >= text_.size()) fail("unexpected end of input");
        switch (text_[position_]) {
        case 'n': parseLiteral("null"); return JsonValue{};
        case 't': parseLiteral("true"); return JsonValue(true);
        case 'f': parseLiteral("false"); return JsonValue(false);
        case '"': return JsonValue(parseString());
        case '[': return JsonValue(parseArray(depth + 1));
        case '{': return JsonValue(parseObject(depth + 1));
        default:
            if (text_[position_] == '-' ||
                std::isdigit(static_cast<unsigned char>(text_[position_])) != 0) {
                return JsonValue(JsonNumber{parseNumber()});
            }
            fail("unexpected JSON token");
        }
    }

    void parseLiteral(std::string_view literal) {
        if (text_.substr(position_, literal.size()) != literal) {
            fail("invalid JSON literal");
        }
        position_ += literal.size();
    }

    static void appendUtf8(std::string& output, std::uint32_t codePoint) {
        if (codePoint <= 0x7FU) {
            output.push_back(static_cast<char>(codePoint));
        } else if (codePoint <= 0x7FFU) {
            output.push_back(static_cast<char>(0xC0U | (codePoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
        } else if (codePoint <= 0xFFFFU) {
            output.push_back(static_cast<char>(0xE0U | (codePoint >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((codePoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
        } else {
            output.push_back(static_cast<char>(0xF0U | (codePoint >> 18U)));
            output.push_back(static_cast<char>(0x80U | ((codePoint >> 12U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | ((codePoint >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (codePoint & 0x3FU)));
        }
    }

    std::uint32_t parseHexQuad() {
        if (text_.size() - position_ < 4) fail("truncated Unicode escape");
        std::uint32_t value = 0;
        for (int count = 0; count < 4; ++count) {
            const char character = text_[position_++];
            value <<= 4U;
            if (character >= '0' && character <= '9') {
                value |= static_cast<std::uint32_t>(character - '0');
            } else if (character >= 'a' && character <= 'f') {
                value |= static_cast<std::uint32_t>(character - 'a' + 10);
            } else if (character >= 'A' && character <= 'F') {
                value |= static_cast<std::uint32_t>(character - 'A' + 10);
            }
            else fail("invalid Unicode escape");
        }
        return value;
    }

    std::string parseString() {
        if (!consume('"')) fail("expected string");
        std::string output;
        while (position_ < text_.size()) {
            const char character = text_[position_++];
            if (character == '"') return output;
            if (static_cast<unsigned char>(character) < 0x20U) {
                fail("control character in string");
            }
            if (character != '\\') {
                output.push_back(character);
                continue;
            }
            if (position_ >= text_.size()) fail("truncated string escape");
            switch (text_[position_++]) {
            case '"': output.push_back('"'); break;
            case '\\': output.push_back('\\'); break;
            case '/': output.push_back('/'); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            case 'u': {
                auto codePoint = parseHexQuad();
                if (codePoint >= 0xD800U && codePoint <= 0xDBFFU) {
                    if (text_.size() - position_ < 2 || text_[position_] != '\\' ||
                        text_[position_ + 1] != 'u') {
                        fail("missing low Unicode surrogate");
                    }
                    position_ += 2;
                    const auto low = parseHexQuad();
                    if (low < 0xDC00U || low > 0xDFFFU) {
                        fail("invalid low Unicode surrogate");
                    }
                    codePoint = 0x10000U + ((codePoint - 0xD800U) << 10U) +
                                (low - 0xDC00U);
                } else if (codePoint >= 0xDC00U && codePoint <= 0xDFFFU) {
                    fail("unexpected low Unicode surrogate");
                }
                appendUtf8(output, codePoint);
                break;
            }
            default: fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }

    std::string parseNumber() {
        const auto start = position_;
        if (text_[position_] == '-') ++position_;
        if (position_ >= text_.size()) fail("truncated number");
        if (text_[position_] == '0') {
            ++position_;
        } else {
            if (text_[position_] < '1' || text_[position_] > '9') fail("invalid number");
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') ++position_;
        }
        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            const auto fractionStart = position_;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') ++position_;
            if (position_ == fractionStart) fail("invalid number fraction");
        }
        if (position_ < text_.size() &&
            (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() &&
                (text_[position_] == '+' || text_[position_] == '-')) ++position_;
            const auto exponentStart = position_;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') ++position_;
            if (position_ == exponentStart) fail("invalid number exponent");
        }
        return std::string(text_.substr(start, position_ - start));
    }

    JsonValue::Array parseArray(std::size_t depth) {
        if (!consume('[')) fail("expected array");
        JsonValue::Array array;
        if (consume(']')) return array;
        while (true) {
            array.push_back(parseValue(depth));
            if (consume(']')) return array;
            if (!consume(',')) fail("expected comma in array");
        }
    }

    JsonValue::Object parseObject(std::size_t depth) {
        if (!consume('{')) fail("expected object");
        JsonValue::Object object;
        if (consume('}')) return object;
        while (true) {
            skipWhitespace();
            if (position_ >= text_.size() || text_[position_] != '"') {
                fail("expected object key");
            }
            auto key = parseString();
            if (!consume(':')) fail("expected colon after object key");
            auto [entry, inserted] = object.emplace(std::move(key), parseValue(depth));
            (void)entry;
            if (!inserted) fail("duplicate object key");
            if (consume('}')) return object;
            if (!consume(',')) fail("expected comma in object");
        }
    }

    std::string_view text_;
    std::size_t position_{};
};

template <typename T>
const T& requireType(const JsonValue::Storage& value, std::string_view name) {
    const auto* typed = std::get_if<T>(&value);
    if (typed == nullptr) throw MetadataError("JSON value is not " + std::string(name));
    return *typed;
}

} // namespace

JsonValue::JsonValue(bool value) : value_(value) {}
JsonValue::JsonValue(JsonNumber value) : value_(std::move(value)) {}
JsonValue::JsonValue(std::string value) : value_(std::move(value)) {}
JsonValue::JsonValue(Array value) : value_(std::move(value)) {}
JsonValue::JsonValue(Object value) : value_(std::move(value)) {}
bool JsonValue::isNull() const noexcept { return std::holds_alternative<std::monostate>(value_); }
bool JsonValue::isBool() const noexcept { return std::holds_alternative<bool>(value_); }
bool JsonValue::isNumber() const noexcept { return std::holds_alternative<JsonNumber>(value_); }
bool JsonValue::isString() const noexcept { return std::holds_alternative<std::string>(value_); }
bool JsonValue::isArray() const noexcept { return std::holds_alternative<Array>(value_); }
bool JsonValue::isObject() const noexcept { return std::holds_alternative<Object>(value_); }
bool JsonValue::asBool() const { return requireType<bool>(value_, "a boolean"); }

double JsonValue::asDouble() const {
    const auto& text = requireType<JsonNumber>(value_, "a number").text;
    double result{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw MetadataError("JSON number is outside double range");
    }
    return result;
}

std::uint64_t JsonValue::asUInt64() const {
    const auto& text = requireType<JsonNumber>(value_, "an unsigned integer").text;
    std::uint64_t result{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw MetadataError("JSON value is not an unsigned 64-bit integer");
    }
    return result;
}

const std::string& JsonValue::asString() const {
    return requireType<std::string>(value_, "a string");
}
const JsonValue::Array& JsonValue::asArray() const {
    return requireType<Array>(value_, "an array");
}
const JsonValue::Object& JsonValue::asObject() const {
    return requireType<Object>(value_, "an object");
}
const JsonValue& JsonValue::require(std::string_view key) const {
    const auto& object = asObject();
    const auto found = object.find(key);
    if (found == object.end()) throw MetadataError("missing JSON field: " + std::string(key));
    return found->second;
}
const JsonValue* JsonValue::find(std::string_view key) const noexcept {
    const auto* object = std::get_if<Object>(&value_);
    if (object == nullptr) return nullptr;
    const auto found = object->find(key);
    return found == object->end() ? nullptr : &found->second;
}

JsonValue parseJson(std::string_view text) { return Parser(text).parse(); }

JsonValue parseJsonFile(const std::filesystem::path& path,
                        std::uintmax_t maximumBytes) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) throw MetadataError("cannot stat model manifest: " + error.message());
    if (size > maximumBytes) throw MetadataError("model manifest exceeds size limit");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw MetadataError("cannot open model manifest");
    std::string contents((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
    if (input.bad()) throw MetadataError("failed reading model manifest");
    if (contents.size() > maximumBytes) {
        throw MetadataError("model manifest exceeded size limit while reading");
    }
    return parseJson(contents);
}

} // namespace hypermoe::models::metadata
