#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <cstring>

static std::vector<std::string_view> split(std::string_view sv, char delim) {
    std::vector<std::string_view> parts;
    size_t pos = 0;
    while (pos < sv.size()) {
        size_t next = sv.find(delim, pos);
        if (next == std::string_view::npos) {
            parts.push_back(sv.substr(pos));
            break;
        }
        parts.push_back(sv.substr(pos, next - pos));
        pos = next + 1;
    }
    return parts;
}

static bool starts_with(std::string_view sv, std::string_view prefix) {
    return sv.size() >= prefix.size() && sv.substr(0, prefix.size()) == prefix;
}

static bool ends_with(std::string_view sv, std::string_view suffix) {
    return sv.size() >= suffix.size() &&
           sv.substr(sv.size() - suffix.size()) == suffix;
}

static void print(std::string_view sv) {
    std::cout << "\"" << sv << "\" (len=" << sv.size() << ")\n";
}

int main() {
    const char* raw = "hello world from C++17";

    std::string_view sv1 = raw;
    print(sv1);

    std::string_view sv2 = "hello world from C++17";
    print(sv2);

    std::string s = "hello world from C++17";
    std::string_view sv3 = s;
    print(sv3);

    std::string_view sv4 = sv1.substr(0, 5);
    print(sv4);

    std::string_view sv5 = sv1.substr(6, 5);
    print(sv5);

    std::string_view sv6 = "  trim me  ";
    sv6.remove_prefix(2);
    sv6.remove_suffix(2);
    print(sv6);

    std::cout << "\nsplit(\"" << sv1 << "\", ' '):\n";
    for (auto part : split(sv1, ' ')) {
        print(part);
    }

    std::cout << "\n";
    std::cout << "starts_with(\"hello world\", \"hello\"): "
              << starts_with("hello world", "hello") << "\n";
    std::cout << "ends_with(\"hello world\", \"world\"): "
              << ends_with("hello world", "world") << "\n";
    std::cout << "starts_with(\"hello world\", \"world\"): "
              << starts_with("hello world", "world") << "\n";

    std::cout << "\nzero-allocation split on string literal:\n";
    std::string_view csv = "a,bb,ccc,dddd";
    for (auto part : split(csv, ',')) {
        print(part);
    }

    std::cout << "\ntrimming:\n";
    const char* spaced = "   spaced   ";
    std::string_view sv7 = spaced;
    auto start = sv7.find_first_not_of(' ');
    auto end = sv7.find_last_not_of(' ');
    if (start != std::string_view::npos && end != std::string_view::npos) {
        std::string_view trimmed = sv7.substr(start, end - start + 1);
        print(trimmed);
    }
}
