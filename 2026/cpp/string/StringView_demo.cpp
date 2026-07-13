#include "StringView.hpp"
#include <iostream>
#include <vector>

int main() {
    const char* raw = "hello world from C++17";

    StringView sv1 = raw;
    std::cout << "sv1: " << sv1 << " | size=" << sv1.size() << '\n';

    StringView sv2("hello world from C++17");
    std::cout << "sv2: " << sv2 << " | size=" << sv2.size() << '\n';

    std::string s = "hello world from C++17";
    StringView sv3 = s;
    std::cout << "sv3: " << sv3 << " | size=" << sv3.size() << '\n';

    StringView sv4 = sv1.substr(0, 5);
    std::cout << "sv4 (0,5): " << sv4 << '\n';

    StringView sv5 = sv1.substr(6, 5);
    std::cout << "sv5 (6,5): " << sv5 << '\n';

    StringView sv6("  trim me  ");
    sv6.remove_prefix(2);
    sv6.remove_suffix(2);
    std::cout << "sv6 trimmed: " << sv6 << " | size=" << sv6.size() << '\n';

    std::cout << "\n== Iterators ==\n";
    for (auto it = sv1.begin(); it != sv1.end(); ++it) {
        std::cout << *it;
    }
    std::cout << '\n';

    for (char ch : sv1) {
        putchar(ch);
    }
    std::cout << "\n\n";

    std::cout << "== find ==\n";
    std::cout << "find('w'): " << sv1.find('w') << '\n';
    std::cout << "find('z'): " << sv1.find('z') << " (npos=" << StringView::npos << ")\n";
    std::cout << "find(\"world\"): " << sv1.find(StringView("world")) << '\n';

    std::cout << "\n== trimming ==\n";
    StringView csv("   spaced   ");
    auto start = csv.find_first_not_of(' ');
    auto end   = csv.find_last_not_of(' ');
    if (start != StringView::npos && end != StringView::npos) {
        StringView trimmed = csv.substr(start, end - start + 1);
        std::cout << '"' << trimmed << '"' << '\n';
    }

    std::cout << "\n== split (zero copy) ==\n";
    auto split = [](StringView sv, char delim) {
        std::vector<StringView> parts;
        size_t pos = 0;
        while (pos < sv.size()) {
            size_t next = sv.find(delim, pos);
            if (next == StringView::npos) {
                parts.push_back(sv.substr(pos));
                break;
            }
            parts.push_back(sv.substr(pos, next - pos));
            pos = next + 1;
        }
        return parts;
    };

    StringView csv2("a,bb,ccc,dddd");
    for (auto part : split(csv2, ',')) {
        std::cout << '[' << part << "] len=" << part.size() << '\n';
    }

    std::cout << "\n== operators ==\n";
    StringView a("apple"), b("banana"), c("apple");
    std::cout << std::boolalpha;
    std::cout << "a==c: " << (a == c) << '\n';
    std::cout << "a!=b: " << (a != b) << '\n';
    std::cout << "a<b:  " << (a < b) << '\n';
    std::cout << "b<a:  " << (b < a) << '\n';

    std::cout << "\n== at with bounds check ==\n";
    try {
        std::cout << "sv4.at(10) = ";
        std::cout << sv4.at(10) << '\n';
    } catch (const std::out_of_range& e) {
        std::cout << "caught: " << e.what() << '\n';
    }

    std::cout << "\n== default construction ==\n";
    StringView empty;
    std::cout << "empty: size=" << empty.size() << " empty=" << empty.empty() << '\n';

    std::cout << "\n== swap ==\n";
    std::cout << "before: a=" << a << " b=" << b << '\n';
    a.swap(b);
    std::cout << "after:  a=" << a << " b=" << b << '\n';

    std::cout << "\n== copy ==\n";
    StringView orig("original");
    StringView copied(orig);
    std::cout << "copy ctor: orig=" << orig << ", copied=" << copied << "\n";
    std::cout << "  same data pointer? " << (orig.data() == copied.data()) << "\n";

    StringView assigned("old");
    std::cout << "before copy assign: " << assigned << "\n";
    assigned = orig;
    std::cout << "after copy assign:  " << assigned << "\n";
    std::cout << "  same data pointer? " << (orig.data() == assigned.data()) << "\n";

    std::cout << "\n== copy edge cases ==\n";
    StringView empty_view;
    StringView empty_copied(empty_view);
    std::cout << "copy empty: size=" << empty_copied.size() << " empty=" << empty_copied.empty() << "\n";

    StringView hello("hello");
    hello = hello;
    std::cout << "self-assign: " << hello << " (still valid)\n";

    std::string temp = "dynamic string content";
    StringView view_temp(temp);
    StringView copied_view_temp(view_temp);
    temp[0] = 'D';
    std::cout << "after modifying original string: view_temp=" << view_temp
              << ", copied_view=" << copied_view_temp << "\n";
    std::cout << "  (both affected — non-owning view)\n";

    std::cout << "\n== move ==\n";
    StringView moved(std::move(copied));
    std::cout << "moved-from: size=" << copied.size() << " empty=" << copied.empty() << '\n';
    std::cout << "moved-to: " << moved << '\n';

    StringView move_assigned("will be replaced");
    move_assigned = std::move(moved);
    std::cout << "moved-from (after assign): size=" << moved.size() << '\n';
    std::cout << "move-assigned-to: " << move_assigned << '\n';
}
