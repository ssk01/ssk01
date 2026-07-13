#include "StringView.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <cstdlib>

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

static double elapsed(Ms d) { return d.count(); }

int main() {
    std::ios::sync_with_stdio(false);

    std::string line = "alpha,beta,gamma,delta,epsilon,zeta,eta,theta,iota,kappa";
    int repeat = 500'000;

    std::cout << "=== split (repeated " << repeat << " times) ===\n\n";

    {
        auto t0 = Clock::now();
        for (int r = 0; r < repeat; ++r) {
            std::vector<std::string> parts;
            std::stringstream ss(line);
            std::string item;
            while (std::getline(ss, item, ',')) {
                parts.push_back(item);
            }
            if (parts.size() != 10) std::abort();
        }
        auto t1 = Clock::now();
        std::cout << "  string + stringstream: " << elapsed(t1 - t0) << " ms\n";
    }

    {
        auto t0 = Clock::now();
        for (int r = 0; r < repeat; ++r) {
            std::vector<std::string> parts;
            size_t pos = 0;
            while (pos <= line.size()) {
                size_t next = line.find(',', pos);
                if (next == std::string::npos) next = line.size();
                parts.push_back(line.substr(pos, next - pos));
                pos = next + 1;
            }
            if (parts.size() != 10) std::abort();
        }
        auto t1 = Clock::now();
        std::cout << "  string split:          " << elapsed(t1 - t0) << " ms\n";
    }

    {
        auto t0 = Clock::now();
        for (int r = 0; r < repeat; ++r) {
            std::vector<StringView> parts;
            StringView sv(line);
            size_t pos = 0;
            while (pos < sv.size()) {
                size_t next = sv.find(',', pos);
                if (next == StringView::npos) {
                    parts.push_back(sv.substr(pos));
                    break;
                }
                parts.push_back(sv.substr(pos, next - pos));
                pos = next + 1;
            }
            if (parts.size() != 10) std::abort();
        }
        auto t1 = Clock::now();
        std::cout << "  StringView split:      " << elapsed(t1 - t0) << " ms\n";
    }

    std::cout << "\n=== substr (repeated " << repeat << " times) ===\n\n";

    {
        auto t0 = Clock::now();
        volatile size_t total = 0;
        for (int r = 0; r < repeat; ++r) {
            for (size_t i = 0; i < line.size(); ++i) {
                auto s = line.substr(0, i + 1);
                total += s.size();
            }
        }
        auto t1 = Clock::now();
        std::cout << "  string substr:         " << elapsed(t1 - t0) << " ms\n";
    }

    {
        auto t0 = Clock::now();
        volatile size_t total = 0;
        StringView sv(line);
        for (int r = 0; r < repeat; ++r) {
            for (size_t i = 0; i < sv.size(); ++i) {
                auto s = sv.substr(0, i + 1);
                total += s.size();
            }
        }
        auto t1 = Clock::now();
        std::cout << "  StringView substr:     " << elapsed(t1 - t0) << " ms\n";
    }

    std::cout << "\n=== compare (repeated " << (repeat * 10) << " times) ===\n\n";

    {
        auto t0 = Clock::now();
        volatile int result = 0;
        for (int r = 0; r < repeat * 10; ++r) {
            result += (line > "alpha");
            result += (line == "alpha,beta,gamma,delta,epsilon,zeta,eta,theta,iota,kappa");
            result += (line < "zzz");
        }
        auto t1 = Clock::now();
        std::cout << "  string compare:        " << elapsed(t1 - t0) << " ms\n";
    }

    {
        auto t0 = Clock::now();
        volatile int result = 0;
        StringView sv(line);
        for (int r = 0; r < repeat * 10; ++r) {
            result += (sv > StringView("alpha"));
            result += (sv == StringView("alpha,beta,gamma,delta,epsilon,zeta,eta,theta,iota,kappa"));
            result += (sv < StringView("zzz"));
        }
        auto t1 = Clock::now();
        std::cout << "  StringView compare:    " << elapsed(t1 - t0) << " ms\n";
    }

    std::cout << "\n=== pass-by-value overhead (repeated " << (repeat * 2) << " times) ===\n\n";

    auto by_string = [](std::string s) { return s.size(); };
    auto by_sv = [](StringView sv) { return sv.size(); };

    {
        auto t0 = Clock::now();
        volatile size_t total = 0;
        for (int r = 0; r < repeat * 2; ++r) {
            total += by_string(line);
        }
        auto t1 = Clock::now();
        std::cout << "  pass std::string:      " << elapsed(t1 - t0) << " ms\n";
    }

    {
        auto t0 = Clock::now();
        volatile size_t total = 0;
        StringView sv(line);
        for (int r = 0; r < repeat * 2; ++r) {
            total += by_sv(sv);
        }
        auto t1 = Clock::now();
        std::cout << "  pass StringView:       " << elapsed(t1 - t0) << " ms\n";
    }
}
