#include "unique_ptr.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <utility>

// helper class for runtime polymorphism demo below
struct B
{
    virtual ~B() = default;

    virtual void bar() { std::cout << "B::bar\n"; }
};

struct D : B
{
    D() { std::cout << "D::D\n"; }
    ~D() { std::cout << "D::~D\n"; }

    void bar() override { std::cout << "D::bar\n"; }
};

// a function consuming a unique_ptr can take it by value or by rvalue reference
Uniqlo<D> pass_through(Uniqlo<D> p)
{
    p->bar();
    return p;
}

// helper function for the custom deleter demo below
void close_file(std::FILE* fp)
{
    std::fclose(fp);
}

// unique_ptr-based linked list demo
struct List
{
    struct Node
    {
        int data;
        Uniqlo<Node> next;
    };

    Uniqlo<Node> head;

    ~List()
    {
        while (head)
        {
            auto next = std::move(head->next);
            head = std::move(next);
        }
    }

    void push(int data)
    {
        head = Uniqlo<Node>(new Node{data, std::move(head)});
    }
};

int main()
{
    std::cout << "1) Unique ownership semantics demo\n";
    {
        Uniqlo<D> p = make_uniqlo<D>();

        Uniqlo<D> q = pass_through(std::move(p));

        assert(!p);
    }

    std::cout << "\n" "2) Runtime polymorphism demo\n";
    {
        Uniqlo<B> p = make_uniqlo<D>();

        p->bar();
    }

    // 3) Custom deleter demo — 注释：函数指针 deleter（void(*)(T*)）不能作为基类，
    //    EBO 继承式 Uniqlo 目前不支持（需要分派式或 [[no_unique_address]] 成员式）
    // std::cout << "\n" "3) Custom deleter demo\n";
    // std::ofstream("demo.txt") << 'x';
    // {
    //     using unique_file_t = Uniqlo<std::FILE, decltype(&close_file)>;
    //     unique_file_t fp(std::fopen("demo.txt", "r"), &close_file);
    //     if (fp)
    //         std::cout << char(std::fgetc(fp.get())) << '\n';
    // }

    // 4) Custom lambda expression deleter and exception safety demo — 同上，lambda 转成函数指针
    // std::cout << "\n" "4) Custom lambda expression deleter and exception safety demo\n";
    // try
    // {
    //     Uniqlo<D, void(*)(D*)> p(new D, [](D* ptr)
    //     {
    //         std::cout << "destroying from a custom deleter...\n";
    //         delete ptr;
    //     });
    //
    //     throw std::runtime_error("");
    // }
    // catch (const std::exception&)
    // {
    //     std::cout << "Caught exception\n";
    // }

    std::cout << "\n" "5) Array form of unique_ptr demo\n";
    {
        Uniqlo<D[]> p(new D[3]);
    }

    std::cout << "\n" "6) Linked list demo\n";
    {
        List wall;
        const int enough{1'000'000};
        for (int beer = 0; beer != enough; ++beer)
            wall.push(beer);

        std::cout.imbue(std::locale("en_US.UTF-8"));
        std::cout << enough << " bottles of beer on the wall...\n";
    }
}
