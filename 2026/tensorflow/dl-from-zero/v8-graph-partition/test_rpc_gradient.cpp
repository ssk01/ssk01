#include <iostream>
#include "demo_rpc_gradient.h"

using namespace lf;

int main() {
    std::cout << "== RPC Send/Recv 梯度处理 Demo ==" << std::endl;
    std::cout << "展示为什么需要 Send/Recv 的梯度处理\n" << std::endl;

    demo_rpc_sendrecv_gradient();

    std::cout << "\n== Demo 完成 ==" << std::endl;
    return 0;
}
