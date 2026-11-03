#include "infinicore/ops/add.hpp"
#include "../../utils.hpp"
#include <sstream>
#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Add);

Add::Add(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    try {
        INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b);
    } catch (const std::exception &error) {
        std::ostringstream message;
        message << error.what() << " (add shapes: output=[";
        for (auto dim : c->shape()) message << dim << ",";
        message << "], a=[";
        for (auto dim : a->shape()) message << dim << ",";
        message << "], b=[";
        for (auto dim : b->shape()) message << dim << ",";
        message << "])";
        throw std::runtime_error(message.str());
    }
}

void Add::execute(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Add, c, a, b);
}

Tensor add(const Tensor &a, const Tensor &b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    add_(c, a, b);
    return c;
}

void add_(Tensor c, const Tensor &a, const Tensor &b) {
    Add::execute(c, a, b);
}

} // namespace infinicore::op
