#if defined(ENABLE_HYGON_API) && defined(ENABLE_HYGON_NINETOOTHED_MHA)
#include "infinicore/ops/mha.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "h3_nt_attention.h"
#include <c10/hip/HIPGuard.h>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace infinicore::op::mha_impl::ninetoothed_hygon {
struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    float scale;
    bool causal;
};

void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
           std::optional<Tensor> alibi, float scale, bool causal) {
    if (alibi) {
        throw std::invalid_argument("NineToothed Hygon MHA does not support ALiBi; no vendor fallback");
    }
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("NineToothed Hygon MHA requires finite scale");
    }
    return new PlannedMeta{graph::GraphTensor(out), graph::GraphTensor(q),
                           graph::GraphTensor(k), graph::GraphTensor(v), scale, causal};
}

// Reinterpret B,S,H,D as B,H,S,D using metadata only.
H3NTTensor argument(const at::Tensor &t) {
    H3NTTensor result{};
    result.data = t.data_ptr();
    const int order[] = {0, 2, 1, 3};
    for (int i = 0; i < 4; ++i) {
        result.shape[i] = t.size(order[i]);
        result.strides[i] = t.stride(order[i]);
    }
    return result;
}

void run(void *meta) {
#if INFINICORE_TORCH_VERSION_GE_2_11
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#endif
    auto *p = static_cast<PlannedMeta *>(meta);
    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);
    auto out = infinicore::adaptor::to_aten_tensor(p->out);
    for (const auto &t : {q, k, v, out}) {
        if (t.dim() != 4 || t.scalar_type() != q.scalar_type())
            throw std::invalid_argument("NineToothed Hygon MHA expects matching 4D BF16/FP16 tensors");
        for (int i = 0; i < 4; ++i) {
            if (t.size(i) <= 0 || t.size(i) > std::numeric_limits<int32_t>::max() || t.stride(i) < 0)
                throw std::invalid_argument("NineToothed Hygon MHA: invalid dimension/stride");
        }
    }
    const auto d = q.size(3);
    if ((d != 72 && d != 96 && d != 128)
        || (q.scalar_type() != at::kBFloat16 && q.scalar_type() != at::kHalf))
        throw std::invalid_argument("NineToothed Hygon MHA supports head_dim 72/96/128 and BF16/FP16 only; no vendor fallback");
    if (q.size(0) != k.size(0) || k.sizes() != v.sizes() || out.sizes() != q.sizes()
        || k.size(3) != d || q.size(2) % k.size(2) != 0)
        throw std::invalid_argument("NineToothed Hygon MHA: incompatible Q/K/V/output shapes");
    // AOT specializes the innermost stride to 1. Other positive strides are dynamic.
    if (q.stride(3) != 1) q = q.contiguous();
    if (k.stride(3) != 1) k = k.contiguous();
    if (v.stride(3) != 1) v = v.contiguous();
    bool copy_back = out.stride(3) != 1 || out.data_ptr() == q.data_ptr()
                     || out.data_ptr() == k.data_ptr() || out.data_ptr() == v.data_ptr();
    auto work = copy_back ? at::empty(out.sizes(), out.options()) : out;
    auto stream = infinicore::adaptor::get_hip_stream().stream();
    const int status = h3_nt_attention(stream, argument(q), argument(k), argument(v),
        argument(work), p->scale, q.scalar_type() == at::kBFloat16 ? 0 : 1, p->causal);
    if (status != 0)
        throw std::runtime_error("NineToothed Hygon attention launch failed: " + std::to_string(status));
    if (copy_back) out.copy_(work);
}

void cleanup(void **meta) {
    delete static_cast<PlannedMeta *>(*meta);
    *meta = nullptr;
}
static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();
} // namespace infinicore::op::mha_impl::ninetoothed_hygon
#endif
