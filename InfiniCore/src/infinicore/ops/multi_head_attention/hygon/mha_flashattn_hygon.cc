#if defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN) && !defined(ENABLE_HYGON_NINETOOTHED_MHA)
#include "infinicore/ops/mha.hpp"

#include "../../../adaptor/flash_attn/hygon/flash_attn_hygon.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <limits>
#include <stdexcept>

#include <c10/hip/HIPGuard.h>

namespace infinicore::op::mha_impl::flashattn {
struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    bool is_causal;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {

    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale,
        is_causal};
}

void run(void *planned_meta) {
#if INFINICORE_TORCH_VERSION_GE_2_11
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_hip_stream());
#else
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
#endif
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);

    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work_ic = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_work = infinicore::adaptor::to_aten_tensor(out_work_ic);
    auto scale = p->scale;
    auto is_causal = p->is_causal;

    // Pack equal-length batches for the vendor's FlashAttention varlen API.
    // Do not materialize the quadratic [B, H, S, S] attention matrix here.
    if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) {
        throw std::runtime_error("Hygon MHA expects [batch, sequence, heads, head_dim]");
    }
    // The installed Hygon kernel needs padding for non-power-of-two heads.
    // Pad Q/K/V to 128 but retain the caller's original softmax scale.
    const int64_t head_dim = q.size(3);
    const bool pad_head = head_dim == 72 || head_dim == 96;
    if (pad_head) {
        q = at::constant_pad_nd(q, {0, 128 - head_dim}, 0);
        k = at::constant_pad_nd(k, {0, 128 - head_dim}, 0);
        v = at::constant_pad_nd(v, {0, 128 - head_dim}, 0);
    }
    // The vendor kernel is incorrect when two inputs have identical addresses.
    // Keep distinct buffers even for the valid Q=K=V self-attention case.
    if (q.data_ptr() == k.data_ptr()) {
        k = k.clone();
    }
    if (v.data_ptr() == q.data_ptr() || v.data_ptr() == k.data_ptr()) {
        v = v.clone();
    }
    const int batch = q.size(0);
    const int q_len = q.size(1);
    const int k_len = k.size(1);
    if (batch != k.size(0) || batch != v.size(0) || k_len != v.size(1)) {
        throw std::runtime_error("Hygon MHA batch/sequence dimensions do not match");
    }
    auto q_packed = q.reshape({batch * q_len, q.size(2), q.size(3)});
    auto k_packed = k.reshape({batch * k_len, k.size(2), k.size(3)});
    auto v_packed = v.reshape({batch * k_len, v.size(2), v.size(3)});
    auto cu_q = at::arange(batch + 1, q.options().dtype(at::kInt)) * q_len;
    auto cu_k = at::arange(batch + 1, k.options().dtype(at::kInt)) * k_len;
    std::optional<at::Tensor> out = pad_head
        ? at::empty_like(q_packed) : out_work.reshape(q_packed.sizes());
    std::optional<at::Tensor> seqused = std::nullopt;
    std::optional<const at::Tensor> leftpad = std::nullopt;
    std::optional<at::Tensor> block_table = std::nullopt;
    auto alibi = p->alibi_slopes
        ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
        : std::nullopt;
    flash::vllm_mha_varlen_fwd(
        q_packed, k_packed, v_packed, out, cu_q, cu_k, seqused, leftpad,
        block_table, alibi, q_len, k_len, 0.0f, scale, false, is_causal,
        -1, -1, 0.0f, false, std::nullopt);
    if (pad_head) {
        out_work.copy_(out->narrow(2, 0, head_dim).reshape(out_work.sizes()));
    }

    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::flashattn
#endif // ENABLE_HYGON_API && ENABLE_FLASH_ATTN
