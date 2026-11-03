"""NineToothed online-softmax MHA for Hygon, derived from infiniop flash_attention.

Inputs are logical [batch, heads, sequence, head_dim]. No quadratic buffer.
Causal masking is lower-right, matching the existing FlashAttention MHA API.
"""
import functools
import ninetoothed as nt
import ninetoothed.language as ntl


def arrangement(q, k, v, scale, out, causal, block_m=64, block_n=64):
    def arrange_q(x):
        y = x.tile((1, 1, block_m, -1)).tile((1, q.shape[1] // k.shape[1], 1, 1))
        y.dtype = y.dtype.squeeze((0, 2, 3))
        y.dtype.dtype = y.dtype.dtype.squeeze((0, 1))
        return y

    qa = arrange_q(q)

    def arrange_kv(x):
        y = x.tile((1, 1, block_n, -1)).tile((1, 1, -1, -1)).expand((-1, -1, qa.shape[-2], -1))
        y.dtype = y.dtype.squeeze((0, 1, 3))
        y.dtype.dtype = y.dtype.dtype.squeeze((0, 1))
        return y

    return qa, arrange_kv(k), arrange_kv(v), scale, arrange_q(out), causal


def application(q, k, v, scale, out, causal):
    for i in range(q.shape[0]):
        qi = q[i]
        acc = ntl.zeros((qi.shape[0], qi.shape[1]), dtype=ntl.float32)
        denom = ntl.full((qi.shape[0],), 0, dtype=ntl.float32)
        maximum = ntl.full((qi.shape[0],), float('-inf'), dtype=ntl.float32)
        for j in range(k.shape[0]):
            scores = ntl.dot(qi, ntl.trans(k[j])) * (scale * 1.4426950408889634)
            kp = k[j].offsets(-2)
            scores = ntl.where(kp[None, :] < k.source.shape[-2], scores, float('-inf'))
            if causal:
                qp = q[i].offsets(-2)
                scores = ntl.where(qp[:, None] + k.source.shape[-2] - q.source.shape[-2] >= kp[None, :], scores, float('-inf'))
            new_max = ntl.maximum(maximum, ntl.max(scores, 1))
            # Fully masked rows must return zero, including unequal causal lengths.
            safe_max = ntl.where(new_max == float('-inf'), 0, new_max)
            prob = ntl.exp2(scores - safe_max[:, None])
            alpha = ntl.exp2(maximum - safe_max)
            acc = acc * alpha[:, None] + ntl.dot(prob.to(v[j].dtype), v[j])
            denom = denom * alpha + ntl.sum(prob, 1)
            maximum = new_max
        out[i] = acc / ntl.where(denom == 0, 1, denom)[:, None]


def premake(head_dim=128, dtype=nt.bfloat16, causal=False, block_m=64, block_n=64):
    tensors = [nt.Tensor(shape=(None, None, None, head_dim), dtype=dtype) for _ in range(4)]
    q, k, v, out = tensors
    scale = nt.Tensor(0, dtype=nt.float32)
    causal_t = nt.Tensor(0, constexpr=True, value=int(causal))
    return functools.partial(arrangement, block_m=block_m, block_n=block_n), application, (q, k, v, scale, out, causal_t)
