"""AOT compile NineToothed MHA to HIP without changing installed packages.

NineToothed 0.26's public AOT launcher assumes CUDA .c templates and cuCtxGetId.
This adapter uses its CodeGenerator, then Triton's compiler for the generated AST.
Only host launch/argument plumbing is emitted here; no handwritten GPU kernel.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import ninetoothed as nt
import ninetoothed.generation as generation
import ninetoothed.naming as naming
from ninetoothed.tensor import Tensor
import triton
from triton.compiler import ASTSource
from ninetoothed.aot import _GridExtractor
from attention import premake
# DTK 26.04's assembly parser predates LLVM's GEP nuw/nusw flags.
# Dropping these optional no-wrap promises only weakens optimization assumptions.
# Keep this compatibility shim local to this AOT process, never site-packages.
from triton.backends.amd.compiler import HIPBackend
import re
_original_make_amdgcn = HIPBackend.make_amdgcn

def _dtk_make_amdgcn(src, metadata, options):
    if 'das.opt1.dtk2604' in __import__('importlib.metadata').metadata.version('triton'):
        src, removed = re.subn(r'(getelementptr(?: inbounds)?) (?:nuw|nusw)\b', r'\1', src)
        if removed:
            print(f'DTK LLVM compatibility: removed {removed} GEP no-wrap flags', flush=True)
    return _original_make_amdgcn(src, metadata, options)

HIPBackend.make_amdgcn = staticmethod(_dtk_make_amdgcn)

HEADER = '''#pragma once
#include <stdint.h>
struct H3NTTensor { void *data; int64_t shape[4]; int64_t strides[4]; };
extern "C" int h3_nt_attention(void *stream, H3NTTensor q, H3NTTensor k,
    H3NTTensor v, H3NTTensor out, float scale, int dtype, int causal);
extern "C" uint64_t h3_nt_launch_count(int device, int head_dim, int causal);
'''
PRELUDE = '''#include "h3_nt_attention.h"
#include <hip/hip_runtime.h>
#include <atomic>
#include <map>
#include <mutex>
static std::atomic<uint64_t> counts[64][257][2]{};
extern "C" uint64_t h3_nt_launch_count(int device, int head_dim, int causal) {
    if (device < 0 || device >= 64 || head_dim < 0 || head_dim > 256 || causal < 0 || causal > 1) return 0;
    return counts[device][head_dim][causal].load();
}
'''


def build_variant(out_dir, dtype, head_dim, causal):
    name = f'h3_nt_{dtype}_{head_dim}_{int(causal)}'
    arrangement, application, tensors = premake(head_dim, dtype, causal)
    application.__annotations__ = dict(zip(inspect.signature(application).parameters, arrangement(*tensors)))
    generator = generation.CodeGenerator()
    src = Path(generator(application, caller='cuda', kernel_name=name,
                         num_warps=4, num_stages=1, max_num_configs=None, prettify=False))
    # Stable named copy retains the complete NineToothed-generated GPU source.
    generated = out_dir / f'{name}.py'
    generated.write_text(src.read_text())
    spec = importlib.util.spec_from_file_location(name, generated)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, name)
    tensor_map = {naming.remove_prefixes(t.name): t for t in tensors}
    cpp_tensor_map = {naming.remove_prefixes(t.name): label for t, label in zip(tensors, ('q','k','v','scale','out','causal'))}
    signature, constants, declarations, arguments = {}, {}, [], []
    for param in kernel.arg_names:
        if match := Tensor.pointer_pattern().fullmatch(param):
            key = naming.remove_prefixes(match[1]); t = tensor_map[key]
            signature[param] = f'*{t.dtype}'
            expression = f'{cpp_tensor_map[key]}.data'; ctype = 'void *'
        elif match := Tensor.size_pattern().fullmatch(param):
            key = naming.remove_prefixes(match[1]); dim = int(match[3])
            signature[param] = 'i32'
            expression = f'{cpp_tensor_map[key]}.shape[{dim}]'; ctype = 'int32_t'
        elif match := Tensor.stride_pattern().fullmatch(param):
            key = naming.remove_prefixes(match[1]); dim = int(match[3])
            if dim == 3:
                signature[param] = 'constexpr'; constants[param] = 1
                continue
            signature[param] = 'i64'
            expression = f'{cpp_tensor_map[key]}.strides[{dim}]'; ctype = 'int64_t'
        else:
            key = naming.remove_prefixes(param); t = tensor_map[key]
            if t.constexpr:
                signature[param] = 'constexpr'; constants[param] = t.value
                continue
            signature[param] = 'fp32'; expression = 'scale'; ctype = 'float'
        declarations.append(f'    {ctype} {param} = ({ctype})({expression});')
        arguments.append(f'&{param}')
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != 'hip':
        raise RuntimeError(f'Expected HIP target, got {target}')
    compiled = triton.compile(ASTSource(kernel, signature, constexprs=constants),
                              target=target, options={'num_warps':4,'num_stages':1})
    meta = compiled.metadata
    if getattr(meta,'global_scratch_size',0) or getattr(meta,'profile_scratch_size',0):
        raise RuntimeError('Scratch-requiring kernels are not supported by this launcher')
    binary = compiled.asm['hsaco']
    (out_dir / f'{name}.hsaco').write_bytes(binary)
    extractor = _GridExtractor(); extractor.visit(generator.raw_grid)
    grid = ast.unparse(extractor.grid[0])
    # The grid references dimension arguments; all are emitted as C++ locals above.
    cpp = f'''#include "h3_nt_attention.h"
#include <hip/hip_runtime.h>
#include <map>
static const unsigned char binary[] = {{{','.join(str(x) for x in binary)}}};
int launch_{name}(void *stream, H3NTTensor q, H3NTTensor k, H3NTTensor v, H3NTTensor out, float scale) {{
    struct Loaded {{ hipModule_t module; hipFunction_t function; }};
    // Rank workers use different devices/contexts. Never share a function handle across them.
    static thread_local std::map<std::pair<int, hipCtx_t>, Loaded> loaded;
    hipCtx_t context; auto status = hipCtxGetCurrent(&context);
    if (status != hipSuccess) return status;
    int device; status = hipGetDevice(&device);
    if (status != hipSuccess) return status;
    const auto key = std::make_pair(device, context);
    auto it = loaded.find(key);
    if (it == loaded.end()) {{
        Loaded item{{}};
        status = hipModuleLoadData(&item.module, binary);
        if (status != hipSuccess) return status;
        status = hipModuleGetFunction(&item.function, item.module, "{meta.name}");
        if (status != hipSuccess) {{ hipModuleUnload(item.module); return status; }}
        it = loaded.emplace(key, item).first;
    }}
{chr(10).join(declarations)}
    void *global_scratch = nullptr, *profile_scratch = nullptr;
    void *args[] = {{{', '.join(arguments)}, &global_scratch, &profile_scratch}};
    return hipModuleLaunchKernel(it->second.function, {grid}, 1, 1,
        {meta.num_warps * target.warp_size}, 1, 1, {meta.shared}, (hipStream_t)stream, args, nullptr);
}}
'''
    (out_dir / f'{name}.cpp').write_text(cpp)
    return name, {'name':name,'target':str(target),'shared_memory':meta.shared,
                  'generated_sha256':hashlib.sha256(generated.read_bytes()).hexdigest(),
                  'hsaco_sha256':hashlib.sha256(binary).hexdigest()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--head-dims',type=int,nargs='+',default=[72,96,128])
    parser.add_argument('--dtypes',nargs='+',default=['bf16','fp16'])
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    generation.CACHE_DIR=out
    (out/'h3_nt_attention.h').write_text(HEADER)
    variants=[];declarations=[];branches=[]
    for dtype in args.dtypes:
        for head in args.head_dims:
            for causal in (False,True):
                name,record=build_variant(out,dtype,head,causal);variants.append(record)
                declarations.append(f'int launch_{name}(void *, H3NTTensor, H3NTTensor, H3NTTensor, H3NTTensor, float);')
                branches.append(f'    if (dtype == {0 if dtype=="bf16" else 1} && q.shape[3] == {head} && causal == {int(causal)}) status = launch_{name}(stream,q,k,v,out,scale);')
                print(json.dumps(record),flush=True)
    dispatch=PRELUDE+'\n'.join(declarations)+'''
extern "C" int h3_nt_attention(void *stream, H3NTTensor q, H3NTTensor k, H3NTTensor v, H3NTTensor out, float scale, int dtype, int causal) {
    int status = hipErrorNotSupported;
'''+ '\n'.join(branches)+'''
    if (status == hipSuccess) {
        int device; if (hipGetDevice(&device) == hipSuccess && device >= 0 && device < 64)
            ++counts[device][q.shape[3]][causal];
    }
    return status;
}
'''
    (out/'h3_nt_attention.cpp').write_text(dispatch)
    (out/'manifest.json').write_text(json.dumps({'ninetoothed':__import__('importlib.metadata').metadata.version('ninetoothed'),'triton':triton.__version__,'variants':variants},indent=2))
    sources=[out/'h3_nt_attention.cpp']+[out/f"{v['name']}.cpp" for v in variants]
    subprocess.run(['c++','-shared','-fPIC','-O2','-std=c++17','-D__HIP_PLATFORM_AMD__',
                    '-I/opt/dtk/include','-L/opt/dtk/lib','-Wl,-rpath,/opt/dtk/lib',
                    *map(str,sources),'-lamdhip64','-o',str(out/'libh3_ninetoothed.so')],check=True)

if __name__=='__main__': main()
