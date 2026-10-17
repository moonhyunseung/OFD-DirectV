"""Pinned diagnostic graph benchmark. No quantizer changes or performance claim beyond the declared target.
Re-run from repository root with: python experiments/h200_graph_boundary.py --out /absolute/new/output
Candidate source SHA and repository revision are recorded before execution. Results are intentionally external.
"""
import argparse,hashlib,json,random,statistics,subprocess
from pathlib import Path
import torch,triton
import triton.language as tl
STAGE='V'
CANDIDATES=['offset','signed']

@triton.jit
def directv_native(P,V,S,O,L:tl.constexpr,SIGNED:tl.constexpr,BT:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);g=tl.program_id(2)
    t=b*BT+tl.arange(0,BT);j=tl.arange(0,32)
    p=tl.load(P+h*L+t,t<L,0).to(tl.float32)
    s=tl.load(S+t*8+g,t<L,0).to(tl.float32)
    w=p*s
    byte=tl.load(V+t[:,None]*128+g*16+j[None,:]//2,t[:,None]<L,0).to(tl.int32)
    c=(byte>>((j[None,:]%2)*4))&15
    if SIGNED:
        n=c^8
        y=tl.sum(w[:,None]*tl.where(n>=8,n-16,n).to(tl.float32),0)
    else:
        y=tl.sum(w[:,None]*c.to(tl.float32),0)-8*tl.sum(w,0)
    tl.store(O+b*1536+h*256+g*32+j,y)

def pack(c): return (c[:,:,0::2]|(c[:,:,1::2]<<4)).byte().contiguous()
def decode(pk,s):
    L=pk.shape[0]
    c=torch.stack(((pk&15).float(),(pk>>4).float()),-1).reshape(L,8,32)
    return ((c-8)*s.float()[:,:,None]).reshape(L,256)

def audit(kernel):
    ptx=kernel.asm['ptx']
    return dict(registers=getattr(kernel,'n_regs',None),spills=getattr(kernel,'n_spills',None),
                shared_bytes=getattr(kernel.metadata,'shared',None),
                mma_sync=ptx.count('mma.sync'),wgmma=ptx.count('wgmma.'),
                ld_global=ptx.count('ld.global'),st_global=ptx.count('st.global'),
                ld_local=ptx.count('ld.local'),st_local=ptx.count('st.local'))

def invoke(method,q,p,pk,s):
    L=pk.shape[0]
    if STAGE=='K':
        y=torch.empty(6,L,device='cuda',dtype=torch.float32)
        if method=='tensor_signed':
            kernel=directk_tensor[(triton.cdiv(L,64),)](q,pk,s,y,L,64,num_warps=4,enable_fp_fusion=False)
        else:
            kernel=directk_native[(triton.cdiv(L,8),6)](q,pk,s,y,L,method=='signed',8,enable_fp_fusion=False)
    else:
        nb=triton.cdiv(L,128)
        partial=torch.empty(nb,6,256,device='cuda',dtype=torch.float32)
        kernel=directv_native[(nb,6,8)](p,pk,s,partial,L,method=='signed',128,enable_fp_fusion=False)
        y=partial.sum(0)
    return y,kernel

def gate(out):
    rows=[]; kernels={}
    torch.manual_seed(1729 if STAGE=='K' else 1730)
    for L in [1,17,129,1024,8192]:
        for case in ['random','zero_code8','alternating','max_scale','tiny_scale']:
            c=torch.randint(0,16,(L,8,32),device='cuda',dtype=torch.int32)
            if case=='zero_code8':c.fill_(8)
            if case=='alternating':c=(torch.arange(32,device='cuda')%2*15).expand(L,8,32).contiguous().int()
            s=(torch.rand(L,8,device='cuda')+.01).half()
            if case=='max_scale':s.fill_(65504)
            if case=='tiny_scale':s.fill_(2**-24)
            q=torch.randn(6,256,device='cuda').half()
            p=torch.softmax(torch.randn(6,L,device='cuda'),1)
            pk=pack(c);dense=((c.double()-8)*s.double()[:,:,None]).reshape(L,256)
            if STAGE=='K':
                ref=q.double()@dense.T/16
                env=q.double().abs()@dense.abs().T/16
            else:
                ref=p.double()@dense
                env=p.double().abs()@dense.abs()
            for method in CANDIDATES:
                y,k=invoke(method,q,p,pk,s);kernels[method]=k
                if STAGE=='K': bound=64*torch.finfo(torch.float32).eps*env+1e-6
                else:
                    e=env
                    if method=='offset':e=p.double().abs()@((c.double()+8)*s.double()[:,:,None]).reshape(L,256)
                    bound=(512+2*triton.cdiv(L,128))*torch.finfo(torch.float32).eps*e+1e-6
                err=(y.double()-ref).abs()
                rows.append(dict(length=L,case=case,method=method,passed=bool(torch.isfinite(y).all() and (err<=bound).all()),
                                 max_abs=err.max().item(),max_budget_ratio=(err/bound).max().item()))
    for name,k in kernels.items():
        for kind in ['ptx','llir']:(out/(name+'.'+kind)).write_text(k.asm[kind])
    (out/'numerical.json').write_text(json.dumps(rows,indent=2))
    (out/'instruction_audit.json').write_text(json.dumps({n:audit(k) for n,k in kernels.items()},indent=2))
    if not all(r['passed'] for r in rows):raise RuntimeError('Predeclared diagnostic gate failed; do not benchmark')
    print('GATE_PASS',len(rows),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--profile-method',default='')
    args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(1)
    cfg=dict(status='RUNNING',stage=STAGE,seed_list=[1731,1732,1733],lengths=[129,1024,8192,32768,131072],
             graph_operations=32,samples=31,warmup=8,shape='B=1,KV=1,r=6,G=8,m=32,d=256',
             scope='resident repeated-operand microbenchmark; CUDA Graphs; not HBM/serving throughput',
             candidates=CANDIDATES,torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,
             gpu=torch.cuda.get_device_name(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
             source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=Path(__file__).resolve().parents[1],text=True).strip()),
             numerical_budget='Diagnostic only, fixed before data: K 64*eps32*abs-term-envelope+1e-6; V (512+2*ceil(L/128))*eps32*envelope+1e-6. Not proven worst-case or production quality contract.',
             precision='Q/scales FP16, P FP32, accumulation/output FP32; TF32 off; fp fusion off')
    (out/'MANIFEST.json').write_text(json.dumps(cfg,indent=2))
    def save(): (out/'MANIFEST.json').write_text(json.dumps(cfg,indent=2))
    if args.profile_method:
        torch.manual_seed(1731);L=8192;c=torch.randint(0,16,(L,8,32),device='cuda',dtype=torch.int32)
        s=(torch.rand(L,8,device='cuda')+.01).half();pk=pack(c);q=torch.randn(6,256,device='cuda').half();p=torch.softmax(torch.randn(6,L,device='cuda'),1)
        q32=q.float();dense=decode(pk,s)
        def fn():
            if args.profile_method=='decode_first':return q32@decode(pk,s).T/16 if STAGE=='K' else p@decode(pk,s)
            return invoke(args.profile_method,q,p,pk,s)[0]
        for _ in range(3):fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push('ofd_profile')
        y=fn()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        cfg.update(status='PROFILE_ONLY',profile_method=args.profile_method);save();return
    try:
        gate(out)
        rows=[];rng=random.Random(1731)
        for seed in cfg['seed_list']:
            torch.manual_seed(seed)
            for L in cfg['lengths']:
                c=torch.randint(0,16,(L,8,32),device='cuda',dtype=torch.int32)
                s=(torch.rand(L,8,device='cuda')+.01).half();pk=pack(c);q=torch.randn(6,256,device='cuda').half()
                q32=q.float();p=torch.softmax(torch.randn(6,L,device='cuda'),1);dense=decode(pk,s)
                def base():return q32@decode(pk,s).T/16 if STAGE=='K' else p@decode(pk,s)
                def oracle():return q32@dense.T/16 if STAGE=='K' else p@dense
                funcs={n:(lambda n=n:invoke(n,q,p,pk,s)[0]) for n in CANDIDATES}
                funcs.update(decode_first=base,predecoded_oracle=oracle)
                ref=oracle();graphs={};outputs={};stream=torch.cuda.Stream()
                for name,fn in funcs.items():
                    for _ in range(cfg['warmup']):fn()
                    torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        for _ in range(cfg['graph_operations']):y=fn()
                    graph.replay();torch.cuda.synchronize()
                    torch.testing.assert_close(y,ref,atol=1e-4,rtol=2e-4)
                    graphs[name]=graph;outputs[name]=y
                samples={n:[] for n in graphs}
                for _ in range(cfg['samples']):
                    names=list(graphs);rng.shuffle(names)
                    for name in names:
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                        a.record();graphs[name].replay();b.record();b.synchronize()
                        samples[name].append(a.elapsed_time(b)/cfg['graph_operations'])
                for name,ms in samples.items():
                    rows.append(dict(seed=seed,length=L,method=name,median_ms=statistics.median(ms),
                                     min_ms=min(ms),max_ms=max(ms),raw_ms=ms))
                (out/'timings.json').write_text(json.dumps(rows,indent=2))
                print(json.dumps(dict(seed=seed,L=L,median_ms={n:statistics.median(v) for n,v in samples.items()})),flush=True)
                del graphs,outputs,ref,dense
                torch.cuda.empty_cache()
        cfg.update(status='PASS',timing_rows=len(rows));save();print('GRAPH_PASS',out,flush=True)
    except Exception as e:
        cfg.update(status='FAIL',error=repr(e));save();raise

if __name__=='__main__':main()
