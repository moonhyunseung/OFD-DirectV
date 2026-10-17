"""FP32-coefficient native V via exact three-part TF32 operand decomposition.
For a normal float32 w, truncate the low 13 mantissa bits into w0; repeat on residual into w1.
The remaining w2 has at most two significant residual bits. With |w|>=2^-100, all nonzero
parts are normal TF32-exact operands and w=w0+w1+w2 over the reals. Signed INT4 integers
are TF32-exact. Thus three dot products preserve the coefficient value, subject only to
the separately declared FP32 product/reduction rounding. This is not an FP16 probability cast.
Tiles containing nonzero coefficients below that range use a SIMT FP32 fallback.
"""
import argparse,hashlib,json,subprocess,random,statistics
from pathlib import Path
import torch,triton
import triton.language as tl
import h200_graph_boundary as base
import h200_tile_holdout as bench
EXPECTED_BASE='174ff55f19888523c8bdd671f601dda1ebda5b24ca1a01772fdd1ef4f690b646'
EXPECTED_BENCH='764914d779589895fea971860d110181b605004383dd49eacd973e81ab7c75cf'
@triton.jit
def value_threepart(P,V,S,O,L:tl.constexpr,GM:tl.constexpr):
    block=tl.program_id(0);g=tl.program_id(1)
    t=block*128+tl.arange(0,128);h=tl.arange(0,16);j=tl.arange(0,32)
    p=tl.load(P+h[:,None]*L+t[None,:],(h[:,None]<6)&(t[None,:]<L),0).to(tl.float32)
    if GM:
        s=tl.load(S+g*L+t,t<L,0).to(tl.float32)
        byte=tl.load(V+g*L*16+t[:,None]*16+j[None,:]//2,t[:,None]<L,0).to(tl.int32)
    else:
        s=tl.load(S+t*8+g,t<L,0).to(tl.float32)
        byte=tl.load(V+t[:,None]*128+g*16+j[None,:]//2,t[:,None]<L,0).to(tl.int32)
    code=(((byte>>((j[None,:]%2)*4))&15)-8).to(tl.float32)
    w=p*s[None,:]
    small=tl.sum(tl.sum(((tl.abs(w)>0)&(tl.abs(w)<1.0e-30)).to(tl.int32),1),0)>0
    # 1e-30 > 2^-100: conservative normal-component eligibility.
    if small:
        for hh in range(6):
            pp=tl.load(P+hh*L+t,t<L,0).to(tl.float32)
            yy=tl.sum((pp*s)[:,None]*code,0)
            tl.store(O+block*1536+hh*256+g*32+j,yy)
    else:
        w0=(w.to(tl.uint32,bitcast=True)&0xffffe000).to(tl.float32,bitcast=True)
        residual=w-w0
        w1=(residual.to(tl.uint32,bitcast=True)&0xffffe000).to(tl.float32,bitcast=True)
        w2=residual-w1
        y0=tl.dot(w0,code,input_precision='tf32')
        y1=tl.dot(w1,code,input_precision='tf32')
        y2=tl.dot(w2,code,input_precision='tf32')
        y=(y0+y1)+y2
        tl.store(O+block*1536+h[:,None]*256+g*32+j[None,:],y,h[:,None]<6)

def invoke(p,pk,s,gm):
    L=p.shape[1];partial=torch.empty(triton.cdiv(L,128),6,256,device='cuda')
    k=value_threepart[(triton.cdiv(L,128),8)](p,pk,s,partial,L,gm,num_warps=4,enable_fp_fusion=False)
    return partial.sum(0),k
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()==EXPECTED_BASE
    assert hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest()==EXPECTED_BENCH
    torch.manual_seed(1780);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=True
    # The existing SIMT/PyTorch baselines must still use IEEE FP32 matmul, while explicit
    # triton tl.dot(input_precision='tf32') consumes only already TF32-exact parts.
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=dict(status='RUNNING',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=EXPECTED_BASE,bench_sha256=EXPECTED_BENCH,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,
        representation='unchanged offset codes/scales; optional pure V byte permutation [L,8,16]->[8,L,16]',
        precision='P FP32; S FP16; w rounded FP32; exact w0+w1+w2 operands to explicit TF32 dot; FP32 reduction',
        fallback='any nonzero |w|<1e-30 in tile -> signed SIMT FP32; no coefficient dropping',
        heldout_lengths=[2048,16384,65536,262144],heldout_seeds=[1781,1782,1783],
        timing='resident graph; 32 operations; 31 randomized samples; eight warmups; not end-to-end',
        gate='gamma(8*128+2*ceil(L/128)+32)*absolute_term_envelope + 8*L*tiny_normal32; no 1e-6 floor',
        scope='conditional floating-point realization, not bitwise equality to decode-first or a universal hardware proof')
    def save():(out/'MANIFEST.json').write_text(json.dumps(cfg,indent=2))
    save();checks=[];kernels={}
    try:
        # Finite observed check complements the binary decomposition argument.
        w=torch.pow(2.,torch.randint(-99,16,(50000,),device='cuda'))*(torch.rand(50000,device='cuda')+1)
        w0=(w.view(torch.int32)&-8192).view(torch.float32);r=w-w0
        w1=(r.view(torch.int32)&-8192).view(torch.float32);w2=r-w1
        exact=bool(((w0.double()+w1.double()+w2.double())==w.double()).all())
        if not exact:raise RuntimeError('Operand decomposition is not value-preserving')
        cfg['observed_exact_decomposition_values']=50000;save()
        for L in [1,17,129,1024,8192]:
            for case in ['random','zero_code8','max_scale','tiny_scale','tiny_probability','zero_probability','alternating']:
                c=torch.randint(0,16,(L,8,32),device='cuda',dtype=torch.int32)
                if case=='zero_code8':c.fill_(8)
                if case=='alternating':c=(torch.arange(32,device='cuda')%2*15).expand(L,8,32).contiguous().int()
                s=(torch.rand(L,8,device='cuda')+.01).half()
                if case=='max_scale':s.fill_(65504)
                if case=='tiny_scale':s.fill_(2**-24)
                p=torch.softmax(torch.randn(6,L,device='cuda'),1)
                if case=='tiny_probability':p.fill_(2**-120)
                if case=='zero_probability':p.zero_()
                pk=base.pack(c);pg=pk.permute(1,0,2).contiguous();sg=s.T.contiguous()
                vd=((c.double()-8)*s.double()[:,:,None]).reshape(L,256);ref=p.double()@vd;env=p.double().abs()@vd.abs()
                u=torch.finfo(torch.float32).eps/2;n=8*128+2*triton.cdiv(L,128)+32
                bound=n*u/(1-n*u)*env+8*L*torch.finfo(torch.float32).tiny
                for gm,data,sc in [(False,pk,s),(True,pg,sg)]:
                    y,k=invoke(p,data,sc,gm);kernels[str(gm)]=k
                    err=(y.double()-ref).abs();ratio=torch.where(bound>0,err/bound,torch.where(err==0,0.,float('inf')))
                    checks.append(dict(L=L,case=case,groupmajor=gm,passed=bool(torch.isfinite(y).all() and (err<=bound).all()),
                        max_abs=err.max().item(),max_budget_ratio=ratio.max().item()))
        (out/'numerical.json').write_text(json.dumps(checks,indent=2))
        (out/'instruction_audit.json').write_text(json.dumps({n:base.audit(k) for n,k in kernels.items()},indent=2))
        for n,k in kernels.items():
            for kind in ['ptx','llir']:(out/(n+'.'+kind)).write_text(k.asm[kind])
        if not all(x['passed'] for x in checks):raise RuntimeError('Numerical contract failed; no timing promotion')
        print('V_THREEPART_GATE_PASS',len(checks),flush=True)
        rng=random.Random(1781);rows=[]
        for seed in cfg['heldout_seeds']:
            torch.manual_seed(seed)
            for L in cfg['heldout_lengths']:
                c=torch.randint(0,16,(L,8,32),device='cuda',dtype=torch.int32);pk=base.pack(c)
                s=(torch.rand(L,8,device='cuda')+.01).half();p=torch.softmax(torch.randn(6,L,device='cuda'),1)
                pg=pk.permute(1,0,2).contiguous();sg=s.T.contiguous();dense=base.decode(pk,s)
                funcs=dict(offset_simt=lambda:bench.invoke(p,pk,s,128,False)[0],
                    threepart_original=lambda:invoke(p,pk,s,False)[0],threepart_groupmajor=lambda:invoke(p,pg,sg,True)[0],
                    decode_first=lambda:p@base.decode(pk,s),predecoded_oracle=lambda:p@dense)
                timings=bench.measure(funcs,rng,p@dense)
                rows.append(dict(seed=seed,L=L,timings=timings))
                (out/'timings.json').write_text(json.dumps(rows,indent=2))
                print(json.dumps(dict(seed=seed,L=L,ms={n:v['median_ms'] for n,v in timings.items()})),flush=True)
        cfg.update(status='PASS',numerical_cases=len(checks),timing_rows=len(rows));save();print('V_THREEPART_PASS',out,flush=True)
    except Exception as e:cfg.update(status='FAIL',error=repr(e));save();raise
if __name__=='__main__':main()
