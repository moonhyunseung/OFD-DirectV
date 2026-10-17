"""DirectV tile selection on training shapes, then fixed-policy held-out evaluation.
No probability casting, quantizer change, token dropping or modified scales.
"""
import argparse,hashlib,json,random,statistics,math,subprocess
from pathlib import Path
import torch,triton
import h200_graph_boundary as base
EXPECTED='174ff55f19888523c8bdd671f601dda1ebda5b24ca1a01772fdd1ef4f690b646'
def invoke(p,pk,s,bt,signed):
    L=p.shape[1];nb=triton.cdiv(L,bt)
    partial=torch.empty(nb,6,256,device='cuda')
    k=base.directv_native[(nb,6,8)](p,pk,s,partial,L,signed,bt,enable_fp_fusion=False)
    return partial.sum(0),k

def measure(funcs,rng,ref):
    graphs={};outputs={};stream=torch.cuda.Stream()
    for name,fn in funcs.items():
        for _ in range(8):fn()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):
            for _ in range(32):y=fn()
        g.replay();torch.cuda.synchronize()
        torch.testing.assert_close(y,ref,atol=1e-4,rtol=2e-4)
        graphs[name]=g;outputs[name]=y
    samples={n:[] for n in graphs}
    for _ in range(31):
        names=list(graphs);rng.shuffle(names)
        for n in names:
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();graphs[n].replay();b.record();b.synchronize()
            samples[n].append(a.elapsed_time(b)/32)
    return {n:dict(median_ms=statistics.median(v),raw_ms=v) for n,v in samples.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()==EXPECTED
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.set_num_threads(1)
    cfg=dict(status='RUNNING',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=EXPECTED,
      selection='one global tile+signed flag, minimize geometric mean of train times; no holdout retuning',
      train_lengths=[1024,8192,131072],train_seed=1731,holdout_lengths=[2048,16384,65536,262144],holdout_seeds=[1741,1742,1743],
      candidates=[(b,s) for b in [64,128,256,512] for s in [False,True]],precision='FP32 P, FP16 scales, FP32 accumulation and output',
      timing='resident CUDA Graph; 32 ops per replay; 31 samples; 8 warmups; randomized method order',
      scope='one KV head,r=6,d=256,m=32; not serving throughput',gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__)
    def save():(out/'MANIFEST.json').write_text(json.dumps(cfg,indent=2))
    save();rng=random.Random(1731);torch.manual_seed(1730);checks=[];audits={}
    try:
        for L in [17,129,8192]:
            for case in ['random','zero_code8','max_scale','tiny_scale']:
                c=torch.randint(0,16,(L,8,32),device='cuda',dtype=torch.int32)
                if case=='zero_code8':c.fill_(8)
                s=(torch.rand(L,8,device='cuda')+.01).half()
                if case=='max_scale':s.fill_(65504)
                if case=='tiny_scale':s.fill_(2**-24)
                pk=base.pack(c);p=torch.softmax(torch.randn(6,L,device='cuda'),1)
                vd=((c.double()-8)*s.double()[:,:,None]).reshape(L,256);ref=p.double()@vd
                for bt,signed in cfg['candidates']:
                    y,k=invoke(p,pk,s,bt,signed);audits[str((bt,signed))]=base.audit(k)
                    env=p.double().abs()@((c.double().abs()+8)*s.double()[:,:,None]).reshape(L,256)
                    bound=(2*bt+2*triton.cdiv(L,bt)+32)*torch.finfo(torch.float32).eps*env+1e-6
                    err=(y.double()-ref).abs()
                    checks.append(dict(L=L,case=case,tile=bt,signed=signed,passed=bool(torch.isfinite(y).all() and (err<=bound).all()),max_abs=err.max().item(),max_budget_ratio=(err/bound).max().item()))
        (out/'numerical.json').write_text(json.dumps(checks,indent=2));(out/'instruction_audit.json').write_text(json.dumps(audits,indent=2))
        assert all(x['passed'] for x in checks),'Numerical diagnostic failed'
        train=[]
        torch.manual_seed(cfg['train_seed'])
        for L in cfg['train_lengths']:
            c=torch.randint(0,16,(L,8,32),device='cuda',dtype=torch.int32);pk=base.pack(c)
            s=(torch.rand(L,8,device='cuda')+.01).half();p=torch.softmax(torch.randn(6,L,device='cuda'),1)
            funcs={str((bt,sg)):(lambda bt=bt,sg=sg:invoke(p,pk,s,bt,sg)[0]) for bt,sg in cfg['candidates']}
            timings=measure(funcs,rng,p@base.decode(pk,s));train.append(dict(length=L,timings=timings))
        (out/'train.json').write_text(json.dumps(train,indent=2))
        scores={str((bt,sg)):math.exp(sum(math.log(r['timings'][str((bt,sg))]['median_ms']/r['timings'][str((128,False))]['median_ms']) for r in train)/len(train)) for bt,sg in cfg['candidates']}
        best=min(cfg['candidates'],key=lambda x:scores[str(tuple(x))]);cfg.update(selected=best,train_scores=scores);save()
        print('SELECTED',best,flush=True)
        held=[]
        for seed in cfg['holdout_seeds']:
            torch.manual_seed(seed)
            for L in cfg['holdout_lengths']:
                c=torch.randint(0,16,(L,8,32),device='cuda',dtype=torch.int32);pk=base.pack(c)
                s=(torch.rand(L,8,device='cuda')+.01).half();p=torch.softmax(torch.randn(6,L,device='cuda'),1);dense=base.decode(pk,s)
                funcs=dict(selected=lambda:invoke(p,pk,s,best[0],best[1])[0],
                    default_offset=lambda:invoke(p,pk,s,128,False)[0],default_signed=lambda:invoke(p,pk,s,128,True)[0],
                    decode_first=lambda:p@base.decode(pk,s),predecoded_oracle=lambda:p@dense)
                t=measure(funcs,rng,p@dense);held.append(dict(seed=seed,length=L,timings=t))
                (out/'heldout.json').write_text(json.dumps(held,indent=2))
                print(json.dumps(dict(seed=seed,L=L,ms={n:v['median_ms'] for n,v in t.items()})),flush=True)
        cfg.update(status='PASS',numerical_cases=len(checks),heldout_rows=len(held));save();print('V_TILE_PASS',out,flush=True)
    except Exception as e:cfg.update(status='FAIL',error=repr(e));save();raise
if __name__=='__main__':main()
