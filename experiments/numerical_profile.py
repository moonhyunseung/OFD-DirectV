"""CPU characterization; never a production tolerance or speed benchmark."""
import csv,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from upstream.directk.quant import PackedInt4,quantize_int4_groupwise,unpack_codes
from reference.direct_value import decode_then_value,direct_value
SIDE='V'

def run(out):
    torch.set_num_threads(1)
    rows=[]
    for dtype in [torch.float64,torch.float32]:
      for length in [1,17,129,512]:
       for case in ['random','endpoints','offset_cancellation','min_half_scale','max_half_scale','zero_scale']:
        gen=torch.Generator().manual_seed(1729+length)
        v=quantize_int4_groupwise(torch.randn(length,256,generator=gen))
        if case=='endpoints':v=PackedInt4(torch.full_like(v.packed,0xf0),v.scale,v.orig_shape,32)
        if case=='offset_cancellation':v=PackedInt4(torch.full_like(v.packed,0x88),v.scale,v.orig_shape,32)
        if case.endswith('_scale'):
            scale={'min_half_scale':2**-24,'max_half_scale':65504,'zero_scale':0.}[case]
            v=PackedInt4(v.packed,torch.full_like(v.scale,scale),v.orig_shape,32)
        x=torch.randn(6,256 if SIDE=='K' else length,generator=gen,dtype=torch.float32).to(dtype)
        if SIDE=='V':x=torch.softmax(x,dim=-1)
        c=unpack_codes(v).to(torch.float64);s=v.scale.to(torch.float64)
        if SIDE=='K':
            ref=decode_then_score(x,v,1/16,out_dtype=dtype)
            paths=[('offset',direct_packed_score(x,v,1/16,compute_dtype=dtype)),('signed_bytes',direct_signed_bytes_score(x,v,1/16,compute_dtype=dtype))]
            magnitude=torch.einsum('hgj,tgj->ht',x.to(torch.float64).reshape(6,8,32).abs(),(c+8)*s[:,:,None])/16
        else:
            ref=decode_then_value(x,v,compute_dtype=dtype)
            paths=[('offset',direct_value(x,v,compute_dtype=dtype)),('signed_bytes',direct_value(x,v,compute_dtype=dtype,signed=True))]
            magnitude=torch.einsum('ht,tgj->hgj',x.to(torch.float64).abs(),(c+8)*s[:,:,None]).reshape(6,256)
        for name,y in paths:
            e=(y-ref).to(torch.float64);rn=torch.linalg.vector_norm(ref.to(torch.float64)).item();en=torch.linalg.vector_norm(e).item()
            rows.append(dict(case=case,length=length,r=6,d=256,group_size=32,scale_dtype='float16',input_dtype='float32_cast_to_compute',compute_dtype=str(dtype),path=name,max_abs=e.abs().max().item(),relative_l2=en/rn if rn else (0. if en==0 else 'undefined_zero_reference'),relative_l2_defined=bool(rn or not en),error_over_abs_term_envelope=en/max(torch.linalg.vector_norm(magnitude).item(),1e-300),finite=bool(torch.isfinite(y).all()),seed=1729+length))
    with (out/'numerical.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return dict(rows=len(rows),all_finite=all(r['finite'] for r in rows),max_abs=max(r['max_abs'] for r in rows),max_error_over_abs_term_envelope=max(r['error_over_abs_term_envelope'] for r in rows))
