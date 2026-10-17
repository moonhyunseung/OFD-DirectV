from fractions import Fraction
import pytest
import torch
from upstream.directk.quant import PackedInt4,quantize_int4_groupwise
from reference.direct_value import decode_then_value,direct_value

@pytest.mark.parametrize('signed',[False,True])
@pytest.mark.parametrize('length',[0,1,17,129])
def test_geometry(length,signed):
    gen=torch.Generator().manual_seed(1729+length)
    # Quantizer cannot absmax an empty tensor through reshape(-1,...).
    v=quantize_int4_groupwise(torch.randn(max(length,1),256,generator=gen))
    if length==0:
        v=PackedInt4(v.packed[:0],v.scale[:0],(0,256),32)
    p=torch.softmax(torch.randn(6,length,generator=gen,dtype=torch.float64),dim=-1)
    torch.testing.assert_close(direct_value(p,v,signed=signed),decode_then_value(p,v),rtol=1e-12,atol=1e-12)

def test_independent_dyadic_oracle():
    # Bytes cover -8,7,0,-1; exact dyadic arithmetic in this small fixture.
    v=PackedInt4(torch.tensor([[[0xf0]],[[0x78]]],dtype=torch.uint8),torch.tensor([[.5],[2.]],dtype=torch.float16),(2,2),2)
    p=torch.tensor([[.25,.75]],dtype=torch.float64)
    expected=[sum(Fraction(a)*Fraction(s)*q for a,s,q in zip([.25,.75],[.5,2.],qs)) for qs in [(-8,0),(7,-1)]]
    for signed in [False,True]:
        assert direct_value(p,v,signed=signed).tolist()==[[float(x) for x in expected]]

def test_byte_domain():
    v=PackedInt4(torch.arange(256,dtype=torch.uint8).reshape(256,1,1),torch.ones(256,1,dtype=torch.float16),(256,2),2)
    p=torch.eye(256,dtype=torch.float64)
    assert torch.equal(direct_value(p,v,signed=True),decode_then_value(p,v))

def test_invalid_lengths():
    v=quantize_int4_groupwise(torch.ones(2,32))
    with pytest.raises(ValueError):direct_value(torch.ones(1,3),v)
