import torch
from upstream.signedint4.codec import recode_bytes,unpack
from upstream.directk.quant import offset_bytes_to_twos_complement

def test_tensor_codec_matches_pinned_integer_codec():
    raw=bytes(range(256))
    actual=bytes(offset_bytes_to_twos_complement(torch.arange(256,dtype=torch.uint8)).tolist())
    assert actual==recode_bytes(raw)
    assert unpack(raw)==unpack(actual,signed=True)
