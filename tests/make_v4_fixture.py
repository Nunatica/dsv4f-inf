#!/usr/bin/env python3
"""Deterministic small V4 fixture with hash/text/vision routes, for integration tests."""
import array
import ctypes as C
from pathlib import Path
import random
import struct
import sys

root = Path(__file__).resolve().parents[1]
lib = C.CDLL(str(Path(sys.argv[2]) if len(sys.argv)>2 else root/'build/bin/libggml-base.so'))
lib.ggml_quantize_chunk.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_longlong, C.c_longlong, C.c_longlong, C.c_void_p]
lib.ggml_quantize_chunk.restype = C.c_size_t
rng = random.Random(41)
D, FF, DH, NH, RQ, OG, OL, NE, V, L = 512, 512, 256, 2, 128, 2, 128, 8, 32, 3
meta = {'general.architecture':'deepseek4', 'general.name':'V4 RAM/pruning test fixture',
    'deepseek4.block_count':L, 'deepseek4.context_length':128, 'deepseek4.embedding_length':D,
    'deepseek4.attention.head_count':NH, 'deepseek4.attention.head_count_kv':1,
    'deepseek4.attention.key_length':DH, 'deepseek4.attention.value_length':DH,
    'deepseek4.rope.dimension_count':64, 'deepseek4.rope.freq_base':10000.,
    'deepseek4.attention.layer_norm_rms_epsilon':1e-6, 'deepseek4.expert_count':NE,
    'deepseek4.expert_used_count':2, 'deepseek4.expert_gating_func':4,
    'deepseek4.attention.q_lora_rank':RQ, 'deepseek4.attention.sliding_window':128,
    'deepseek4.expert_feed_forward_length':FF, 'deepseek4.expert_shared_count':1,
    'deepseek4.expert_weights_scale':.1, 'deepseek4.expert_weights_norm':True,
    'deepseek4.swiglu_clamp_exp':[1.25]*L, 'deepseek4.attention.indexer.head_count':2,
    'deepseek4.attention.indexer.key_length':128, 'deepseek4.attention.indexer.top_k':8,
    'deepseek4.attention.output_group_count':OG, 'deepseek4.attention.output_lora_rank':OL,
    'deepseek4.attention.compress_ratios':[0,4,128], 'deepseek4.attention.compress_rope_freq_base':160000.,
    'deepseek4.hyper_connection.count':4, 'deepseek4.hyper_connection.sinkhorn_iterations':4,
    'deepseek4.hyper_connection.epsilon':1e-6, 'deepseek4.hash_layer_count':1,
    'tokenizer.ggml.model':'gpt2', 'tokenizer.ggml.pre':'joyai-llm',
    'tokenizer.ggml.tokens':['<bos>','<eos>']+[f'tok{i}' for i in range(2,V)],
    'tokenizer.ggml.token_type':[3,3]+[1]*(V-2), 'tokenizer.ggml.merges':[],
    'tokenizer.ggml.bos_token_id':0, 'tokenizer.ggml.eos_token_id':1,
    'tokenizer.ggml.add_bos_token':False}
tensors = []
def add(name, shape, typ=0, fill=None):
    n = 1
    for dim in shape: n *= dim
    if typ == 26:
        data = struct.pack('<'+'i'*n, *fill)
    else:
        values = array.array('f', (rng.uniform(-.04,.04) for _ in range(n)) if fill is None else
                             ([fill]*n if isinstance(fill, (int,float)) else fill))
        if typ == 0:
            data = values.tobytes()
        else:
            out = C.create_string_buffer(n//32*17)
            wrote = lib.ggml_quantize_chunk(typ, C.addressof((C.c_float*n).from_buffer(values)), C.addressof(out), 0, n//shape[0], shape[0], None)
            assert wrote == len(out)
            data = out.raw
    tensors.append((name, shape, typ, data))
add('token_embd.weight',[D,V]);add('output.weight',[D,V]);add('output_norm.weight',[D],fill=1)
add('output_hc_fn.weight',[D*4,4]);add('output_hc_base.weight',[4],fill=0);add('output_hc_scale.weight',[1],fill=1)
for l in range(L):
    b = f'blk.{l}.'
    for site in ['attn','ffn']:
        add(b+f'hc_{site}_fn.weight',[D*4,24]);add(b+f'hc_{site}_base.weight',[24],fill=0)
        add(b+f'hc_{site}_scale.weight',[3],fill=1);add(b+site+'_norm.weight',[D],fill=1)
    for name,shape,fill in [('attn_sinks.weight',[NH],0),('attn_q_a.weight',[D,RQ],None),
        ('attn_q_a_norm.weight',[RQ],1),('attn_q_b.weight',[RQ,NH*DH],None),
        ('attn_kv.weight',[D,DH],None),('attn_kv_a_norm.weight',[DH],1),
        ('attn_output_a.weight',[NH*DH//OG,OL*OG],None),('attn_output_b.weight',[OG*OL,D],None),
        ('ffn_gate_inp.weight',[D,NE],None),('exp_probs_b_vl.bias',[NE],list(reversed(range(NE))))]:
        add(b+name,shape,fill=fill)
    if l:
        ratio = 4 if l == 1 else 128
        coff = 2 if ratio == 4 else 1
        add(b+'attn_compressor_kv.weight',[D,coff*DH]);add(b+'attn_compressor_gate.weight',[D,coff*DH])
        add(b+'attn_compressor_ape.weight',[coff*DH,ratio]);add(b+'attn_compressor_norm.weight',[DH],fill=1)
        if ratio == 4:
            add(b+'indexer.proj.weight',[D,2]);add(b+'indexer.attn_q_b.weight',[RQ,256])
            add(b+'indexer_compressor_kv.weight',[D,256]);add(b+'indexer_compressor_gate.weight',[D,256])
            add(b+'indexer_compressor_ape.weight',[256,4]);add(b+'indexer_compressor_norm.weight',[128],fill=1)
    if l == 0: add(b+'ffn_gate_tid2eid.weight',[2,V],26,[i%NE for i in range(2*V)])
    else: add(b+'exp_probs_b.bias',[NE],fill=list(range(NE)))
    for role,shape in [('gate',[D,FF]),('up',[D,FF]),('down',[FF,D])]:
        add(b+f'ffn_{role}_exps.weight',shape+[NE],39)
        add(b+f'ffn_{role}_shexp.weight',shape)
def string(value):
    b = value.encode();return struct.pack('<Q',len(b))+b
def kv(k,v):
    if isinstance(v,bool): typ,data=7,struct.pack('<?',v)
    elif isinstance(v,int): typ,data=4,struct.pack('<I',v)
    elif isinstance(v,float): typ,data=6,struct.pack('<f',v)
    elif isinstance(v,str): typ,data=8,string(v)
    else:
        typ=9;et=8 if k.endswith(('tokens','merges')) else 5 if k.endswith('token_type') else 6 if v and isinstance(v[0],float) else 4
        data=struct.pack('<IQ',et,len(v))+b''.join(string(x) if et==8 else struct.pack('<f' if et==6 else '<i' if et==5 else '<I',x) for x in v)
    return string(k)+struct.pack('<I',typ)+data
def write(path, meta, tensors):
    head=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(meta))+b''.join(kv(k,v) for k,v in meta.items())
    payload=bytearray()
    for name,shape,typ,data in tensors:
        head+=string(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)+struct.pack('<IQ',typ,len(payload))
        payload+=data;payload+=bytes(-len(payload)%32)
    head+=bytes(-len(head)%32)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(head+payload)
    print(path)
path=Path(sys.argv[1]);write(path,meta,tensors)
# A separate MTP-only GGUF has no trunk layers. Pruning must leave it intact.
main_tensors=tensors
meta={**meta, 'general.name':'V4 MTP loading fixture', 'deepseek4.block_count':L+1,
      'deepseek4.nextn_predict_layers':1, 'deepseek4.swiglu_clamp_exp':[1.25]*(L+1),
      'deepseek4.attention.compress_ratios':[0,4,128,0]}
tensors=[t for t in main_tensors if not t[0].startswith('blk.')]
for name,shape,typ,data in main_tensors:
    if name.startswith('blk.2.') and 'compressor' not in name:
        tensors.append((name.replace('blk.2.','blk.3.',1),shape,typ,data))
add('blk.3.nextn.eh_proj.weight',[2*D,D]);add('blk.3.nextn.enorm.weight',[D],fill=1);add('blk.3.nextn.hnorm.weight',[D],fill=1)
write(path.with_name('V4-MTP.gguf'),meta,tensors)
