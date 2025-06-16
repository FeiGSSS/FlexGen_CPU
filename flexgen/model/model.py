import os
import numpy as np

from typing import Union, List, Optional

import torch
import torch.nn.functional as F

# import custom module
from flexgen.model.base import BaseModel
from flexgen.utils import (
    ExecutionEnv,
    Policy,
    ValueHolder,
    array_1d, array_2d, array_3d, array_4d,
    DUMMY_WEIGHT,
    Task
)
from flexgen.opt_config import OptConfig, get_opt_config, download_opt_weights
from flexgen.torch_backend import (
    TorchTensor,
    TorchDevice,
    TorchDisk,
    TorchMixedDevice,
    general_copy,
    DeviceType,
    AsyncIOManager
)
from flexgen.timer import timers




class InputEmbed(BaseModel):
    def __init__(
        self,
        config: OptConfig,
        env: ExecutionEnv,
        policy: Policy
    ):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = env.cpu
        self.weight_load_dst = self.compute
        
    def init_weight(self, weight_home: ValueHolder, path:str)-> None:
        """Load weights to DISK/CPU/(GPU) according to the policy."""
        vocab_size = self.config.vocab_size
        hidden_size = self.config.input_dim
        seq_len = self.config.max_seq_len
        dtype = self.config.dtype
        
        weight_specs = [
            # w_token
            ((vocab_size, hidden_size), dtype, path + "decoder.embed_tokens.weight"),
            # w_pos
            ((seq_len + 2, hidden_size), dtype, path + "decoder.embed_positions.weight"),
        ]
        
        weights = self.init_weight_list(weight_specs, self.policy, self.env)
        
        weight_home.store(weights)
        
    def load_weight(self,
                    weight_home: ValueHolder,
                    weight_read_buf: ValueHolder,
                    batch_idx: int) -> None:
        """Load weights from DISK/CPU/(GPU) to CPU/(GPU) according to the policy."""
        if batch_idx != 0: return
        w_token, w_pos = weight_home.val
        dst = self.weight_load_dst
        weight_read_buf.store((w_token.smart_copy(dst),
                               w_pos.smart_copy(dst)))
    
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self,
                hidden: ValueHolder,
                cache_read_buf: ValueHolder,
                weight_read_buf: ValueHolder,
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder,
                i: int,
                batch_idx: int) -> TorchTensor:
        if batch_idx == self.policy.num_batches - 1:
            # w_token, w_pos = weight_read_buf.pop()
            (w_token, w_token_del), (w_pos, w_pos_del) = weight_read_buf.pop()
        else:
            # w_token, w_pos = weight_read_buf.val
            (w_token, _), (w_pos, _) = weight_read_buf.val
        
        token_ids = hidden.val.data
        hidden.val.delete()
        
        _attention_mask, mask_del = attention_mask.val.smart_copy(self.compute)
        mask = _attention_mask.data
        if mask_del: attention_mask.val.delete()
        
        pad_token_id = self.config.pad_token_id
        
        # token embedding
        token_embed = F.embedding(token_ids, w_token.data, pad_token_id)
        
        # pos embedding
        positions = torch.cumsum(mask, dim=1).int() * mask + 1
        
        # cut positions if `past_key_values_length` is > 0
        past_key_values_length = mask.shape[1] - token_ids.shape[1]
        positions = positions[:, past_key_values_length:]
        
        pos_embed = F.embedding(positions, w_pos.data)
        
        if w_token_del: w_token.delete()
        if w_pos_del: w_pos.delete()
        
        data = token_embed + pos_embed
        
        hidden.val = TorchTensor.create_from_torch(data, self.compute)
        
        
class OutputEmbed(BaseModel):
    def __init__(
        self,
        config: OptConfig,
        env: ExecutionEnv,
        policy: Policy
    ):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = env.cpu
        self.weight_load_dst = self.compute
        
    def init_weight(self, weight_home: ValueHolder, path:str)-> None:
        """Load weights to DISK/CPU/(GPU) according to the policy."""
        vocab_size = self.config.vocab_size
        hidden_size = self.config.input_dim
        dtype = self.config.dtype
        
        weight_specs = [
            # w_ln
            ((hidden_size,), dtype, path + "decoder.layer_norm.weight"),
            # b_ln
            ((hidden_size,), dtype, path + "decoder.layer_norm.bias"),
            # w_token
            ((vocab_size, hidden_size), dtype, path + "decoder.embed_tokens.weight"),
        ]
        
        weights = self.init_weight_list(weight_specs, self.policy, self.env)
        
        weight_home.store(weights)
        
    def load_weight(self,
                    weight_home: ValueHolder,
                    weight_read_buf: ValueHolder,
                    batch_idx: int) -> None:
        """Load weights from DISK/CPU/(GPU) to CPU/(GPU) according to the policy."""
        if batch_idx != 0: return
        w_ln, b_ln, w_token = weight_home.val
        dst = self.weight_load_dst
        weight_read_buf.store((w_ln.smart_copy(dst),
                               b_ln.smart_copy(dst),
                               w_token.smart_copy(dst)))
        
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype
    
    def forward(self,
                hidden: ValueHolder,
                cache_read_buf: ValueHolder,
                weight_read_buf: ValueHolder,
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder,
                i: int,
                batch_idx: int) -> TorchTensor:
        
        if batch_idx == self.policy.num_batches - 1:
            (w_ln, w_ln_del), (b_ln, b_ln_del), (w_token, w_token_del) = weight_read_buf.pop()
        else:
            (w_ln, _), (b_ln, _), (w_token, _) = weight_read_buf.val

        hidden_dim = hidden.val.shape[-1]
        inputs = hidden.val
        
        x = F.layer_norm(inputs.data, (hidden_dim,), weight=w_ln.data, bias=b_ln.data)
        inputs.delete()
        
        # output embedding
        logits = F.linear(x, w_token.data)
        last_token_logits = logits[:,-1,:]
        
        do_sample = self.task.do_sample
        temperature = self.task.temperature
        
        if do_sample and not temperature < 1e-5:
            # Sample from the logits
            probs = F.softmax(last_token_logits / temperature, dim=-1)
            ids = torch.multinomial(probs, num_samples=1)
        else:
            ids = last_token_logits.argmax(dim=1, keepdim=True)
        
        hidden.val =  TorchTensor.create_from_torch(ids, self.compute)
        
        
class SelfAttention(BaseModel):
    def __init__(
        self,
        config: OptConfig,
        env: ExecutionEnv,
        policy: Policy,
        layer_idx: int
    ):
        self.config = config
        self.env = env
        self.policy = policy
        self.layer_idx = layer_idx
        self.compute = env.cpu
        self.weight_load_dst = self.compute
        self.attention_compute = self.compute
        
    def init_weight(self, weight_home: ValueHolder, path:str)-> None:
        """Load weights to DISK/CPU/(GPU) according to the policy."""
        h, dtype = self.config.input_dim, self.config.dtype
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_idx}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = self.init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)
    
    def load_weight(self,
                    weight_home: ValueHolder,
                    weight_read_buf: ValueHolder,
                    batch_idx: int) -> None:
        """Load weights from DISK/CPU/(GPU) to CPU/(GPU) according to the policy."""
        if batch_idx != 0: return
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        dst = self.weight_load_dst
        weight_read_buf.store((w_q.smart_copy(dst),
                               b_q.smart_copy(dst),
                               w_k.smart_copy(dst),
                               b_k.smart_copy(dst),
                               w_v.smart_copy(dst),
                               b_v.smart_copy(dst),
                               w_out.smart_copy(dst),
                               b_out.smart_copy(dst),
                               w_ln.smart_copy(dst),
                               b_ln.smart_copy(dst)))
        
    def init_cache_one_batch(self, cache_home: ValueHolder):
        device: Union[TorchDevice, TorchDisk, TorchMixedDevice] = None
        if self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed
            
        cache = device.init_cache_one_batch(self.config,
                                            self.task,
                                            self.policy)
        cache_home.store(cache)
        
    def load_cache(self,
                   cache_home: ValueHolder,
                   cache_read_buf: ValueHolder,
                   token_idx: int) -> None:
        
        if token_idx == 0: return # prefill phase
        
        k_home, v_home = cache_home.val
        dst = self.attention_compute
        k_buf, v_buf = dst.next_attention_compute_workspace()
        indices = (slice(0, self.task.prompt_len + token_idx - 1),
                       slice(0, k_home.shape[1]))
        general_copy(k_buf, indices, k_home, indices)
        general_copy(v_buf, indices, v_home, indices)
        cache_read_buf.store(((k_buf, False), (v_buf, False)))
        
    def store_cache(self,
                    cache_home: ValueHolder,
                    cache_write_buf: ValueHolder,
                    token_idx: int) -> None:
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()
        
        if token_idx == self.task.gen_len - 1: return
        
        if token_idx == 0:
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:
            pos = self.task.prompt_len + token_idx
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))
            
        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)
        
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype
    
    def mha(self,
            inputs: TorchTensor,
            attention_mask: TorchTensor,
            w_q: TorchTensor,
            b_q: TorchTensor,
            w_k: TorchTensor,
            b_k: TorchTensor,
            w_v: TorchTensor,
            b_v: TorchTensor,
            w_out: TorchTensor,
            b_out: TorchTensor,
            w_ln: TorchTensor,
            b_ln: TorchTensor,
            n_head: int,
            donate: list):
        """Multi-head attention (prefill phase)."""
        
        b, s, h = inputs.shape
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, s, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, s, n_head, head_dim)
        q = q.view(b, s, n_head, head_dim)
        k = k.view(b, s, n_head, head_dim)
        v = v.view(b, s, n_head, head_dim)

        # shape: (b * n_head, s, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)
        # shape: (b * n_head, head_dim, s)
        k = k.permute(0, 2, 3, 1).reshape(b * n_head, head_dim, s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(0, 2, 1, 3).reshape(b * n_head, s, head_dim)

        # shape: (b * n_head, s, s)
        attn_weights = torch.bmm(q, k)

        # shape: (b, 1, s, s)
        idx = torch.arange(s, device=torch.device("cpu")) # TODO: not sure if this is correct
        causal_mask = (idx <= idx.view(s, 1)).view(1, 1, s, s)
        mask = attention_mask.data.view(b, 1, 1, s) & causal_mask

        # shape: (b, n_head, s, s)
        attn_weights = attn_weights.view(b, n_head, s, s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, s, s)
        attn_weights = F.softmax(attn_weights, dim=2)
        # shape: (b, n_head, s, head_dim)
        value = torch.bmm(attn_weights, v).view(b, n_head, s, head_dim)
        # shape: (b, s, h)
        value = value.transpose(1, 2).reshape(b, s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        # (s, b * n_head, head_dim)
        k = k.permute(2, 0, 1)
        v = v.permute(1, 0, 2)

        k = TorchTensor.create_from_torch(k, self.compute)
        v = TorchTensor.create_from_torch(v, self.compute)

        return TorchTensor.create_from_torch(value, self.compute), k, v

    def mha_gen(self,
                inputs: TorchTensor,
                attention_mask: TorchTensor,
                w_q: TorchTensor,
                b_q: TorchTensor,
                w_k: TorchTensor,
                b_k: TorchTensor,
                w_v: TorchTensor,
                b_v: TorchTensor,
                w_out: TorchTensor,
                b_out: TorchTensor,
                w_ln: TorchTensor,
                b_ln: TorchTensor,
                n_head: int,
                k_cache: TorchTensor,
                v_cache: TorchTensor,
                donate: list):
        """Multi-head attention (decoding phase)."""
        b, tgt_s, h = inputs.shape
        src_s = attention_mask.shape[1]
        head_dim = h // n_head
        scaling = head_dim ** -0.5

        hidden = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)

        # shape: (b, 1, h)
        q = F.linear(hidden, w_q.data, bias=b_q.data) * scaling
        k = F.linear(hidden, w_k.data, bias=b_k.data)
        v = F.linear(hidden, w_v.data, bias=b_v.data)
        # shape: (b, 1, n_head, head_dim)
        q = q.view(b, tgt_s, n_head, head_dim)
        k = k.view(b, tgt_s, n_head, head_dim)
        v = v.view(b, tgt_s, n_head, head_dim)

        # shape: (b * n_head, 1, head_dim)
        q = q.permute(0, 2, 1, 3).reshape(b * n_head, tgt_s, head_dim)
        # shape: (1, b * n_head, head_dim)
        k_new = k.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)
        # shape: (1, b * n_head, head_dim)
        v_new = v.permute(1, 0, 2, 3).reshape(tgt_s, b * n_head, head_dim)

        # shape: (s, b * n_head, head_dim)
        k = k_cache.data[:src_s]
        v = v_cache.data[:src_s]
        k[src_s - 1:src_s] = k_new
        v[src_s - 1:src_s] = v_new

        # shape: (b * n_head, head_dim, s)
        k = k.permute(1, 2, 0).reshape(b * n_head, head_dim, src_s)
        # shape: (b * n_head, s, head_dim)
        v = v.permute(1, 0, 2).reshape(b * n_head, src_s, head_dim)

        value = self._attention_value(q, k, v, attention_mask.data, b, src_s, tgt_s, n_head, head_dim)

        # shape: (b, 1, h)
        value = value.transpose(1, 2).view(b, tgt_s, h)
        value = F.linear(value, w_out.data, bias=b_out.data)

        value.add_(inputs.data)

        if donate[0]: inputs.delete()
        if donate[1]: attention_mask.delete()

        k_new = TorchTensor.create_from_torch(k_new, self.compute)
        v_new = TorchTensor.create_from_torch(v_new, self.compute)

        return TorchTensor.create_from_torch(value, self.compute), k_new, v_new

    def _attention_weights(self, q, k, mask, b, src_s, n_head):
        # shape: (b * n_head, 1, s)
        attn_weights = torch.bmm(q, k)
        # shape: (b, 1, 1, s)
        mask = mask.view(b, 1, 1, src_s)
        # shape: (b * n_head, 1, s)
        attn_weights = attn_weights.view(b, n_head, 1, src_s)
        attn_weights = torch.where(mask, attn_weights, -1e4)
        attn_weights = attn_weights.view(b * n_head, 1, src_s)
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        # shape: (b * n_head, 1, s)
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)
        # shape: (b, n_head, 1, head_dim)
        return torch.bmm(attn_weights, v).view(b, n_head, tgt_s, head_dim)
    
    def forward(self,
                hidden: ValueHolder,
                cache_read_buf: ValueHolder,
                weight_read_buf: ValueHolder,
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder,
                token_idx: int,
                batch_idx: int) -> None:
        n_head = self.config.n_head
        
        donate = [False] * 14
        h, donate[0] = hidden.val, True

        if batch_idx == self.policy.num_batches - 1:
            # Clear the weight_read_buf if it is the last batch
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
             (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        if token_idx == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.mha(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.mha_gen(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, donate)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h
        
        
        
class MLP(BaseModel):
    def __init__(self,
                 config: OptConfig,
                 env: ExecutionEnv,
                 policy: Policy,
                 layer_idx: int):
        self.config = config
        self.env = env
        self.policy = policy
        self.layer_idx = layer_idx
        self.compute = env.cpu
        self.weight_load_dst = self.compute
        
    def init_weight(self, weight_home: ValueHolder, path:str)-> None:
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_idx}."))
        weight_specs = [
            # wi
            ((4 * h, h), dtype, path + "fc1.weight"),
            # bi
            ((4 * h,), dtype, path + "fc1.bias"),
            # wo
            ((h, 4 * h), dtype, path + "fc2.weight"),
            # bo
            ((h,), dtype, path + "fc2.bias"),
            # w_ln
            ((h,), dtype, path + "final_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "final_layer_norm.bias"),
        ]
        weights = self.init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self,
                    weight_home: ValueHolder,
                    weight_read_buf: ValueHolder,
                    batch_idx: int) -> None:
        """Load weights from DISK/CPU/(GPU) to CPU/(GPU) according to the policy."""
        wi, bi, wo, bo, w_ln, b_ln = weight_home.val
        if batch_idx == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                wi.smart_copy(dst1), bi.smart_copy(dst2),
                wo.smart_copy(dst1), bo.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))
    
    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype
    
    def mlp(self, inputs, wi, bi, wo, bo, w_ln, b_ln, donate):
        b, s, h = inputs.shape

        out = F.layer_norm(inputs.data, (h,), weight=w_ln.data, bias=b_ln.data)
        out = F.linear(out, wi.data, bias=bi.data)
        F.relu(out, inplace=True)
        out = F.linear(out, wo.data, bias=bo.data)

        out.add_(inputs.data)
        if donate[0]: inputs.delete()
        return TorchTensor.create_from_torch(out, self.compute)
    
    def forward(self,
                hidden: ValueHolder,
                cache_read_buf: ValueHolder,
                weight_read_buf: ValueHolder,
                attention_mask: ValueHolder,
                cache_write_buf: ValueHolder,
                token_idx: int,
                batch_idx: int) -> None:
        donate = [False] * 7
        h, donate[0] = hidden.val, True

        if batch_idx == self.policy.num_batches - 1:
            ((wi, donate[1]), (bi, donate[2]), (wo, donate[3]), (bo, donate[4]),
             (w_ln, donate[5]), (b_ln, donate[6])) = weight_read_buf.pop()
        else:
            ((wi, _), (bi, _), (wo, _), (bo, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        h = self.mlp(h, wi, bi, wo, bo, w_ln, b_ln, donate)
        hidden.val = h
        
class TransformerLayer(BaseModel):
    """Transformer layer that combines self-attention and MLP."""
    def __init__(self,
                 config:OptConfig,
                 env: ExecutionEnv,
                 policy: Policy,
                 layer_idx: int):
        self.attention = SelfAttention(config, env, policy, layer_idx)
        self.mlp = MLP(config, env, policy, layer_idx)
        self.policy = policy
        self.compute = self.attention.compute

    def set_task(self, task):
        self.attention.set_task(task)
        self.mlp.set_task(task)

    def init_weight(self, weight_home, path):
        home1, home2 = ValueHolder(), ValueHolder()
        self.attention.init_weight(home1, path)
        self.mlp.init_weight(home2, path)
        weight_home.store((home1, home2))

    def load_weight(self, weight_home, weight_read_buf, batch_idx):
        read_buf1, read_buf2 = ValueHolder(), ValueHolder()
        home1, home2 = weight_home.val
        self.attention.load_weight(home1, read_buf1, batch_idx)
        self.mlp.load_weight(home2, read_buf2, batch_idx)
        if batch_idx == 0:
            weight_read_buf.store((read_buf1, read_buf2))

    def init_cache_one_batch(self, cache_home):
        self.attention.init_cache_one_batch(cache_home)

    def load_cache(self, cache_home, cache_read_buf, token_idx):
        self.attention.load_cache(cache_home, cache_read_buf, token_idx)

    def store_cache(self, cache_home, cache_write_buf, token_idx):
        self.attention.store_cache(cache_home, cache_write_buf, token_idx)

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, token_idx, batch_idx):
        if batch_idx == self.policy.num_batches - 1:
            read_buf1, read_buf2 = weight_read_buf.pop()
        else:
            read_buf1, read_buf2 = weight_read_buf.val

        self.attention.forward(hidden, cache_read_buf, read_buf1, attention_mask,
                               cache_write_buf, token_idx, batch_idx)
        self.mlp.forward(hidden, None, read_buf2, attention_mask, None, token_idx, batch_idx)
        
import concurrent.futures
from threading import Lock

class CPUStreamExecutor:
    """A simple CPU stream executor to simulate CUDA streams functionality."""

    def __init__(self, max_workers=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        self.lock = Lock()
        
    def submit(self, fn, *args, **kwargs):
        """Submit a task to the thread pool for execution."""
        future = self.executor.submit(fn, *args, **kwargs)
        with self.lock:
            self.futures.append(future)
        return future
    
    def synchronize(self):
        """Wait for all tasks to complete, similar to cuda.synchronize()"""
        with self.lock:
            futures = self.futures.copy()
            self.futures = []

        # Wait for all tasks to complete
        for future in futures:
            future.result()

    def __del__(self):
        self.executor.shutdown(wait=True)


class OptLM:
    def __init__(self,
                 config: Union[str, OptConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy):
        if isinstance(config, str):
            config = get_opt_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_batches = policy.num_batches
        
        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i))
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)
        
        if self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()
        
        # IO streams
        self.load_weight_stream = CPUStreamExecutor()
        self.load_cache_stream = CPUStreamExecutor()
        self.store_cache_stream = CPUStreamExecutor()
        
        self.async_io_manager = AsyncIOManager()

        num_layers, num_batches = self.num_layers, self.policy.num_batches
        
        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_batches, ValueHolder)
        # weight[j]
        self.weight_home = array_1d(num_layers, ValueHolder)
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_batches, ValueHolder)

        self.task = None
        for j in range(self.num_layers):
            self.init_weight(j)
            
    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)
            
    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "decoder.embed_positions.weight")
        if not os.path.exists(check_path) and DUMMY_WEIGHT not in check_path:
            download_opt_weights(self.config.name, self.path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)
        
    def load_weight(self,
                    token_idx: int,
                    layer_idx: int,
                    batch_idx: int,
                    overlap: bool=True):
        # Handle corner cases
        if layer_idx == self.num_layers:
            layer_idx = 0
            token_idx += 1
            if token_idx == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            self.load_weight_stream.submit(
                self.layers[layer_idx].load_weight,
                self.weight_home[layer_idx],
                self.weight_read_buf[layer_idx],
                batch_idx
            )
        else:
            self.layers[layer_idx].load_weight(self.weight_home[layer_idx],
                                               self.weight_read_buf[layer_idx],
                                               batch_idx)
            
    def delete_weight(self, layer_idx:int, batch_idx:int):
        if batch_idx == 0:
            for x in self.weight_home[layer_idx].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()
                    
    def init_cache(self, layer_idx: int, batch_idx: int):
        """Initialize cache for a layer and a batch."""
        self.layers[layer_idx].init_cache_one_batch(self.cache_home[layer_idx][batch_idx])
        
    def load_cache(self, token_idx, layer_idx, batch_idx, overlap=True):
        # Handle corner cases
        if token_idx == 0:  return
        if batch_idx == self.num_batches:
            batch_idx = 0
            layer_idx += 1
        if layer_idx == self.num_layers:
            layer_idx = 0
            token_idx += 1
            if token_idx == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            self.load_cache_stream.submit(
                self.layers[layer_idx].load_cache,
                self.cache_home[layer_idx][batch_idx],
                self.cache_read_buf[layer_idx][batch_idx],
                token_idx
            )
        else:
            self.layers[layer_idx].load_cache(self.cache_home[layer_idx][batch_idx],
                                              self.cache_read_buf[layer_idx][batch_idx],
                                              token_idx)
    def store_cache(self, token_idx, layer_idx, batch_idx, overlap=True):
        # Handle corner cases
        if batch_idx == -1:
            batch_idx = self.num_batches - 1
            layer_idx -= 1
        if layer_idx == -1:
            layer_idx = self.num_layers - 1
            token_idx -= 1
            if token_idx == -1:
                return
        if token_idx == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[layer_idx][batch_idx].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            self.store_cache_stream.submit(
                self.layers[layer_idx].store_cache,
                self.cache_home[layer_idx][batch_idx],
                self.cache_write_buf[layer_idx][batch_idx],
                token_idx
            )
        else:
            self.layers[layer_idx].store_cache(self.cache_home[layer_idx][batch_idx],
                                               self.cache_write_buf[layer_idx][batch_idx],
                                               token_idx)
    def delete_cache(self, layer_idx: int, batch_idx: int):
        v = self.cache_home[layer_idx][batch_idx].pop()
        if v:
            for x in v:
                x.delete()
                
    
    def load_hidden(self, token_idx, layer_idx, batch_idx):
        # Handle corner cases
        if batch_idx == self.num_batches:
            batch_idx = 0
            layer_idx += 1
        if layer_idx == self.num_layers:
            layer_idx = 0
            token_idx += 1
            if token_idx == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[layer_idx].compute
        if layer_idx == 0:
            batch_size = self.policy.batch_size
            left, right = batch_idx * batch_size, (batch_idx + 1) * batch_size
            if token_idx == 0:  # load from the input ids
                val = dst.allocate((batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + token_idx
                val = dst.allocate((batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[token_idx][layer_idx-1][batch_idx].pop().move(dst)
        self.hidden[token_idx][layer_idx][batch_idx].store(val)

    def store_hidden(self, token_idx, layer_idx, batch_idx):
        # Handle corner cases
        if batch_idx == -1:
            batch_idx = self.num_batches - 1
            layer_idx -= 1
        if layer_idx == -1:
            layer_idx = self.num_layers - 1
            token_idx -= 1
            if token_idx == -1:
                return

        # Store to hidden states buffers
        if layer_idx == self.num_layers - 1:  # store to output
            batch_size = self.policy.batch_size
            left, right = batch_idx * batch_size, (batch_idx + 1) * batch_size
            ids = self.hidden[token_idx][layer_idx][batch_idx].pop().data.detach().numpy()
            pos = self.task.prompt_len + token_idx
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[token_idx][layer_idx][batch_idx]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    def compute_layer(self, token_idx, layer_idx, batch_idx):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[layer_idx].forward(self.hidden[token_idx][layer_idx][batch_idx],
                                       self.cache_read_buf[layer_idx][batch_idx],
                                       self.weight_read_buf[layer_idx],
                                       self.attention_mask[batch_idx],
                                       self.cache_write_buf[layer_idx][batch_idx],
                                       token_idx,
                                       batch_idx)
        
    
    def sync(self):
        """Synchronize all streams."""
        self.load_weight_stream.synchronize()
        self.load_cache_stream.synchronize()
        self.store_cache_stream.synchronize()
        self.async_io_manager.synchronize()
        
    def update_attention_mask(self, token_idx, batch_idx):
        """Update the attention mask for the given token index and batch index."""
        if token_idx > 0:
            mask = self.attention_mask[batch_idx]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        batch_size = self.policy.batch_size
        left = batch_idx * batch_size
        right = left + batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = self.env.cpu
        val = attention_compute.allocate((self.policy.batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[batch_idx].store(val)
        
        
    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        
        num_layers = self.num_layers
        num_batches = self.num_batches
        batch_size = self.policy.batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len
        
        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
                                  self.config.pad_token_id,
                                  dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert batch_size * num_batches == len(task.inputs)
        
        for j in range(num_layers):
            for k in range(num_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
                
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
            
        for k in range(num_batches):
            self.attention_mask[k].clear()
            
        self.hidden = array_3d(gen_len, num_layers, num_batches, ValueHolder)
        
        self.set_task(task)
        
        for j in range(num_layers):
            for k in range(num_batches):
                self.init_cache(j, k)
        
        self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)
        
        if not overlap:
            # No overlap, easy to understand, suitable for debugging
            self.generation_loop_normal()
        else:
            # Overlap I/O and compute
            if num_batches == 1:
                self.generation_loop_overlap_single_batch()
            else:
                self.generation_loop_overlap_multi_batch()
        
        # Delete cache
        for j in range(num_layers):
            for k in range(num_batches):
                self.delete_cache(j, k)
                
        self.env.cpu.del_attention_compute_workspace()

        return self.output_ids
    
    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.sync()
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
                    self.sync()
            timers("generate").stop()

    
    def generation_loop_overlap_single_batch(self):
        # Prologue
        for k in range(self.num_batches):
            self.load_weight(0, 0, k)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            self.update_attention_mask(i, 0)
            for j in range(self.num_layers):
                self.load_weight(i, j+1, 0)
                self.load_cache(i, j+1, 0)
                self.load_hidden(i, j, 0)
                self.compute_layer(i, j, 0)
                self.store_cache(i, j-1, 0)
                self.store_hidden(i, j, 0)
                self.sync()
            timers("generate").stop()

            if self.task.stop and np.all(self.stopped):
                break

    def generation_loop_overlap_multi_batch(self):
        # Prologue
        for k in range(self.num_gpu_batches):
            self.load_weight(0, 0, k)
        self.load_hidden(0, 0, 0)
        self.sync()

        # Generate
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j+1, k)
                    self.load_cache(i, j, k+1) 
                    self.store_hidden(i, j, k-1)
                    self.load_hidden(i, j, k+1)
                    self.compute_layer(i, j, k)
                    self.store_cache(i, j, k-1)
                    self.sync()
            timers("generate").stop()

        # Epilogue
        self.store_hidden(
            self.execute_gen_len-1, self.num_layers-1, self.num_gpu_batches-1)