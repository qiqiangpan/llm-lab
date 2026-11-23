# Implementing a KV Cache from Scratch
- Original author's artical: https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms

## 1. Registering the Cache Buffers
- Inside the `MultiHeadAttention` constructor, we add two buffers, `cache_k` and `cache_v`, which will hold concatenated keys and values across steps:

```python
self.register_buffer("cache_k", None, persistent=False)
self.register_buffer("cache_v", None, persistent=False)
self.ptr_current_pos = 0
```

> **Note**
> 1. `persistent=False`: `persistent` 参数控制是否将 buffer 保存进模型 state_dict（checkpoint 权重文件）
> 2. `self.ptr_current_pos = 0`: `ptr_current_pos` 参数用于记录当前 cache 中已经存了多少个 token 的 k/v

## 2. Forward pass with `use_cache` flag
- Next, we extend the `forward` method of the `MultiHeadAttention` class to accept a `use_cache` argument:

```python
def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape # [batch_size, seq_len, d_in]

        keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        # NEW
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new
        ####################################################

        # ...
```

> **Note**
> 1. 第一次用户输入 prompt 时，`queries` 是 prompt 最后一个 token 对应的查询向量吗？ `key_new` 和 `values_new` 也分别是最后一个 token 对应的键向量和值向量吗？  
>> - `if self.cache_k is None` : 第一次用户输入 prompt 时为 Prefill 阶段，此时的 `queries` 、`key_new` 和 `values_new` 是 prompt 中所有 tokens 对应的查询向量、键向量和值向量。`queries` 、`key_new` 和 `values_new` 的 shape 为 `[b, num_tokens, num_heads, head_dim]`
>> - `else`: 后续模型生成新 token 时为 Decode 阶段，此时 `queries` 、`key_new` 和 `values_new` 是新生成的这**一个** token 对应的查询向量、键向量和值向量。`queries` 、`key_new` 和 `values_new` 的 shape 为 `[b, 1, num_heads, head_dim]`
> 
> 2. the core store & retrieve mechanism of a KV cache
>> - Storing: Concretely, after the cache is initialized via the `if self.cache_k is None: ...`, we add the newly generated keys and values via `self.cache_k = torch.cat(...)` and `self.cache_v = torch.cat(...)` to the cache, respectively.
>> - Retrieving: Then, `keys, values = self.cache_k, self.cache_v` retrieves the stored values and keys from the cache.

## 3. Clearing the Cache
When generating text, we have to remember to reset both the keys and value buffers between two separate text-generation calls. Otherwise, the queries of a new prompt will attend to stale keys left over from the previous sequence, which causes the model to rely on irrelevant context and produce incoherent output. To prevent this, we add a `reset_kv_cache` method to the `MultiHeadAttention` class that we can use between text-generation calls later:

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

> **Note**
> 1. `Otherwise, the queries of a new prompt will attend to stale keys left over from the previous sequence` 如果在 `use_cache=True` 的条件下遇到这种情况 `new prompt` 是如何处理的？（**学习完KV Cache 代码实现后解答**）  
> **解答**： `use_cache` 版本是允许处理多个 tokens 的，执行流程和 `Prefill` 阶段一样，此时的 `queries` 在多头处理之后，未 `queries.transpose(1, 2)` 之前的形状为 `[batch_size, num_heads, new_prompt_tokens_num, head_dim]`, 相应的 `causal attention mask matrix` 也会进行适配

## 4. Propagating `use_cache` in the Full Model
- With the changes to the `MultiHeadAttention` class in place, we now modify the `GPTModel` class. First, we add a position tracking for the token indices to the constructor:

```python
# self.trf_blocks = nn.Sequential(
    #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
####################################################
# NEW
self.trf_blocks = nn.ModuleList(
    [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

self.current_pos = 0
####################################################
```

- Then, we replace the one-liner block call with an explicit loop, passing use_cache through each transformer block:

```python
def forward(self, in_idx, use_cache=False):
    batch_size, seq_len = in_idx.shape
    tok_embeds = self.tok_emb(in_idx) # tok_embeds 的 shape: [batch_size, seq_len, emb_dim]

    # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

    ####################################################
    # NEW

    if use_cache:
        pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    ####################################################

    x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
    x = self.drop_emb(x)

    # x = self.trf_blocks(x)
    ####################################################
    # NEW
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
    ####################################################

    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits
```

- The above change then also requires a small modification to the `TransformerBlock` class to accept the `use_cache` argument:

```python
def forward(self, x, use_cache=False):
    # Shortcut connection for attention block
    shortcut = x
    x = self.norm1(x)

    # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
    ####################################################
    # NEW
    x = self.att(x, use_cache=use_cache)
    ####################################################
```

- Lastly, we add a model-level reset to GPTModel to clear all block caches at once for our convenience:

```python
def forward(self, in_idx, use_cache=False):
    # ...
    ####################################################
    # NEW
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################
```

## 5. Using the Cache in Generation
With the changes to the `GPTModel`, `TransformerBlock`, and `MultiHeadAttention`, finally, here's how we use the KV cache in a simple text generation function:

```python
####################################################
# NEW
def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt 
            # Prefill 阶段
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            # Decode 阶段
            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
####################################################
```

> **Note**  
> we only feed the model the new token in c) via `logits = model(next_idx, use_cache=True)`. Without caching, we feed the model the whole input `logits = model(idx[:, -ctx_len:], use_cache=False)` as it has no stored keys and values to reuse.

## Other modifications
we extend the `forward` method of the `MultiHeadAttention` class, 让 causal attention mask matrix 适配 use_cache 版本

```python
def forward(self, x, use_cache=False):
        # ...

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        ####################################################
        # NEW
        num_tokens_Q = queries.shape[-2] # num_tokens
        num_tokens_K = keys.shape[-2] # num_tokens
        if use_cache:
            mask_bool = self.mask.bool()[
                self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
            ]
            self.ptr_current_pos += num_tokens_Q
        ####################################################
        # Original mask truncated to the number of tokens and converted to boolean
        else:
            mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # ...
```

> **Note**
> 1. `queries` 的 shape: 
>> - Prefill 阶段：`[batch_size, num_heads, prompt_tokens_num, head_dim]`, 其中 `prompt_tokens_num` 表示第一次用户输入 prompt 时，prompt 中的 token 数量
>> - Decode 阶段： `[batch_size, num_heads, 1, head_dim]`, 其中 `1` 表示后续模型一个一个的生成新 token
> 
> 2. `mask_bool = self.mask.bool()[self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K]` 理解：对 `mask` 矩阵进行切分，使其得到的 `causal attention mask matrix` 能够适配 `queries` 的形状
> 