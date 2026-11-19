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
        # ...

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

## 3. Clearing the Cache
When generating text, we have to remember to reset both the keys and value buffers between two separate text-generation calls. Otherwise, the queries of a new prompt will attend to stale keys left over from the previous sequence, which causes the model to rely on irrelevant context and produce incoherent output. To prevent this, we add a `reset_kv_cache` method to the `MultiHeadAttention` class that we can use between text-generation calls later:

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

## 4. Propagating `use_cache` in the Full Model
- With the changes to the `MultiHeadAttention` class in place, we now modify the `GPTModel` class. First, we add a position tracking for the token indices to the constructor:

```python
self.current_pos = 0
```

- Then, we replace the one-liner block call with an explicit loop, passing use_cache through each transformer block:

```python

```