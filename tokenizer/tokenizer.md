# tokenizer 的底层代码实现
## 实例化 tokenizer 

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
```

## 实现 `tiktoken.get_encoding()` 方法

```python
def get_encoding(encoding_name: str) -> Encoding:
    if not isinstance(encoding_name, str):
        raise ValueError(f"Expected a string in get_encoding, got {type(encoding_name)}")

    if encoding_name in ENCODINGS:
        return ENCODINGS[encoding_name]

    with _lock:
        if encoding_name in ENCODINGS:
            return ENCODINGS[encoding_name]

        if ENCODING_CONSTRUCTORS is None:
            _find_constructors()
            assert ENCODING_CONSTRUCTORS is not None

        if encoding_name not in ENCODING_CONSTRUCTORS:
            raise ValueError(
                f"Unknown encoding {encoding_name}.\n"
                f"Plugins found: {_available_plugin_modules()}\n"
                f"tiktoken version: {tiktoken.__version__} (are you on latest?)"
            )

        constructor = ENCODING_CONSTRUCTORS[encoding_name]
        enc = Encoding(**constructor())
        ENCODINGS[encoding_name] = enc
        return enc
```
> tokenizer 为一个实例化的 `Encoding` 对象

## 实现 Encoding 类的 `encode()` 方法

```python
    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),  # noqa: B006
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[int]:
        """Encodes a string into tokens.

        Special tokens are artificial tokens used to unlock capabilities from a model,
        such as fill-in-the-middle. So we want to be careful about accidentally encoding special
        tokens, since they can be used to trick a model into doing something we don't want it to do.

        Hence, by default, encode will raise an error if it encounters text that corresponds
        to a special token. This can be controlled on a per-token level using the `allowed_special`
        and `disallowed_special` parameters. In particular:
        - Setting `disallowed_special` to () will prevent this function from raising errors and
          cause all text corresponding to special tokens to be encoded as natural text.
        - Setting `allowed_special` to "all" will cause this function to treat all text
          corresponding to special tokens to be encoded as special tokens.

        ```
        >>> enc.encode("hello world")
        [31373, 995]
        >>> enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        [50256]
        >>> enc.encode("<|endoftext|>", allowed_special="all")
        [50256]
        >>> enc.encode("<|endoftext|>")
        # Raises ValueError
        >>> enc.encode("<|endoftext|>", disallowed_special=())
        [27, 91, 437, 1659, 5239, 91, 29]
        ```
        """
        if allowed_special == "all":
            allowed_special = self.special_tokens_set
        if disallowed_special == "all":
            disallowed_special = self.special_tokens_set - allowed_special
        if disallowed_special:
            if not isinstance(disallowed_special, frozenset):
                disallowed_special = frozenset(disallowed_special)
            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())

        try:
            return self._core_bpe.encode(text, allowed_special) # `._core_bpe.encode()`: text --> token IDs
        except UnicodeEncodeError:
            # BPE operates on bytes, but the regex operates on unicode. If we pass a str that is
            # invalid UTF-8 to Rust, it will rightfully complain. Here we do a quick and dirty
            # fixup for any surrogate pairs that may have sneaked their way into the text.
            # Technically, this introduces a place where encode + decode doesn't roundtrip a Python
            # string, but given that this is input we want to support, maybe that's okay.
            # Also we use errors="replace" to handle weird things like lone surrogates.
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace") # `.encode()`: 字符串转换成字节
            return self._core_bpe.encode(text, allowed_special)
```
> `self._core_bpe.encode()`
>   - `self`: 指的是实例化的 `Encoding` 类（即 tokenizer）
>   - `.core_bpe`: 是 `Encoding` 类实例对象的属性，属性值为 `CoreBPE` 类实例化的对象, `self._core_bpe = _tiktoken.CoreBPE(mergeable_ranks, special_tokens, pat_str)`
>   - `.encode()`: 是 `CoreBPE` 类实例对象的方法(即 BPE 算法)，底层由 `rust` 语言实现

## 实现 `._core_bpe.encode()` 方法（**待补充**）