# polyglot-jax-inference

## Introduction

본 레포지토리는 Jax/Flax 기반 [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) 모델 추론 코드로, TPU에서 GPT-NeoX 기반 LLM으로 문장을 생성할 수 있습니다. 해당 코드로 한국어용 LLM인 [polyglot-ko](https://github.com/EleutherAI/polyglot), [KORani](https://github.com/krafton-ai/KORani), [KULLM](https://github.com/nlpai-lab/KULLM) 등을 실행하는 것을 목표로 합니다.

다음은 실행이 확인된 모델 목록입니다.

- [EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)
- [EleutherAI/polyglot-ko-3.8b](https://huggingface.co/EleutherAI/polyglot-ko-3.8b)
- [EleutherAI/polyglot-ko-5.8b](https://huggingface.co/EleutherAI/polyglot-ko-5.8b)
- [EleutherAI/polyglot-ko-12.8b](https://huggingface.co/EleutherAI/polyglot-ko-12.8b)
- [KRAFTON/KORani-v1-13B](https://huggingface.co/KRAFTON/KORani-v1-13B)

KORani의 경우 v2와 v3는 LLaMA 모델 기반이므로 본 레포지토리로 실행할 수 없습니다. 그 외에 GPT-NeoX 기반 모델은 모두 실행 가능합니다.

## Requirements

해당 코드를 실행하기 위해 다음의 라이브러리가 필요합니다.

- jax
- flax
- chex
- torch
- transformers

Cloud TPU VM 환경에서 작업할 경우 다음의 명령어를 통해 설치할 수 있습니다.

```bash
$ pip install torch --index-url https://download.pytorch.org/whl/cpu
$ pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
$ pip install flax chex transformers
```

## Usage

자세한 사용 방식은 [이 노트북](./example.ipynb)에서 확인해 볼 수 있습니다. 테스트를 위해 해당 노트북을 참고하기 바랍니다.

본 레포지토리는 LLM 추론을 위해 data parallelism과 model parallelism을 지원합니다. 다음을 통해 병렬화 방법을 정의하세요.

```python
mesh = Mesh(mesh_utils.create_device_mesh((1, 8)), ("dp", "mp"))
```

다음으로 subword tokenizer를 불러옵니다. 본 레포지토리는 huggingface와는 달리 새로운 토큰을 우측으로 삽입하므로 다음과 같이 설정하세요.

```python
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-12.8b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
```

이후 huggingface에서 PyTorch용 모델을 불러오십시오. 그리고 본 레포지토리 구현체를 통해 가중치를 변환하세요.

```python
model_hf = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-12.8b")

head_dim = model_hf.config.hidden_size // model_hf.config.num_attention_heads
rotary_dim = int(model_hf.config.rotary_pct * head_dim)

model = Transformer(
    vocab_size=model_hf.config.vocab_size,
    layers=model_hf.config.num_hidden_layers,
    dim=model_hf.config.hidden_size,
    heads=model_hf.config.num_attention_heads,
    hidden=model_hf.config.intermediate_size,
    rotary=rotary_dim,
    eps=model_hf.config.layer_norm_eps,
)
params = jax.tree_map(
    lambda param, spec: jax.device_put(param, NamedSharding(mesh, spec)),
    convert_weights(model_hf.state_dict(), get_conversion_rules(model)),
    get_sharding_rules(model)
)
```

이제 문장 생성을 위한 함수를 정의하고 `jax.pjit`로 컴파일하세요.

```python
temperature = 0.8
max_length = 1024

@pjit
def generate(x: chex.Array, mask: chex.Array, params: chex.ArrayTree, rng: chex.PRNGKey) -> chex.Array:
    rng, new_rng = jax.random.split(rng)
    generated = jnp.zeros((x.shape[0], max_length), dtype=jnp.int32)

    logits, variables = model.apply({"params": params}, x, mask, mutable=["cache"])
    new_tokens = jax.random.categorical(rng, logits[:, -1, :] / temperature)
    generated = jnp.roll(generated, -1, 1).at[:, -1].set(new_tokens)

    def body_fn(_: int, state: tuple[chex.Array, ...]):
        x, cache, rng, generated = state
        rng, new_rng = jax.random.split(rng)

        logits, variables = model.apply({"params": params, "cache": cache}, x[:, None], mutable=["cache"])
        new_tokens = jax.random.categorical(rng, logits[:, -1, :] / temperature)
        generated = jnp.roll(generated, -1, 1).at[:, -1].set(new_tokens)
        return new_tokens, variables["cache"], new_rng, generated

    state = (new_tokens, variables["cache"], new_rng, generated)
    state = jax.lax.fori_loop(0, max_length - 1, body_fn, init_val=state)
    return state[3]
```

최종적으로 다음의 코드를 통해 문장을 생성할 수 있습니다.

```python
encodings = tokenizer("이 문장은 GPT-NeoX Jax 구현체 테스트를 위한 예제 문장입니다.", max_length=2048, padding="max_length", truncation=True, return_tensors="np")

with mesh:
    generated = generate(
        jnp.asarray(encodings.input_ids, dtype=jnp.int32),
        jnp.asarray(encodings.attention_mask, dtype=jnp.bool_),
        params,
        jax.random.PRNGKey(1),
    ).block_until_ready()

print(tokenizer.decode(generated[0].tolist()))
```

## Acknowledgement

본 레포지토리는 [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/)의 지원으로 테스트되었습니다.

## License

[MIT License](./LICENSE)
