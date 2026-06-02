# Multi-Head Attention 복습 노트

## 현재 구현에서 봐야 할 핵심 흐름

`MultiHeadAttention`의 목표는 입력 `x`를 받아서 각 토큰이 이전 토큰들을 얼마나 참고할지 계산한 뒤, 다시 `(batch_size, seq_len, d_model)` 형태의 출력으로 만드는 것이다.

입력 `x`의 기본 shape:

```python
x.shape == (batch_size, num_tokens, d_model)
```

- `batch_size`: 한 번에 처리하는 문장/샘플 개수
- `num_tokens`: sequence length, 즉 토큰 개수
- `d_model`: 각 토큰을 표현하는 벡터 크기

---

## d_model

`d_model`은 Transformer 안에서 각 토큰을 표현하는 벡터의 차원이다.

GPT 계열에서는 보통 같은 값이 여러 곳에서 쓰인다.

- token embedding 차원
- positional embedding 차원
- attention 입력/출력 차원
- Transformer block 사이를 흐르는 hidden state 차원

즉 `d_model`은 "임베딩 크기"라고 이해해도 맞다. 다만 Transformer 내부에서도 계속 유지되는 토큰 표현 차원이기 때문에 hidden size라고 부르기도 한다.

---

## self.d_model에 x.shape를 다시 넣는 문제

이런 코드는 문법적으로는 가능하다.

```python
batch_size, num_tokens, self.d_model = x.shape
```

하지만 좋은 방식은 아니다.

`self.d_model`은 `__init__`에서 정해둔 모델 설정값이다.

```python
self.d_model = d_model
```

그런데 `forward()`에서 입력이 들어올 때마다 `self.d_model`에 다시 값을 넣으면, 모델 설정값을 입력 shape로 덮어쓰는 꼴이 된다.

보통은 이렇게 로컬 변수로 받는다.

```python
batch_size, num_tokens, d_model = x.shape
```

세 번째 값이 필요 없다면 이렇게 받을 수도 있다.

```python
batch_size, num_tokens, _ = x.shape
```

이건 효율 문제가 아니라, 모델 설정값을 함부로 바꾸지 않기 위한 로직 안정성 문제다.

---

## Sequence Length

sequence length는 입력에 들어온 토큰 개수다.

현재 코드 흐름에서는 보통 `num_tokens`가 sequence length 역할을 한다.

예를 들어:

```python
x.shape == (2, 10, 64)
```

이면:

- `2`: batch size
- `10`: sequence length / num_tokens
- `64`: d_model

causal mask는 토큰끼리 볼 수 있는 위치를 정하는 것이므로 `(num_tokens, num_tokens)` 크기가 필요하다.

---

## Q, K, V Projection

attention에서는 입력 `x`에서 세 가지 표현을 만든다.

- `Q` / query: 내가 무엇을 찾고 있는가
- `K` / key: 각 토큰이 어떤 특징을 가지고 있는가
- `V` / value: 실제로 가져올 정보

보통 한 번의 linear layer로 세 개를 같이 만든다.

```python
qkv = qkv_proj(x)
queries, keys, values = qkv.chunk(3, dim=-1)
```

이때 `qkv`의 마지막 차원은 `3 * d_model`이어야 한다. 그래야 `chunk(3, dim=-1)`로 나눴을 때 각각 `(B, T, d_model)`이 된다.

---

## Head 분리

multi-head attention은 `d_model` 차원을 여러 head로 나눠서 병렬로 attention을 계산한다.

예를 들어:

```python
d_model = 64
n_heads = 4
head_dim = 16
```

이면 각 head는 16차원씩 담당한다.

shape 변화:

```python
(B, T, d_model)
-> (B, T, n_heads, head_dim)
-> (B, n_heads, T, head_dim)
```

두 번째 변환에서 `transpose(1, 2)`를 사용한다.

---

## Attention Score

attention score는 query와 key의 유사도다.

```python
attn_scores = queries @ keys.transpose(-2, -1)
```

shape:

```python
(B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T)
-> (B, n_heads, T, T)
```

마지막 `(T, T)`는 각 query 위치가 각 key 위치를 얼마나 볼지 나타낸다.

score는 보통 `sqrt(head_dim)`으로 나눈다.

```python
attn_scores = attn_scores / (head_dim ** 0.5)
```

이 scaling을 하지 않으면 softmax가 너무 뾰족해져서 학습이 불안정해질 수 있다.

---

## Mask

mask는 attention에서 보면 안 되는 위치를 막는 장치다.

GPT 같은 causal language model은 미래 토큰을 보면 안 된다.

예를 들어 토큰이 4개라면 볼 수 있는 위치는 이렇게 된다.

```text
        key 위치
        0   1   2   3
query 0 O   X   X   X
query 1 O   O   X   X
query 2 O   O   O   X
query 3 O   O   O   O
```

`X` 위치가 미래 토큰이므로 가려야 하는 곳이다.

boolean mask로 표현하면:

```python
[
  [False, True,  True,  True ],
  [False, False, True,  True ],
  [False, False, False, True ],
  [False, False, False, False],
]
```

`True`인 위치가 가려질 위치다.

---

## mask_bool

`mask_bool`은 boolean mask라는 뜻이다.

```python
mask_bool = mask.bool()
```

이 값은 `True` / `False`로 이루어진 텐서다.

`True`인 위치는 attention score에서 `-inf`로 바뀐다.

```python
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

그 뒤 softmax를 하면 `-inf` 위치는 attention weight가 0이 된다.

---

## masked_fill_

`masked_fill_`은 mask가 `True`인 위치를 특정 값으로 채우는 함수다.

```python
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

뜻:

> `mask_bool`이 `True`인 위치의 `attn_scores` 값을 `-inf`로 바꿔라.

함수 이름 끝의 `_`는 in-place 연산이라는 뜻이다. 즉 원본 `attn_scores`를 직접 바꾼다.

---

## Attention Weight

attention weight는 attention score에 softmax를 적용한 값이다.

```python
attn_weights = torch.softmax(attn_scores, dim=-1)
```

shape:

```python
(B, n_heads, T, T)
```

의미:

```python
attn_weights[batch, head, query_position, key_position]
```

즉 특정 query 토큰이 특정 key 토큰을 얼마나 참고하는지 나타낸다.

---

## return_attention_weights

`return_attention_weights`는 attention 출력과 함께 attention weight도 반환할지 정하는 옵션이다.

기본적으로 모델 학습이나 추론에서는 attention 출력만 필요하다.

```python
out = mha(x)
```

디버깅, 테스트, 시각화에서는 attention weight도 같이 볼 수 있다.

```python
out, attn_weights = mha(x, return_attention_weights=True)
```

보는 경우:

- causal mask가 잘 적용됐는지 확인할 때
- 미래 토큰의 attention weight가 0인지 확인할 때
- 모델이 어떤 토큰을 많이 참고하는지 분석할 때
- attention map을 heatmap으로 시각화할 때

안 보는 경우:

- 일반 학습
- 일반 추론
- 다음 Transformer block으로 출력만 넘기면 되는 경우
- 메모리 사용량을 줄이고 싶은 경우

---

## contiguous()

`contiguous()`는 텐서의 메모리 배치를 연속적인 형태로 정리하는 함수다.

attention 구현에서는 보통 `transpose()` 뒤에 나온다.

```python
context_vec = context_vec.transpose(1, 2)
context_vec = context_vec.contiguous().view(batch_size, num_tokens, d_model)
```

`transpose()`는 차원 순서를 바꾸지만, 실제 메모리 배치를 새로 정렬하지 않을 수 있다.

그 상태에서 `view()`를 하면 PyTorch가 에러를 낼 수 있다.

그래서 `view()` 전에 `contiguous()`를 호출해서 메모리 배치를 정리한다.

요즘은 `reshape()`를 사용하면 내부에서 필요한 경우 알아서 처리해주기도 한다.

```python
context_vec = context_vec.reshape(batch_size, num_tokens, d_model)
```

---

## 현재 코드에서 확인했던 주요 오류

이 노트 작성 시점에 확인했던 주요 오류는 다음과 같다.

### 1. `self.d_out`이 정의되지 않음

`forward()`에서 `self.d_out`을 사용하지만, `__init__`에서 정의된 적이 없으면 다음 오류가 난다.

```text
AttributeError: 'MultiHeadAttention' object has no attribute 'd_out'
```

### 2. qkv projection과 output projection 역할 구분

Q/K/V를 만들 projection은 `d_model -> 3 * d_model` 구조여야 한다.

반면 output projection은 여러 head를 합친 뒤 다시 정리하는 용도이므로 보통 `d_model -> d_model` 구조다.

두 projection은 역할이 다르다.

### 3. mask 크기

causal mask는 토큰 위치를 가리는 것이므로 sequence length 기준이어야 한다.

즉 `(num_tokens, num_tokens)` 형태가 핵심이다.

`d_model`은 토큰 벡터 차원이므로 mask 크기의 기준으로 쓰기에는 개념상 맞지 않다.

### 4. `causal_mask` 인자

`causal_mask=True`일 때만 미래 토큰을 가리는 mask를 적용하고, `False`라면 mask를 적용하지 않는 구조가 자연스럽다.

### 5. 반환값

`return_attention_weights=False`일 때는 최종 attention 출력만 반환한다.

```python
return out
```

`return_attention_weights=True`일 때는 출력과 attention weight를 같이 반환한다.

```python
return out, attn_weights
```

---

## 한 줄 요약

Multi-head attention 구현의 핵심은 `Q/K/V 만들기 -> head 나누기 -> score 계산 -> mask 적용 -> softmax -> value 가중합 -> head 합치기 -> output projection` 순서다.

