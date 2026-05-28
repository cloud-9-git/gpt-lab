# BPE tokenizer 구현을 위한 Python 문법 문서

대상 과제:

```text
1. BPE tokenizer
파일: src/bpe.py
테스트: pytest tests/test_bpe.py -v
```

이 문서는 BPE 개념 설명이 아니라, `src/bpe.py`를 구현할 때 필요한 Python 문법을 정리한 문서입니다.
예시는 과제 코드에 바로 연결되도록 `BPETokenizer`, `id_to_token`, `token_to_id`, `merges`, `ids` 같은 이름을 사용합니다.

---

## 1. 클래스와 `self`

`BPETokenizer`는 클래스입니다.

```python
class BPETokenizer:
    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
```

여기서 `self`는 “지금 만들어진 tokenizer 객체 자신”입니다.

예를 들어:

```python
tok = BPETokenizer(vocab_size=300)
```

이렇게 만들면 `tok` 안에 아래 값들이 생깁니다.

```python
tok.vocab_size
tok.id_to_token
tok.token_to_id
tok.merges
```

클래스 안에서는 `tok`라는 이름을 직접 쓰지 않고 `self`라고 씁니다.

```python
self.id_to_token = {}
```

이 말은 “이 tokenizer 객체의 `id_to_token`을 빈 dict로 만들겠다”는 뜻입니다.

---

## 2. 타입 힌트 문법

`src/bpe.py`에는 이런 형태가 나옵니다.

```python
def train(self, corpus: str):
```

`corpus: str`는 `corpus`가 문자열일 거라는 표시입니다.

```python
def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
```

이 뜻은:

```text
text는 문자열
add_bos_eos는 True/False 값
반환값은 int가 들어 있는 list
```

`-> list[int]`는 실행에 필수는 아니지만, 읽는 사람과 테스트 작성자에게 “이 함수는 정수 리스트를 반환해야 한다”고 알려줍니다.

```python
def save(self, path: str | Path):
```

`str | Path`는 `path`가 문자열일 수도 있고 `Path` 객체일 수도 있다는 뜻입니다.

예:

```python
tok.save("vocab.json")
tok.save(Path("vocab.json"))
```

---

## 3. dict 문법

BPE tokenizer에서는 dict를 많이 씁니다.

```python
self.id_to_token = {}
self.token_to_id = {}
```

dict는 key와 value를 연결하는 자료구조입니다.

```python
d = {}
d["apple"] = 10
d["banana"] = 20
```

이제 `"apple"`로 `10`을 찾을 수 있습니다.

```python
print(d["apple"])
# 10
```

BPE에서는 이런 식으로 씁니다.

```python
self.id_to_token[0] = "<pad>"
self.token_to_id["<pad>"] = 0
```

두 dict는 방향이 반대입니다.

```text
id_to_token:
0 -> "<pad>"

token_to_id:
"<pad>" -> 0
```

---

## 4. dict key로 쓸 수 있는 값

Python dict의 key는 변하지 않는 값이어야 합니다.

가능:

```python
d["text"] = 1
d[123] = 2
d[(10, 20)] = 3
d[b"A"] = 4
```

불가능:

```python
d[[10, 20]] = 3
# TypeError: unhashable type: 'list'
```

리스트는 나중에 내용이 바뀔 수 있어서 dict key로 못 씁니다.

BPE merge pair는 그래서 list가 아니라 tuple로 저장합니다.

```python
pair = (241, 153)
self.token_to_id[pair] = 260
```

---

## 5. list 문법

list는 순서가 있는 값들의 묶음입니다.

```python
ids = [241, 153, 160]
```

인덱스로 값을 꺼낼 수 있습니다.

```python
ids[0]  # 241
ids[1]  # 153
ids[2]  # 160
```

맨 뒤에 값을 추가할 때는 `append`를 씁니다.

```python
new_ids = []
new_ids.append(260)
new_ids.append(160)

print(new_ids)
# [260, 160]
```

길이는 `len()`으로 구합니다.

```python
len(ids)
# 3
```

---

## 6. tuple 문법

tuple은 list처럼 여러 값을 담지만, 한 번 만들면 안의 값을 바꿀 수 없습니다.

```python
pair = (241, 153)
```

BPE에서 pair는 항상 token 2개짜리 tuple입니다.

```python
pair = (ids[i], ids[i + 1])
```

예:

```python
ids = [241, 153, 160]
i = 0
pair = (ids[i], ids[i + 1])

print(pair)
# (241, 153)
```

---

## 7. bytes 문법

문자열을 UTF-8 byte로 바꾸면 bytes 객체가 됩니다.

```python
text = "한"
raw = text.encode("utf-8")

print(raw)
# b'\xed\x95\x9c'
```

이 bytes를 숫자 리스트로 보고 싶으면 `list()`를 씁니다.

```python
list(raw)
# [237, 149, 156]
```

byte 하나를 bytes 객체로 만들 때는:

```python
bytes([65])
# b'A'
```

주의:

```python
bytes([237])
# b'\xed'
```

`bytes([237])` 하나만으로는 `"한"`이 아닙니다.
`"한"`은 세 byte가 모여야 합니다.

```python
bytes([237, 149, 156]).decode("utf-8")
# "한"
```

BPE 초기화에서는 byte 0부터 255까지를 기본 token으로 넣습니다.

```python
for byte_value in range(256):
    token = bytes([byte_value])
```

---

## 8. `range()` 문법

`range(n)`은 `0`부터 `n - 1`까지 반복합니다.

```python
for i in range(4):
    print(i)
```

출력:

```text
0
1
2
3
```

byte는 0부터 255까지 있으므로:

```python
for byte_value in range(256):
    ...
```

이렇게 쓰면 `byte_value`가 0, 1, 2, ..., 255 순서로 들어옵니다.

---

## 9. `enumerate()` 문법

리스트를 돌면서 값과 인덱스를 같이 얻고 싶을 때 `enumerate()`를 씁니다.

```python
tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

for idx, token in enumerate(tokens):
    print(idx, token)
```

출력:

```text
0 <pad>
1 <unk>
2 <bos>
3 <eos>
```

특수 token을 초기화할 때 이런 형태로 쓸 수 있습니다.

```python
for idx, token in enumerate(SPECIAL_TOKENS):
    self.id_to_token[idx] = token
    self.token_to_id[token] = idx
```

---

## 10. dict comprehension 문법

`src/bpe.py`에는 이미 이런 코드가 있습니다.

```python
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
```

이 코드는 아래와 같습니다.

```python
SPECIAL_IDS = {}

for idx, token in enumerate(SPECIAL_TOKENS):
    SPECIAL_IDS[token] = idx
```

결과:

```python
{
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}
```

---

## 11. `zip(ids, ids[1:])` 문법

BPE에서는 이웃 pair를 세어야 합니다.

```python
ids = [10, 20, 10, 20]
```

`ids[1:]`는 1번 인덱스부터 끝까지 자른 리스트입니다.

```python
ids[1:]
# [20, 10, 20]
```

`zip(ids, ids[1:])`는 두 리스트를 나란히 묶습니다.

```python
for a, b in zip(ids, ids[1:]):
    print(a, b)
```

출력:

```text
10 20
20 10
10 20
```

그래서 pair를 만들 수 있습니다.

```python
for a, b in zip(ids, ids[1:]):
    pair = (a, b)
```

---

## 12. dict의 `.get()` 문법

pair 빈도를 셀 때 dict를 씁니다.

```python
pair_counts = {}
pair = (10, 20)
```

처음 보는 pair면 아직 dict에 없습니다.
이때 바로 `pair_counts[pair]`를 읽으면 에러가 납니다.

```python
pair_counts[pair]
# KeyError
```

그래서 `.get()`을 씁니다.

```python
pair_counts.get(pair, 0)
```

이 뜻은:

```text
pair가 dict에 있으면 그 값을 가져오고,
없으면 0을 가져와라.
```

빈도 세기 코드는 이렇게 됩니다.

```python
pair_counts[pair] = pair_counts.get(pair, 0) + 1
```

예:

```python
pair_counts = {}

pair = (10, 20)
pair_counts[pair] = pair_counts.get(pair, 0) + 1
# {(10, 20): 1}

pair_counts[pair] = pair_counts.get(pair, 0) + 1
# {(10, 20): 2}
```

---

## 13. `max(..., key=...)` 문법

가장 많이 나온 pair를 찾을 때 `max()`를 쓸 수 있습니다.

```python
pair_counts = {
    (10, 20): 3,
    (20, 30): 1,
    (30, 40): 5,
}
```

가장 큰 key를 찾고 싶은 게 아닙니다.
가장 큰 value를 가진 key를 찾고 싶은 것입니다.

```python
best_pair = max(pair_counts, key=pair_counts.get)

print(best_pair)
# (30, 40)
```

`key=pair_counts.get`의 뜻은:

```text
pair 자체를 비교하지 말고,
pair_counts[pair] 값을 기준으로 비교해라.
```

---

## 14. `while` 문법

pair를 치환할 때는 `for`보다 `while`이 편합니다.
왜냐하면 어떤 경우에는 한 칸 이동하고, 어떤 경우에는 두 칸 이동해야 하기 때문입니다.

```python
ids = [10, 20, 30]
i = 0

while i < len(ids):
    print(ids[i])
    i += 1
```

출력:

```text
10
20
30
```

BPE pair 치환에서는 이런 식으로 씁니다.

```python
new_ids = []
i = 0

while i < len(ids):
    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
        new_ids.append(new_id)
        i += 2
    else:
        new_ids.append(ids[i])
        i += 1
```

`i < len(ids) - 1`이 필요한 이유:

```text
ids[i + 1]을 읽으려면 다음 칸이 있어야 함.
마지막 위치에서는 다음 칸이 없으므로 검사하면 안 됨.
```

---

## 15. `and`의 짧은 평가

이 조건을 봅시다.

```python
if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
```

Python의 `and`는 앞 조건이 False이면 뒤 조건을 보지 않습니다.

그래서 `i`가 마지막 위치일 때:

```python
i < len(ids) - 1
```

이 False가 되고, 뒤의:

```python
ids[i + 1]
```

를 읽지 않습니다.
이 덕분에 인덱스 에러를 피할 수 있습니다.

---

## 16. `if not pair_counts`

빈 dict는 False처럼 동작합니다.

```python
pair_counts = {}

if not pair_counts:
    print("비어 있음")
```

BPE train에서 더 이상 pair가 없으면 멈춰야 합니다.

```python
if not pair_counts:
    break
```

`break`는 반복문을 즉시 종료합니다.

---

## 17. `break` 문법

`break`는 `while`이나 `for` 반복문을 멈춥니다.

```python
i = 0

while True:
    if i == 3:
        break
    print(i)
    i += 1
```

출력:

```text
0
1
2
```

BPE train에서는 보통 이런 상황에서 씁니다.

```python
if not pair_counts:
    break
```

또는:

```python
if len(self.id_to_token) >= self.vocab_size:
    break
```

---

## 18. 파일 경로 `Path`

`pathlib.Path`는 파일 경로를 다루는 클래스입니다.

```python
from pathlib import Path

path = Path("vocab.json")
```

문자열 path가 들어와도 `Path`로 바꿀 수 있습니다.

```python
path = Path(path)
```

파일을 읽고 쓸 때는 `open()`을 쓸 수 있습니다.

```python
with open(path, "w", encoding="utf-8") as f:
    ...
```

또는:

```python
path.write_text("hello", encoding="utf-8")
text = path.read_text(encoding="utf-8")
```

JSON 저장에서는 `open()`과 `json.dump()`를 같이 쓰는 편이 자연스럽습니다.

---

## 19. `with open(...) as f` 문법

파일을 열 때:

```python
with open(path, "w", encoding="utf-8") as f:
    f.write("hello")
```

`with`를 쓰면 파일을 다 쓴 뒤 자동으로 닫아줍니다.

모드:

```text
"w" = 쓰기. 기존 내용이 있으면 덮어씀.
"r" = 읽기.
```

---

## 20. JSON 저장과 읽기

먼저 import가 필요합니다.

```python
import json
```

저장:

```python
data = {
    "vocab_size": 300,
    "merges": [[10, 20], [30, 40]],
}

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

읽기:

```python
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
```

`ensure_ascii=False`는 한글을 JSON에 사람이 읽을 수 있게 저장하는 옵션입니다.

`indent=2`는 보기 좋게 들여쓰기해서 저장하는 옵션입니다.

---

## 21. JSON이 바로 저장하지 못하는 타입

JSON은 이런 값은 잘 저장합니다.

```python
{
    "name": "tokenizer",
    "size": 300,
    "items": [1, 2, 3],
}
```

하지만 Python의 `bytes`는 바로 저장하지 못합니다.

```python
json.dumps(bytes([65]))
# TypeError
```

tuple도 JSON에 저장하면 list처럼 바뀝니다.

```python
json.dumps((10, 20))
# "[10, 20]"
```

그래서 저장할 때 type 정보를 따로 넣는 방식이 필요합니다.

예:

```python
{"type": "special", "value": "<pad>"}
{"type": "byte", "value": 65}
{"type": "merge", "value": [241, 153]}
```

읽을 때는 type을 보고 원래 Python 타입으로 되돌립니다.

---

## 22. `isinstance()` 문법

값의 타입을 확인할 때 씁니다.

```python
token = bytes([65])

if isinstance(token, bytes):
    print("bytes입니다")
```

BPE 저장에서는 token 종류에 따라 다르게 처리해야 합니다.

```python
if isinstance(token, str):
    ...
elif isinstance(token, bytes):
    ...
elif isinstance(token, tuple):
    ...
```

`elif`는 “앞 조건이 아니면 이 조건을 검사한다”는 뜻입니다.

---

## 23. 문자열 encode / bytes decode

문자열을 bytes로 바꾸기:

```python
text = "한글"
raw = text.encode("utf-8")
```

bytes를 문자열로 바꾸기:

```python
text = raw.decode("utf-8")
```

decode할 때 깨진 byte가 있을 가능성을 방어하려면:

```python
text = raw.decode("utf-8", errors="replace")
```

하지만 정상적인 encode/decode 왕복에서는 보통 깨진 byte가 나오면 안 됩니다.

---

## 24. list comprehension 문법

반복해서 리스트를 만들 때 짧게 쓸 수 있습니다.

```python
values = [1, 2, 3]
doubled = [x * 2 for x in values]

print(doubled)
# [2, 4, 6]
```

BPE encode에서 byte 값을 token ID로 바꿀 때 쓸 수 있습니다.

```python
ids = [BYTE_OFFSET + b for b in text.encode("utf-8")]
```

위 코드는 아래와 같습니다.

```python
ids = []

for b in text.encode("utf-8"):
    ids.append(BYTE_OFFSET + b)
```

---

## 25. 리스트 앞뒤에 값 붙이기

`add_bos_eos=True`일 때 앞뒤에 특수 token을 붙여야 합니다.

```python
ids = [241, 153, 160]
ids = [2] + ids + [3]

print(ids)
# [2, 241, 153, 160, 3]
```

과제 코드에서는 직접 숫자 `2`, `3`을 써도 되지만, 함수로 가져오는 편이 더 읽기 좋습니다.

```python
ids = [self.get_bos_id()] + ids + [self.get_eos_id()]
```

---

## 26. 재귀 함수 문법

decode에서는 merge token을 byte token까지 펼쳐야 합니다.

merge token이 이렇게 저장되어 있다고 합시다.

```python
id_to_token[260] = (241, 153)
id_to_token[261] = (260, 160)
```

`261`을 펼치려면:

```text
261
-> 260, 160
-> 241, 153, 160
```

이런 구조는 재귀 함수로 처리할 수 있습니다.
재귀 함수는 자기 자신을 다시 호출하는 함수입니다.

```python
def flatten(token_id):
    token = self.id_to_token[token_id]

    if isinstance(token, bytes):
        return list(token)

    if isinstance(token, tuple):
        left, right = token
        return flatten(left) + flatten(right)
```

위 코드는 예시입니다.
실제 구현에서는 특수 token, byte token, merge token을 과제 구조에 맞게 처리해야 합니다.

---

## 27. tuple unpacking

tuple 안의 값을 변수 두 개로 나눠 받을 수 있습니다.

```python
pair = (241, 153)
left, right = pair

print(left)
# 241
print(right)
# 153
```

decode에서 merge token을 펼칠 때 자주 씁니다.

```python
left_id, right_id = token
```

---

## 28. set 문법

특수 token ID인지 확인할 때 set을 만들 수 있습니다.

```python
special_ids = set(SPECIAL_IDS.values())
```

예:

```python
SPECIAL_IDS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}

set(SPECIAL_IDS.values())
# {0, 1, 2, 3}
```

사용:

```python
if token_id in special_ids:
    ...
```

---

## 29. `continue` 문법

`continue`는 반복문의 이번 차례를 여기서 끝내고 다음 반복으로 넘어갑니다.

```python
for x in [1, 2, 3]:
    if x == 2:
        continue
    print(x)
```

출력:

```text
1
3
```

decode에서 특수 token을 건너뛸 때 쓸 수 있습니다.

```python
for token_id in ids:
    if skip_special and token_id in special_ids:
        continue
```

---

## 30. 테스트 실행 문법

BPE 테스트만 실행:

```bash
pytest tests/test_bpe.py -v
```

특정 테스트만 실행:

```bash
pytest tests/test_bpe.py -v -k "init_special"
```

테스트가 실패하면 보통 아래 중 하나입니다.

```text
_init_special_tokens 미구현
save/load 미구현
encode/decode 미구현
train 미구현
```

---

## 31. 함수별 필요한 문법 요약

### `_init_special_tokens`

필요 문법:

```text
dict 대입
for
range
enumerate
bytes([byte_value])
```

주로 쓰는 형태:

```python
for idx, token in enumerate(SPECIAL_TOKENS):
    ...

for byte_value in range(NUM_BYTES):
    ...
```

### `train`

필요 문법:

```text
str.encode
list comprehension
while
zip
dict.get
max(..., key=...)
break
list append
tuple key
```

주로 쓰는 형태:

```python
pair_counts[pair] = pair_counts.get(pair, 0) + 1
best_pair = max(pair_counts, key=pair_counts.get)
```

### `encode`

필요 문법:

```text
str.encode
list comprehension
for merge in self.merges
while
list append
리스트 앞뒤 붙이기
```

주로 쓰는 형태:

```python
ids = [BYTE_OFFSET + b for b in text.encode("utf-8")]
ids = [self.get_bos_id()] + ids + [self.get_eos_id()]
```

### `decode`

필요 문법:

```text
for
continue
set
isinstance
재귀 함수
bytes(byte_values).decode
```

주로 쓰는 형태:

```python
bytes(byte_values).decode("utf-8")
```

### `save`

필요 문법:

```text
Path
with open
json.dump
isinstance
dict/list 구성
```

### `load`

필요 문법:

```text
with open
json.load
int(...)
tuple(...)
bytes([...])
dict 재구성
```

---

## 32. 구현 중 자주 헷갈리는 문법 실수

### list를 dict key로 쓰는 실수

틀림:

```python
self.token_to_id[[241, 153]] = 260
```

맞음:

```python
self.token_to_id[(241, 153)] = 260
```

### byte 값을 문자로 바로 바꾸는 실수

틀림:

```python
chr(237)
```

맞음:

```python
bytes([237, 149, 156]).decode("utf-8")
```

### JSON load 후 key가 문자열인 것을 잊는 실수

JSON 객체의 key는 문자열입니다.

```json
{
  "260": {"type": "merge", "value": [241, 153]}
}
```

Python에서 다시 쓸 때는 int로 바꿔야 합니다.

```python
token_id = int(token_id_str)
```

### tuple이 JSON에서는 list로 바뀌는 것을 잊는 실수

저장된 값:

```python
[241, 153]
```

복원:

```python
tuple([241, 153])
# (241, 153)
```

### 마지막 index에서 `ids[i + 1]`을 읽는 실수

틀림:

```python
if (ids[i], ids[i + 1]) == best_pair:
```

맞음:

```python
if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
```

---

## 33. 문법만 확인하는 작은 연습 코드

아래 코드는 BPE 전체 구현이 아니라, pair counting과 replacement 문법만 확인하는 코드입니다.

```python
ids = [10, 20, 10, 20, 30, 10, 20]

pair_counts = {}
for a, b in zip(ids, ids[1:]):
    pair = (a, b)
    pair_counts[pair] = pair_counts.get(pair, 0) + 1

best_pair = max(pair_counts, key=pair_counts.get)
new_id = 260

new_ids = []
i = 0
while i < len(ids):
    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
        new_ids.append(new_id)
        i += 2
    else:
        new_ids.append(ids[i])
        i += 1

print(pair_counts)
print(best_pair)
print(new_ids)
```

예상 결과:

```text
{(10, 20): 3, (20, 10): 1, (20, 30): 1, (30, 10): 1}
(10, 20)
[260, 260, 30, 260]
```

