# GPT 사전학습 실험 결과 해석과 다음 실험 안내

## 1. 이 문서의 목적

이 문서는 Google Sheet에 정리한 GPT 사전학습 실험 결과를 처음 보는 사람도 이해할 수 있게 풀어쓴 안내서다. 목표는 세 가지다.

1. 지금까지 한 실험이 무엇을 바꿔 본 실험인지 이해한다.
2. 결과 숫자, 특히 `train loss`와 `val loss`가 무엇을 말하는지 해석한다.
3. 다음 실험을 어떤 가설로, 어떤 순서로 진행하면 좋을지 정한다.

이 문서는 시트의 모든 탭에서 1~3행만 사용했다. 1행은 비어 있었고, 2행은 실행 명령어, 3행은 최종 결과였다. 4행 이후는 사용하지 않았다.

중요한 전제는 이것이다.

> 이번 실험은 "정답을 찾았다"가 아니라 "다음에 무엇을 더 실험해야 하는지 알게 되었다"에 가깝다.

실험 초반에는 숫자가 작게라도 좋아지는 방향을 찾는 것이 중요하다. 지금 결과에서는 학습을 더 오래 하고, 모델을 조금 키우고, learning rate를 조금 높였을 때 좋아지는 신호가 보인다.

## 2. 기본 개념

### train loss

`train loss`는 모델이 학습 데이터에서 얼마나 틀리고 있는지를 나타내는 숫자다.

낮을수록 좋다. 예를 들어 `train loss`가 `7.0`에서 `5.0`으로 내려갔다면, 모델이 학습 데이터를 더 잘 맞히게 되었다는 뜻이다.

하지만 `train loss`만 보면 안 된다. 모델이 학습 데이터를 외워 버리면 train loss는 낮아질 수 있지만, 새로운 문장에는 약할 수 있다.

### validation loss 또는 val loss

`val loss`는 학습에 직접 사용하지 않은 검증 데이터에서 얼마나 틀리는지를 나타낸다.

이 실험에서 가장 중요하게 볼 숫자는 `val loss`다. val loss가 낮아졌다는 것은 모델이 학습 데이터뿐 아니라 처음 보는 데이터에도 조금 더 잘 대응한다는 뜻이다.

정리하면 다음과 같다.

| 숫자 | 의미 | 낮아지면 |
| --- | --- | --- |
| train loss | 학습 데이터에서의 오차 | 학습 데이터를 더 잘 맞힘 |
| val loss | 검증 데이터에서의 오차 | 처음 보는 데이터에도 더 잘 맞힘 |

### train-val gap

`train-val gap`은 `val loss - train loss`다.

이 값이 너무 커지면 모델이 학습 데이터에만 과하게 맞춰졌을 가능성이 있다. 이것을 과적합이라고 부른다.

예를 들어 다음 두 상황을 비교해 보자.

| 상황 | train loss | val loss | 해석 |
| --- | ---: | ---: | --- |
| A | 5.0 | 5.1 | 학습 데이터와 검증 데이터 성능이 비슷함 |
| B | 4.0 | 6.0 | 학습 데이터는 잘 맞히지만 검증 데이터는 못 맞힘 |

B는 과적합을 의심해야 한다.

이번 실험에서는 epochs를 늘릴수록 val loss가 계속 내려갔지만, train-val gap도 함께 커졌다. 따라서 아직은 더 학습할 가치가 있지만, 장기 학습에서는 과적합도 같이 감시해야 한다.

### perplexity

`perplexity`는 loss를 조금 더 직관적인 숫자로 바꾼 것이다. 대략적으로는 "모델이 다음 토큰을 고를 때 얼마나 헷갈리는가"라고 볼 수 있다.

perplexity도 낮을수록 좋다. 이 문서에서는 참고용으로만 사용한다. 최종 판단은 val loss를 중심으로 한다.

### epoch

`epoch`는 학습 데이터를 한 바퀴 보는 단위다.

`epochs=1`이면 학습 데이터를 한 번 보고 끝낸다. `epochs=16`이면 같은 학습 데이터를 16번 반복해서 본다.

이번 결과에서는 epoch를 늘릴수록 val loss가 계속 내려갔다. 이것은 모델이 아직 더 배울 여지가 있다는 신호다.

### step

`step`은 모델 파라미터를 한 번 업데이트한 횟수다.

같은 `epochs=1`이어도 batch size나 context length가 달라지면 step 수가 달라질 수 있다. 이 점이 매우 중요하다.

예를 들어 이번 batch size 실험에서는 다음처럼 step 수가 달랐다.

| batch size | final step |
| ---: | ---: |
| 16 | 786 |
| 8 | 1572 |
| 4 | 3144 |
| 2 | 6289 |

batch size 2가 좋아 보이지만, 사실은 step을 훨씬 많이 돌았기 때문에 좋아진 것일 수 있다. 그래서 batch size 실험은 "batch size 2가 정답"이라고 단정하면 안 된다.

### batch size

`batch size`는 한 번의 업데이트에서 몇 개의 학습 예시를 같이 볼지 정하는 값이다.

작은 batch는 업데이트를 자주 하게 만들 수 있고, 큰 batch는 업데이트 횟수가 줄어들 수 있다. 이번 코드에서는 batch size가 작아질수록 한 epoch 안의 step 수가 늘어났다.

따라서 batch size를 비교할 때는 가능하면 step 수를 맞춰야 한다.

### learning rate

`learning rate`, 줄여서 `lr`은 한 번 업데이트할 때 모델을 얼마나 크게 움직일지 정하는 값이다.

너무 작으면 학습이 느리다. 너무 크면 학습이 불안정해지거나 loss가 튈 수 있다.

이번 실험에서는 `1e-4`, `3e-4`, `5e-4` 중 `5e-4`가 가장 좋았다. 그래서 다음 실험에서는 `7e-4`, `1e-3`처럼 조금 더 큰 값도 시험해 볼 만하다.

### dropout 또는 drop rate

`dropout`은 모델이 일부 연결을 일부러 빼고 학습하게 만드는 기법이다. 모델이 학습 데이터를 너무 외우는 것을 막는 데 도움이 될 수 있다.

하지만 학습이 아직 부족한 초반에는 dropout이 오히려 방해가 될 수 있다. 이번 1 epoch 실험에서는 `drop-rate=0.0`이 가장 좋았다.

단, 장기 학습에서는 다시 확인해야 한다. 8 epochs나 16 epochs처럼 오래 학습하면 dropout이 validation loss를 안정시키는 데 도움이 될 수 있다.

### context length

`context length`는 모델이 한 번에 볼 수 있는 토큰 길이다.

길수록 더 긴 문맥을 볼 수 있지만, 계산량이 늘고 한 epoch 안의 step 수가 줄어들 수 있다. 이번 결과에서는 `context-length=64`가 가장 좋았지만, 128과 256은 step 수가 줄어든 상태에서 비교되었다.

그래서 이번 결과만 보고 "긴 context는 나쁘다"라고 결론 내리면 안 된다.

### emb-dim

`emb-dim`은 토큰을 표현하는 벡터의 크기다.

쉽게 말하면 모델이 각 토큰을 얼마나 넓은 공간에 표현할지 정하는 값이다. 값이 커지면 모델 표현력이 좋아질 수 있지만, 계산량도 늘어난다.

이번 실험에서는 `emb-dim=192`가 `64`, `128`보다 확실히 좋았다. 이것은 현재 모델이 조금 더 커져도 효과가 있다는 신호다.

### n-layers

`n-layers`는 Transformer block을 몇 층 쌓을지 정하는 값이다.

층이 많아지면 모델이 더 복잡한 패턴을 배울 수 있다. 대신 계산량도 늘어난다.

이번 실험에서는 `n-layers=4`가 `1`, `2`보다 좋았다. 이것도 모델 용량을 키우는 방향이 유효할 수 있다는 신호다.

## 3. 이번 실험 결과

### 기준 설정

여러 실험에서 반복해서 등장한 기준 설정은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| train text | `data/nsmc_lm_train.txt` |
| val text | `data/nsmc_lm_val.txt` |
| vocab size | `3000` |
| context length | `64` |
| emb dim | `128` |
| n heads | `4` |
| n layers | `2` |
| batch size | `8` |
| epochs | `1` |
| learning rate | 기본값 `3e-4` |
| drop rate | 기본값 `0.1` |
| eval iter | `5` |

기준 결과는 다음과 같다.

| final step | train loss | val loss | train-val gap | perplexity |
| ---: | ---: | ---: | ---: | ---: |
| 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |

이 기준 결과를 중심으로 각 실험을 해석하면 된다.

### 3.1 epochs 변화

epochs 실험은 같은 설정으로 학습을 얼마나 오래 할지 바꾼 실험이다.

| epochs | final step | train loss | val loss | train-val gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 2 | 3144 | 6.0927 | 6.2148 | 0.1221 | 500.1 |
| 4 | 6288 | 5.5249 | 5.6948 | 0.1699 | 297.3 |
| 8 | 12576 | 5.1969 | 5.4550 | 0.2581 | 233.9 |
| 16 | 25152 | 4.9453 | 5.3150 | 0.3697 | 203.4 |

좋아진 것:

- val loss가 `7.0688`에서 `5.3150`까지 계속 내려갔다.
- perplexity도 `1174.7`에서 `203.4`까지 내려갔다.
- 지금까지의 실험 중 가장 큰 개선이다.

주의해서 봐야 할 것:

- train-val gap도 `0.0328`에서 `0.3697`까지 커졌다.
- 아직 val loss가 내려가고 있으므로 당장 과적합이라고 보기는 어렵다.
- 하지만 더 오래 학습하면 어느 순간 val loss가 멈추거나 올라갈 수 있다.

다음 가설:

> 학습을 더 오래 하면 val loss는 더 내려갈 수 있지만, 16 epochs 이후부터는 과적합을 막는 설정이 필요할 수 있다.

다음 실험에서는 8~16 epochs를 기본으로 두고 dropout, learning rate, 모델 크기를 함께 비교하는 것이 좋다.

### 3.2 batch size 변화

batch size 실험은 한 번 업데이트할 때 몇 개의 샘플을 볼지 바꾼 실험이다.

| batch size | final step | train loss | val loss | train-val gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 786 | 7.2665 | 7.2813 | 0.0148 | 1452.9 |
| 8 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 4 | 3144 | 6.6335 | 6.6469 | 0.0134 | 770.4 |
| 2 | 6289 | 6.1651 | 6.2782 | 0.1131 | 532.8 |

좋아진 것:

- 숫자만 보면 `batch-size=2`가 val loss `6.2782`로 가장 좋다.
- batch size가 작아질수록 val loss가 내려갔다.

주의해서 봐야 할 것:

- 이 실험은 공정 비교가 아니다.
- batch size가 작아질수록 final step이 크게 늘었다.
- `batch-size=2`는 `batch-size=8`보다 약 4배 더 많이 업데이트했다.

따라서 이 결과는 다음처럼 해석해야 한다.

> batch size 2가 좋아서라기보다, 업데이트 횟수가 많아져서 좋아졌을 가능성이 크다.

다음 가설:

> batch size 자체보다 step 수가 val loss 개선에 더 큰 영향을 주었을 수 있다.

현재 코드에는 `--max-steps` 옵션이 없으므로 step 수를 완전히 맞춘 비교는 어렵다. 나중에 코드를 개선한다면 `--max-steps`를 추가해서 batch size별로 같은 step 수만 학습하게 만들면 좋다.

### 3.3 drop rate 변화

drop rate 실험은 dropout을 얼마나 적용할지 바꾼 실험이다.

| drop rate | final step | train loss | val loss | train-val gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1572 | 6.8437 | 6.8795 | 0.0358 | 972.1 |
| 0.1 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 0.2 | 1572 | 7.1595 | 7.1852 | 0.0257 | 1319.8 |

좋아진 것:

- 1 epoch 기준으로는 `drop-rate=0.0`이 가장 좋다.
- dropout을 끄면 val loss가 `7.0688`에서 `6.8795`로 내려갔다.

주의해서 봐야 할 것:

- 이 결과는 1 epoch에서만 나온 결과다.
- 학습 초반에는 dropout이 모델의 학습을 방해할 수 있다.
- 장기 학습에서는 dropout이 과적합을 막는 데 도움이 될 수 있다.

다음 가설:

> 짧게 학습할 때는 dropout을 끄는 것이 좋지만, 8~16 epochs에서는 `drop-rate=0.05`나 `0.1`이 validation loss를 안정시킬 수 있다.

### 3.4 learning rate 변화

learning rate 실험은 모델이 한 번 업데이트할 때 얼마나 크게 움직일지 바꾼 실험이다.

| learning rate | final step | train loss | val loss | train-val gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 1572 | 7.2740 | 7.2927 | 0.0187 | 1469.5 |
| 3e-4 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 5e-4 | 1572 | 6.3814 | 6.4468 | 0.0654 | 630.7 |

좋아진 것:

- `5e-4`가 가장 좋다.
- val loss가 `7.2927 -> 7.0688 -> 6.4468`로 크게 내려갔다.
- 아직 더 높은 learning rate도 시도해 볼 가치가 있다.

주의해서 봐야 할 것:

- learning rate는 너무 커지면 불안정해질 수 있다.
- `1e-3`이나 `1.5e-3`에서 loss가 갑자기 커지면 너무 큰 값이라고 보면 된다.

다음 가설:

> `5e-4`에서 학습이 좋아졌으므로, `7e-4`나 `1e-3`에서도 더 빠르게 좋아질 수 있다. 다만 너무 커지면 학습이 불안정해질 수 있다.

### 3.5 context length 변화

context length 실험은 모델이 한 번에 보는 문맥 길이를 바꾼 실험이다.

| context length | final step | train loss | val loss | train-val gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 128 | 786 | 7.2772 | 7.2835 | 0.0063 | 1456.1 |
| 256 | 393 | 7.3043 | 7.3071 | 0.0028 | 1490.8 |

좋아진 것:

- 이번 표에서는 `context-length=64`가 가장 좋다.

주의해서 봐야 할 것:

- 이 결과도 공정 비교가 아니다.
- context length가 길어질수록 final step이 줄었다.
- 256은 64보다 훨씬 긴 문맥을 보지만, 업데이트 횟수는 393 step으로 매우 적다.

따라서 이 결과는 다음처럼 읽어야 한다.

> context length가 길어서 나쁜 것이 아니라, step 수가 줄어서 충분히 학습하지 못했을 가능성이 있다.

다음 가설:

> 같은 step 수 또는 비슷한 학습량으로 비교하면 context length 128이나 256도 좋아질 수 있다. 다만 현재 작은 모델에서는 64가 가장 효율적일 수 있다.

### 3.6 n-layers 변화

n-layers 실험은 Transformer block의 층 수를 바꾼 실험이다.

| n-layers | final step | train loss | val loss | train-val gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1572 | 7.0931 | 7.1236 | 0.0305 | 1240.9 |
| 2 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 4 | 1572 | 6.8880 | 6.9419 | 0.0539 | 1034.7 |

좋아진 것:

- 층 수를 늘릴수록 val loss가 내려갔다.
- `n-layers=4`가 가장 좋다.

주의해서 봐야 할 것:

- 모델이 깊어지면 학습 시간이 늘어난다.
- train-val gap도 조금 커졌다.

다음 가설:

> 현재 모델은 너무 큰 상태가 아니라 오히려 조금 더 깊어져도 배울 수 있는 상태다. `n-layers=4`를 기본 후보로 두고 다른 설정과 조합해 볼 만하다.

### 3.7 emb-dim 변화

emb-dim 실험은 토큰 표현 벡터의 크기를 바꾼 실험이다.

| emb-dim | final step | train loss | val loss | train-val gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1572 | 7.2964 | 7.2916 | -0.0048 | 1467.9 |
| 128 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 192 | 1572 | 6.5216 | 6.5269 | 0.0053 | 683.3 |

좋아진 것:

- `emb-dim=192`가 가장 좋다.
- val loss가 `7.2916 -> 7.0688 -> 6.5269`로 내려갔다.
- train-val gap도 아주 작아서, 1 epoch 기준으로는 과적합 신호가 강하지 않다.

주의해서 봐야 할 것:

- emb-dim을 키우면 계산량과 메모리 사용량이 늘어난다.
- 256 이상은 더 좋아질 수도 있지만, 학습 시간이 부담될 수 있다.

다음 가설:

> 현재 모델은 표현력이 부족한 편일 수 있다. `emb-dim=192` 또는 `256`으로 키우면 validation loss가 더 내려갈 수 있다.

## 4. 전체 결론

이번 실험의 가장 중요한 결론은 다음과 같다.

| 결론 | 근거 |
| --- | --- |
| 더 오래 학습하면 좋아진다 | epochs 1에서 16까지 val loss가 계속 감소 |
| learning rate는 `5e-4`가 현재 가장 좋다 | `1e-4`, `3e-4`, `5e-4` 중 `5e-4`가 최저 val loss |
| 모델을 조금 키우면 좋아진다 | `emb-dim=192`, `n-layers=4`에서 개선 |
| dropout은 짧은 학습에서는 방해가 될 수 있다 | 1 epoch에서 `drop-rate=0.0`이 가장 좋음 |
| batch/context 결과는 조심해야 한다 | 값이 바뀌면서 final step도 같이 바뀜 |

현재 모델은 과적합 때문에 망한 상태라기보다 아직 충분히 배우지 못한 상태에 가깝다.

그 이유는 다음과 같다.

- epochs를 늘릴수록 val loss가 계속 내려갔다.
- emb-dim과 n-layers를 키웠을 때 val loss가 내려갔다.
- learning rate를 높였을 때 val loss가 크게 내려갔다.

즉, 다음 실험의 방향은 "과적합을 줄이기"만이 아니라 "더 잘 학습되게 만들기"가 되어야 한다.

추천 기본 후보는 다음과 같다.

| 항목 | 추천 값 |
| --- | --- |
| context length | `64` |
| emb dim | `192` |
| n heads | `4` |
| n layers | `4` |
| batch size | `8` |
| learning rate | `5e-4` 또는 `7e-4` |
| drop rate | `0.0`, `0.05`, `0.1` 비교 |
| epochs | `4`, `8`, `16` 비교 |

## 5. 다음 추가 실험 계획

아래 계획은 18회 실험이다. 너무 적지도 않고, 너무 많지도 않은 중간 규모다.

모든 실험에서 기본 명령은 기존 명령어를 사용하고, 아래에 표시된 옵션만 바꾸면 된다.

공통으로 유지할 값:

```powershell
--train-text data/nsmc_lm_train.txt
--val-text data/nsmc_lm_val.txt
--vocab-size 3000
--context-length 64
--n-heads 4
--batch-size 8
--eval-freq 20
--eval-iter 5
--tokenizer-path data/tokenizer.json
--train-token-cache data/train_token_ids.pt
--val-token-cache data/val_token_ids.pt
```

Windows PowerShell에서는 기존처럼 앞부분을 붙이면 된다.

```powershell
& 'C:\Users\icon0\anaconda3\envs\gpt-lab\python.exe' .\scripts\pretrain_local.py `
  --train-text data/nsmc_lm_train.txt `
  --val-text data/nsmc_lm_val.txt `
  --vocab-size 3000 `
  --context-length 64 `
  --emb-dim 192 `
  --n-heads 4 `
  --n-layers 4 `
  --batch-size 8 `
  --epochs 4 `
  --lr 5e-4 `
  --drop-rate 0.0 `
  --eval-freq 20 `
  --eval-iter 5 `
  --tokenizer-path data/tokenizer.json `
  --train-token-cache data/train_token_ids.pt `
  --val-token-cache data/val_token_ids.pt
```

### 5.1 1차: 좋은 신호를 합친 조합 실험 4회

목적:

> 지금까지 좋았던 설정을 합치면 정말 더 좋아지는지 확인한다.

| 실험 | emb-dim | n-layers | lr | drop-rate | epochs | 가설 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A1 | 192 | 4 | 5e-4 | 0.0 | 4 | 좋은 신호를 합치면 1 epoch 실험보다 크게 좋아질 것이다 |
| A2 | 192 | 4 | 5e-4 | 0.0 | 8 | 더 오래 학습하면 A1보다 좋아질 것이다 |
| A3 | 192 | 4 | 5e-4 | 0.05 | 8 | 약한 dropout이 validation loss를 안정시킬 수 있다 |
| A4 | 192 | 4 | 5e-4 | 0.1 | 8 | 기존 dropout도 장기 학습에서는 도움이 될 수 있다 |

판단 기준:

- A2가 A1보다 val loss가 낮으면 더 오래 학습할 가치가 있다.
- A3/A4가 A2보다 val loss가 낮으면 장기 학습에서 dropout이 도움이 된다.
- A2가 가장 좋으면 아직 dropout 없이 학습하는 쪽이 낫다.

### 5.2 2차: A3 기준 learning rate 탐색 5회

목적:

> 현재 best인 A3 설정에서 `5e-4`보다 더 좋은 learning rate가 있는지 찾는다.

A1-A4 결과에서는 `emb-dim=192`, `n-layers=4`, `lr=5e-4`, `drop-rate=0.05`, `epochs=8`인 A3가 가장 좋은 후보였다. 따라서 2차 실험은 A3를 기준점으로 두고 learning rate만 바꿔서 비교한다.

고정값:

- `emb-dim=192`
- `n-layers=4`
- `drop-rate=0.05`
- `epochs=8`

| 실험 | lr | 가설 |
| ---: | ---: | --- |
| B1 | 3e-4 | A3보다 안정적일 수 있지만 학습이 느릴 수 있다 |
| B2 | 5e-4 | A3 결과를 기준값으로 재사용한다 |
| B3 | 7e-4 | 더 빠르게 좋아질 수 있다 |
| B4 | 1e-3 | 좋을 수도 있지만 불안정해질 수 있다 |
| B5 | 1.5e-3 | 너무 커서 loss가 튈 수 있다 |

판단 기준:

- A3의 final val loss `5.1477`, best val loss 약 `5.1326`보다 낮아지는지 본다.
- val loss가 가장 낮은 lr을 고른다.
- 학습 중 loss가 갑자기 커지면 그 lr은 너무 큰 값으로 본다.
- 비슷하면 더 안정적인 작은 lr을 고른다.
- B2는 A3와 같은 설정이므로, 같은 seed와 데이터 조건이라면 다시 실행하지 않고 A3 결과를 재사용해도 된다.

### 5.3 3차: B5 기준 dropout 장기 재검증 3회

목적:

> B 시리즈에서 가장 좋았던 learning rate `1.5e-3`을 8 epochs 장기 학습에 사용할 때, dropout이 validation loss를 안정시키는지 확인한다.

B 시리즈에서는 `drop-rate=0.0`, `epochs=4` 조건에서 learning rate를 키울수록 validation loss가 계속 낮아졌고, `lr=1.5e-3`인 B5가 가장 좋은 결과를 냈다. 하지만 이 결과는 4 epochs 기준이다. 8 epochs로 더 오래 학습하면 높은 learning rate와 dropout 없음이 과적합이나 validation loss 흔들림을 만들 수 있으므로, 같은 learning rate에서 dropout을 다시 비교한다.

고정값:

- `emb-dim=192`
- `n-layers=4`
- `lr=1.5e-3`
- `epochs=8`

| 실험 | drop-rate | 가설 |
| ---: | ---: | --- |
| C1 | 0.0 | B5처럼 dropout 없이 높은 lr이 장기 학습에서도 가장 좋을 수 있다 |
| C2 | 0.05 | A 시리즈에서 좋았던 약한 dropout이 높은 lr에서도 val loss를 안정시킬 수 있다 |
| C3 | 0.1 | 더 강한 regularization이 높은 lr과 장기 학습의 과적합을 줄일 수 있다 |

판단 기준:

- train loss는 낮지만 val loss가 높으면 과적합을 의심한다.
- val loss가 가장 낮은 drop-rate를 고른다.
- train-val gap도 함께 본다.
- val loss가 중반 이후 다시 올라가면 해당 설정은 장기 학습에서 불안정한 것으로 본다.

### 5.4 4차: C2 기준 모델 크기 비교 6회

목적:

> B/C 시리즈에서 고른 `lr=1.5e-3`, `drop-rate=0.05` 조건에서 모델을 얼마나 키우는 것이 좋은지 확인한다.

B 시리즈에서는 `lr=1.5e-3`이 가장 좋았고, C 시리즈에서는 `drop-rate=0.05`가 가장 좋았다. 따라서 D 시리즈는 이 두 값을 고정한 뒤 `emb-dim`과 `n-layers`만 바꿔 모델 크기를 비교한다.

주의: C2는 8 epochs 장기 학습 결과이고, D 시리즈는 모델 크기 6개를 빠르게 비교하기 위해 `epochs=4`로 진행한다. D 시리즈에서 고른 모델 크기는 이후 E 실험이나 최종 학습에서 다시 8 epochs 이상으로 확인한다.

고정값:

- `lr=1.5e-3`
- `drop-rate=0.05`
- `epochs=4`
- `context-length=64`

| 실험 | emb-dim | n-layers | 가설 |
| ---: | ---: | ---: | --- |
| D1 | 128 | 2 | 새 lr/dropout에서도 작은 기준 모델이 충분한지 확인한다 |
| D2 | 192 | 2 | 표현 크기만 키웠을 때 좋아지는지 확인한다 |
| D3 | 256 | 2 | 표현 크기를 더 키우면 계속 좋아지는지 확인한다 |
| D4 | 128 | 4 | 깊이만 키웠을 때 좋아지는지 확인한다 |
| D5 | 192 | 4 | 현재 A/B/C 시리즈에서 사용한 유력한 모델 크기 |
| D6 | 256 | 4 | 가장 강하지만 계산량과 과적합 위험이 큰 후보 |

판단 기준:

- val loss가 내려가면 모델 크기를 키울 가치가 있다.
- train loss만 낮고 val loss가 높으면 너무 큰 모델일 수 있다.
- val loss 차이가 작으면 더 작고 빠른 모델을 선택한다.
- D6가 가장 좋더라도 학습 시간이 너무 길거나 개선폭이 작으면 `192/4`를 현실적인 선택으로 둔다.

D 시리즈 결과 요약:

- 순수 final val loss 1위는 D6 `emb-dim=256`, `n-layers=4`이다.
- 다만 D6는 D5 대비 개선폭이 매우 작고 train-val gap이 더 크다.
- 따라서 다음 E 시리즈의 고정 모델 크기는 현실적인 best인 D5 `emb-dim=192`, `n-layers=4`로 둔다.

### 5.5 5차: context length 재검증 3회

목적:

> context length가 정말 64가 좋은지, 아니면 step 수 때문에 그렇게 보였는지 확인한다.

고정값:

- `lr=1.5e-3`
- `drop-rate=0.05`
- `emb-dim=192`
- `n-layers=4`
- `epochs=4`

| 실험 | context length | 가설 |
| ---: | ---: | --- |
| E1 | 64 | 짧은 문맥이 현재 데이터와 모델에는 효율적일 수 있다 |
| E2 | 128 | 더 긴 문맥이 도움이 될 수 있지만 step 수가 줄 수 있다 |
| E3 | 256 | 긴 문맥은 좋을 수 있지만 현재 모델/학습량에는 부담일 수 있다 |

주의:

- 현재 코드에는 `--max-steps`가 없다.
- 그래서 context length별 final step이 달라질 수 있다.
- 이 실험은 완전한 공정 비교가 아니라 "현 코드에서 현실적으로 어떤 설정이 효율적인지"를 보는 실험이다.

향후 코드 개선 아이디어:

> `--max-steps` 옵션을 추가하면 batch size나 context length가 달라도 같은 step 수로 공정하게 비교할 수 있다.

E 시리즈 현재 결과 요약:

| 실험 | context length | final step | train loss | val loss | train-val gap | 판정 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1 | 64 | 6324 | 5.0741 | 5.2254 | 0.1513 | 기준값, 안정적인 baseline |
| E2 | 128 | 미제공 | 4.8942 | 5.3675 | 0.4733 | train은 낮지만 val이 악화 |
| E3 | 256 | 미제공 | 4.8999 | 5.4494 | 0.5495 | 가장 긴 문맥, val과 gap 모두 악화 |

E 시리즈 결과 해석:

- `context-length=64`인 E1이 final val loss와 train-val gap 모두 가장 좋다.
- E2/E3는 train loss는 낮지만 validation loss가 크게 높아졌다.
- 현재 코드와 4 epochs 조건에서는 `context-length=64`가 가장 효율적이고 안정적인 선택이다.

### 5.6 학습 전에 읽는 실험 카드

위 표만 보면 "그래서 이게 무슨 실험이지?"라는 느낌이 들 수 있다. 아래 카드는 실제로 학습을 돌리기 전에 읽는 설명이다.

이번 추가 실험 전체의 큰 질문은 하나다.

> 작은 GPT 모델이 NSMC 문장을 더 잘 예측하게 만들려면, 학습 시간, learning rate, dropout, 모델 크기, context length 중 무엇을 어떻게 바꿔야 할까?

각 실험은 이 큰 질문을 더 작은 질문으로 나눠서 확인한다.

#### A1: 좋은 신호를 합친 첫 후보

### 내가 세운 가설

- 관찰: 기존 실험에서 `emb-dim=192`, `n-layers=4`, `lr=5e-4`, `drop-rate=0.0`이 각각 좋은 신호를 보였다.
- 의심: 현재 모델은 과적합보다 학습 부족과 모델 용량 부족에 더 가까울 수 있다.
- 가설: 좋은 신호를 한 번에 합치고 4 epochs 학습하면 기존 4 epochs 기준 모델보다 val loss가 낮아질 것이다.
- 성공 기준: 기존 4 epochs 기준 결과인 val loss `5.6948`보다 낮아지면 성공이다. train-val gap이 너무 커지지 않는지도 함께 본다.

### 실행 옵션

```text
공통 옵션 +
--emb-dim 192
--n-layers 4
--epochs 4
--lr 5e-4
--drop-rate 0.0
```

### 결과

| 항목 | 값 |
| --- | --- |
| final step |  |
| train loss |  |
| val loss |  |
| train-val gap |  |
| 생성 샘플 |  |

### 결과를 보고 바뀐 생각

-

### 다음 실험

- A1이 좋으면 A2로 넘어가서 더 오래 학습해 본다.
- A1이 나쁘면 한 번에 너무 많이 바꾼 것일 수 있으므로 B 실험처럼 lr부터 다시 확인한다.

#### A2: 좋은 후보를 더 오래 학습하기

### 내가 세운 가설

- 관찰: 기존 epochs 실험에서 1, 2, 4, 8, 16 epochs로 갈수록 val loss가 계속 내려갔다.
- 의심: A1 설정도 4 epochs에서 아직 충분히 학습되지 않았을 수 있다.
- 가설: A1과 같은 설정으로 8 epochs까지 학습하면 A1보다 val loss가 더 낮아질 것이다.
- 성공 기준: A1보다 val loss가 낮아지면 성공이다. 단, train-val gap이 크게 커지면 과적합 가능성을 메모한다.

### 실행 옵션

```text
공통 옵션 +
--emb-dim 192
--n-layers 4
--epochs 8
--lr 5e-4
--drop-rate 0.0
```

### 결과

| 항목 | 값 |
| --- | --- |
| final step |  |
| train loss |  |
| val loss |  |
| train-val gap |  |
| 생성 샘플 |  |

### 결과를 보고 바뀐 생각

-

### 다음 실험

- A2가 A1보다 좋으면 장기 학습은 계속 유효하다.
- A2에서 gap이 커지면 A3, A4로 dropout을 다시 확인한다.

#### A3: 약한 dropout으로 장기 학습 안정화 확인

### 내가 세운 가설

- 관찰: 1 epoch에서는 dropout을 끈 `drop-rate=0.0`이 가장 좋았다.
- 의심: 하지만 8 epochs 이상에서는 모델이 학습 데이터에 더 맞춰지면서 과적합이 생길 수 있다.
- 가설: `drop-rate=0.05`는 너무 강하지 않게 과적합을 막아 A2보다 val loss를 낮추거나 train-val gap을 줄일 수 있다.
- 성공 기준: A2보다 val loss가 낮으면 성공이다. val loss가 비슷하더라도 train-val gap이 더 작으면 의미 있는 후보로 둔다.

### 실행 옵션

```text
공통 옵션 +
--emb-dim 192
--n-layers 4
--epochs 8
--lr 5e-4
--drop-rate 0.05
```

### 결과

| 항목 | 값 |
| --- | --- |
| final step |  |
| train loss |  |
| val loss |  |
| train-val gap |  |
| 생성 샘플 |  |

### 결과를 보고 바뀐 생각

-

### 다음 실험

- A3가 A2보다 좋거나 gap이 작으면 dropout 재검증을 계속한다.
- A3가 나쁘면 dropout이 아직은 방해일 수 있다.

#### A4: 기존 dropout 값 재검증

### 내가 세운 가설

- 관찰: 기존 기본값은 `drop-rate=0.1`이었지만 1 epoch에서는 `0.0`보다 나빴다.
- 의심: `0.1`은 짧은 학습에는 강하지만, 긴 학습에서는 과적합을 막는 데 도움이 될 수 있다.
- 가설: 8 epochs에서는 `drop-rate=0.1`이 A2보다 val loss를 낮추거나 train-val gap을 줄일 수 있다.
- 성공 기준: A2보다 val loss가 낮으면 성공이다. A3보다 좋으면 `0.1`을 장기 학습 후보로 둔다.

### 실행 옵션

```text
공통 옵션 +
--emb-dim 192
--n-layers 4
--epochs 8
--lr 5e-4
--drop-rate 0.1
```

### 결과

| 항목 | 값 |
| --- | --- |
| final step |  |
| train loss |  |
| val loss |  |
| train-val gap |  |
| 생성 샘플 |  |

### 결과를 보고 바뀐 생각

-

### 다음 실험

- A2, A3, A4 중 val loss가 가장 낮고 gap이 납득 가능한 설정을 임시 best 후보로 둔다.

#### B1~B5: A3 기준 learning rate 탐색

이 묶음은 모델 구조를 바꾸는 실험이 아니다. 같은 모델을 두고 "얼마나 큰 보폭으로 학습해야 가장 잘 배우는가?"를 확인하는 실험이다.

### 내가 세운 가설

- 관찰: A1-A4 결과에서 `drop-rate=0.05`, `epochs=8`인 A3가 가장 낮은 validation loss를 기록했다.
- 관찰: 기존 learning rate 실험에서는 `5e-4`가 `1e-4`, `3e-4`보다 좋았다.
- 의심: A3 설정에서도 `5e-4`가 최선인지, 아니면 `7e-4`나 `1e-3`처럼 조금 더 큰 lr이 더 빠르게 낮은 val loss에 도달할 수 있는지 확인해야 한다.
- 가설: A3 설정에서 `7e-4`나 `1e-3`은 `5e-4`보다 낮은 val loss를 만들 수 있다. 하지만 `1.5e-3`은 너무 커서 학습이 불안정해질 수 있다.
- 성공 기준: A3의 final val loss `5.1477`, best val loss 약 `5.1326`보다 낮아지면 성공이다. loss가 튀거나 val loss가 나빠지면 너무 큰 lr로 판단한다.

### 실행 옵션

고정값:

```text
공통 옵션 +
--emb-dim 192
--n-layers 4
--epochs 8
--drop-rate 0.05
```

lr만 바꿔서 5번 실행한다.

| 실험 | 바꿀 옵션 | 이 학습이 묻는 질문 |
| --- | --- | --- |
| B1 | `--lr 3e-4` | A3보다 작은 lr이 더 안정적인가, 아니면 느려지는가? |
| B2 | `--lr 5e-4` | A3 결과를 기준값으로 둔다 |
| B3 | `--lr 7e-4` | A3보다 조금 큰 lr이 더 낮은 val loss를 만드는가? |
| B4 | `--lr 1e-3` | 더 공격적인 lr도 버틸 수 있는가? |
| B5 | `--lr 1.5e-3` | 너무 큰 lr의 실패 지점이 어디인지 확인한다 |

참고: B2는 A3와 같은 설정이므로, 같은 조건에서 재실행하지 않고 A3 결과를 재사용할 수 있다.

### 결과

| 실험 | final step | train loss | val loss | train-val gap | 생성 샘플 메모 |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 |  |  |  |  |  |
| B2 | 12648 | 4.5097 | 5.1477 | 0.6380 | A3 결과 재사용 |
| B3 |  |  |  |  |  |
| B4 |  |  |  |  |  |
| B5 |  |  |  |  |  |

### 결과를 보고 바뀐 생각

-

### 다음 실험

- 가장 좋은 lr을 C 실험과 D 실험의 고정값으로 사용한다.
- B4나 B5가 불안정하면 그보다 작은 lr만 후보로 둔다.

#### C1~C3: B5 기준 dropout 장기 재검증

이 묶음은 "B 시리즈에서 가장 좋았던 높은 learning rate로 오래 학습할 때 dropout이 필요한가?"를 확인한다.

### 내가 세운 가설

- 관찰: B1-B5에서는 `lr=1.5e-3`인 B5가 final val loss `5.1793`으로 가장 좋았다.
- 관찰: A1-A4에서는 8 epochs 장기 학습에서 `drop-rate=0.05`인 A3가 가장 좋았다.
- 의심: `lr=1.5e-3`은 4 epochs에서는 좋았지만, 8 epochs로 길어지면 dropout 없이 과적합하거나 validation loss가 흔들릴 수 있다.
- 가설: `lr=1.5e-3`로 8 epochs 학습할 때 `drop-rate=0.05` 또는 `0.1`이 `0.0`보다 val loss나 train-val gap 측면에서 더 좋을 수 있다.
- 성공 기준: C1-C3 중 val loss가 가장 낮은 값을 고른다. val loss가 비슷하면 train-val gap이 더 작은 값을 고른다. 중반 이후 val loss가 다시 올라가면 장기 학습에서 불안정한 설정으로 본다.

### 실행 옵션

고정값:

```text
공통 옵션 +
--emb-dim 192
--n-layers 4
--epochs 8
--lr 1.5e-3
```

drop-rate만 바꿔서 3번 실행한다.

| 실험 | 바꿀 옵션 | 이 학습이 묻는 질문 |
| --- | --- | --- |
| C1 | `--drop-rate 0.0` | B5처럼 dropout 없이 높은 lr이 장기 학습에서도 좋은가? |
| C2 | `--drop-rate 0.05` | A3에서 좋았던 약한 dropout이 높은 lr에서도 가장 균형 좋은가? |
| C3 | `--drop-rate 0.1` | 더 강한 dropout이 높은 lr과 장기 학습에서 과적합을 줄이는가? |

### 결과

| 실험 | final step | train loss | val loss | best val loss | train-val gap | 생성 샘플 메모 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C1 |  |  |  |  |  |  |
| C2 |  |  |  |  |  |  |
| C3 |  |  |  |  |  |  |

### 결과를 보고 바뀐 생각

-

### 다음 실험

- 가장 좋은 drop-rate를 D 실험과 E 실험의 고정값으로 사용한다.

#### D1~D6: C2 기준 모델 크기 비교

이 묶음은 "best learning rate와 best dropout을 적용했을 때 이 문제에는 어느 정도 크기의 모델이 적당한가?"를 확인한다.

### 내가 세운 가설

- 관찰: B 시리즈에서는 `lr=1.5e-3`이 가장 좋았다.
- 관찰: A 시리즈와 C 시리즈에서는 `drop-rate=0.05`가 가장 좋은 dropout 후보였다.
- 관찰: 기존 1차 실험에서는 `emb-dim=192`와 `n-layers=4`가 각각 기준 모델보다 좋았다.
- 의심: 최적화 설정이 좋아진 상태에서는 더 작은 모델도 충분할 수 있고, 반대로 더 큰 모델이 더 낮은 validation loss를 만들 수도 있다.
- 가설: `emb-dim`과 `n-layers`를 키우면 val loss가 더 내려갈 수 있다. 다만 너무 키우면 학습 시간이 늘고 과적합 가능성도 생긴다.
- 성공 기준: D1-D6 중 val loss가 가장 낮은 모델을 고른다. 성능 차이가 작으면 더 작고 빠른 모델을 선택한다.

### 실행 옵션

고정값:

```text
공통 옵션 +
--epochs 4
--lr 1.5e-3
--drop-rate 0.05
--context-length 64
```

모델 크기만 바꿔서 6번 실행한다.

| 실험 | 바꿀 옵션 | 이 학습이 묻는 질문 |
| --- | --- | --- |
| D1 | `--emb-dim 128 --n-layers 2` | 새 lr/dropout에서는 작은 기준 모델도 충분한가? |
| D2 | `--emb-dim 192 --n-layers 2` | 표현 크기만 키우면 좋아지는가? |
| D3 | `--emb-dim 256 --n-layers 2` | 표현 크기를 더 키우면 계속 좋아지는가? |
| D4 | `--emb-dim 128 --n-layers 4` | 깊이만 키우면 좋아지는가? |
| D5 | `--emb-dim 192 --n-layers 4` | A/B/C 시리즈에서 사용한 유력한 모델 크기가 여전히 좋은가? |
| D6 | `--emb-dim 256 --n-layers 4` | 가장 큰 후보가 성능상 이득을 주는가, 아니면 비용만 커지는가? |

### 결과

| 실험 | final step | train loss | val loss | best val loss | train-val gap | 생성 샘플 메모 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| D1 | 6324 | 5.1549 | 5.3315 | 약 5.3262 | 0.1766 | 미기록 |
| D2 | 6324 | 4.9702 | 5.2748 | 약 5.2632 | 0.3046 | 미기록 |
| D3 | 6324 | 4.8166 | 5.2257 | 약 5.2257 | 0.4091 | 미기록 |
| D4 | 6324 | 5.0195 | 5.3227 | 약 5.3161 | 0.3032 | 미기록 |
| D5 | 6324 | 5.0585 | 5.2280 | 약 5.2232 | 0.1695 | 미기록 |
| D6 | 6324 | 4.7450 | 5.2234 | 약 5.2204 | 0.4784 | 미기록 |

### 결과를 보고 바뀐 생각

- D6가 final val loss와 best val loss 기준으로 가장 낮았다.
- 하지만 D6와 D5의 차이는 매우 작고, D6는 train-val gap이 크게 벌어졌다.
- 다음 context length 실험에서는 계산량과 안정성을 고려해 D5 `emb-dim=192`, `n-layers=4`를 현실적인 best로 사용한다.
- D6 `emb-dim=256`, `n-layers=4`는 최종 학습 전에 계산 여유가 있을 때 장기 재검증 후보로 남긴다.

### 다음 실험

- E 실험에서는 D5 `emb-dim=192`, `n-layers=4`를 고정한다.
- context length만 `64`, `128`, `256`으로 바꿔 비교한다.
- D6는 성능상 1위였지만 개선폭이 작으므로 E 실험의 기본값으로는 쓰지 않는다.

#### E1~E3: context length 재검증

이 묶음은 "모델이 한 번에 보는 문맥 길이를 늘리는 것이 실제로 도움이 되는가?"를 확인한다.

### 내가 세운 가설

- 관찰: 기존 실험에서는 `context-length=64`가 가장 좋았다.
- 의심: 하지만 128과 256은 final step이 줄어들었기 때문에 충분히 학습하지 못했을 수 있다.
- 가설: best lr, best dropout, best model size로 다시 비교하면 `context-length=128`이 좋아질 수도 있다. 256은 현재 규모에서는 부담일 수 있다.
- 성공 기준: E1~E3 중 val loss가 가장 낮은 context length를 고른다. 단, final step이 다르면 "성능"과 "효율"을 분리해서 해석한다.

### 실행 옵션

고정값:

```text
공통 옵션에서 --context-length만 제외하고 +
--emb-dim 192
--n-layers 4
--epochs 4
--lr 1.5e-3
--drop-rate 0.05
```

context length만 바꿔서 3번 실행한다.

| 실험 | 바꿀 옵션 | 이 학습이 묻는 질문 |
| --- | --- | --- |
| E1 | `--context-length 64` | 짧은 문맥이 여전히 가장 효율적인가? |
| E2 | `--context-length 128` | 더 긴 문맥이 실제 성능 개선을 주는가? |
| E3 | `--context-length 256` | 긴 문맥이 현재 모델에는 너무 부담인가? |

### 결과

| 실험 | final step | train loss | val loss | train-val gap | 생성 샘플 메모 |
| --- | ---: | ---: | ---: | ---: | --- |
| E1 | 6324 | 5.0741 | 5.2254 | 0.1513 | 미기록 |
| E2 | 미제공 | 4.8942 | 5.3675 | 0.4733 | 미기록 |
| E3 | 미제공 | 4.8999 | 5.4494 | 0.5495 | 미기록 |

### 결과를 보고 바뀐 생각

- E1은 D5와 거의 같은 설정을 `context-length=64` 기준으로 다시 확인한 실험이다.
- final val loss `5.2254`로 D5의 `5.2280`과 거의 같은 수준이라, `context-length=64` baseline은 안정적으로 재현되었다.
- E2는 train eval loss가 `4.8942`로 더 낮지만, val loss가 `5.3675`로 크게 높아졌다.
- 현재까지는 `context-length=128`이 일반화 성능을 개선하지 못했고, train-val gap도 크게 벌어졌다.
- E3도 train eval loss는 `4.8999`로 낮지만, val loss가 `5.4494`로 가장 높고 train-val gap도 `0.5495`로 가장 크다.
- 따라서 현재 조건에서는 `context-length=64`가 가장 안정적이고 효율적인 선택이다.

### 다음 실험

- context length가 바뀌면 final step도 달라질 수 있으므로, 결과를 쓸 때는 "64가 가장 좋다"보다 "현재 코드와 학습량에서는 64가 가장 효율적이다"처럼 조심해서 적는다.
- 최종 후보 설정은 `lr=1.5e-3`, `drop-rate=0.05`, `emb-dim=192`, `n-layers=4`, `context-length=64`로 둔다.
- 최종 보고서에는 C2의 8 epochs 결과와 E1의 context length 재확인 결과를 함께 근거로 사용한다.

## 6. 실험 기록 템플릿

실험을 할 때마다 아래 양식으로 기록한다. 이 양식을 쓰면 나중에 결과를 보고 가설을 세우기 쉬워진다.

````markdown
## 실험 이름

### 내가 세운 가설

- 관찰:
- 의심:
- 가설:
- 성공 기준:

### 실행 옵션

```text
여기에 실행 명령어를 붙여넣기
```

### 결과

| 항목 | 값 |
| --- | --- |
| final step |  |
| train loss |  |
| val loss |  |
| train-val gap |  |
| 생성 샘플 |  |

### 결과를 보고 바뀐 생각

-

### 다음 실험

-
````

## 7. 가설 세우는 법

가설은 어렵게 쓰지 않아도 된다. 좋은 가설은 다음 구조를 가진다.

| 단계 | 질문 | 예시 |
| --- | --- | --- |
| 관찰 | 무엇을 봤나? | `lr=5e-4`가 `3e-4`보다 val loss가 낮았다 |
| 의심 | 왜 그런 결과가 나왔을까? | 기존 lr이 너무 작아서 학습이 느렸을 수 있다 |
| 가설 | 무엇을 바꾸면 어떤 결과가 나올까? | `lr=7e-4`로 올리면 val loss가 더 낮아질 수 있다 |
| 실험 | 무엇을 고정하고 무엇만 바꿀까? | 다른 값은 고정하고 lr만 바꾼다 |
| 성공 기준 | 어떤 숫자가 나오면 성공인가? | 기존 best val loss보다 낮으면 성공 |

### 예시 가설 1: learning rate

관찰:

- `lr=5e-4`가 `1e-4`, `3e-4`보다 좋았다.

의심:

- 기존 learning rate가 너무 작아서 모델이 충분히 빨리 배우지 못했을 수 있다.

가설:

- `lr=7e-4` 또는 `1e-3`을 쓰면 val loss가 더 내려갈 수 있다.

실험:

- `emb-dim=192`, `n-layers=4`, `epochs=4`를 고정하고 lr만 바꾼다.

성공 기준:

- `lr=5e-4`보다 val loss가 낮으면 성공이다.

### 예시 가설 2: dropout

관찰:

- 1 epoch에서는 `drop-rate=0.0`이 가장 좋았다.
- 하지만 16 epochs에서는 train-val gap이 커졌다.

의심:

- 짧게 학습할 때는 dropout이 방해되지만, 오래 학습할 때는 과적합을 줄일 수 있다.

가설:

- 8 epochs 이상에서는 `drop-rate=0.05`가 `0.0`보다 val loss를 안정시킬 수 있다.

실험:

- 같은 모델, 같은 lr, 같은 epochs에서 drop-rate만 `0.0`, `0.05`, `0.1`로 바꾼다.

성공 기준:

- val loss가 낮고 train-val gap이 너무 크지 않으면 성공이다.

### 예시 가설 3: 모델 크기

관찰:

- `emb-dim=192`가 `128`보다 좋았다.
- `n-layers=4`가 `2`보다 좋았다.

의심:

- 현재 모델은 너무 큰 것이 아니라 오히려 표현력이 부족할 수 있다.

가설:

- `emb-dim=192`, `n-layers=4`를 함께 쓰면 val loss가 더 내려갈 수 있다.

실험:

- `128/2`, `192/2`, `128/4`, `192/4`를 비교한다.

성공 기준:

- `192/4`가 기존 기준 모델보다 val loss가 낮으면 성공이다.

## 8. 결과를 읽을 때의 규칙

실험 결과를 볼 때는 아래 순서로 판단한다.

1. val loss가 내려갔는가?
2. train loss도 같이 내려갔는가?
3. train-val gap이 너무 커지지는 않았는가?
4. final step이 비교 대상과 비슷한가?
5. 학습 시간과 계산량을 감당할 수 있는가?

가장 흔한 실수는 val loss만 보고 결론을 너무 빨리 내리는 것이다.

예를 들어 batch size 2는 val loss가 좋았지만 final step이 훨씬 많았다. 이럴 때는 "batch size 2가 좋다"가 아니라 "더 많이 업데이트해서 좋아졌을 수 있다"라고 써야 한다.

context length도 마찬가지다. context length 64가 좋아 보이지만 128과 256은 step 수가 적었다. 그래서 "긴 context는 나쁘다"가 아니라 "현재 조건에서는 64가 가장 효율적으로 보인다"라고 쓰는 것이 더 정확하다.

## 9. 다음에 가장 먼저 할 일

다음에는 지금까지 고른 최종 후보 설정을 8 epochs 이상으로 장기 재검증하거나, 바로 최종 보고서에 정리한다.

| 우선순위 | 실험 | 이유 |
| ---: | --- | --- |
| 1 | 최종 후보 장기 재검증 | `lr=1.5e-3`, `drop-rate=0.05`, `emb-dim=192`, `n-layers=4`, `context-length=64`를 8 epochs 이상으로 확인 |
| 2 | 결과 보고서 작성 | A-E 실험 흐름과 최종 선택 근거를 정리 |
| 3 | `--max-steps` 추가 후 context length 재비교 | context length별 step 수 차이를 통제하고 싶을 때 진행 |

이 결과를 보면 다음 선택이 쉬워진다.

- 최종 후보 장기 재검증이 C2와 비슷하게 나오면 현재 설정을 최종 best로 둔다.
- 장기 재검증에서 gap이 크게 벌어지면 dropout을 `0.1`로 다시 확인할 수 있다.
- context length 결론을 더 엄밀히 내고 싶으면 먼저 `--max-steps`를 추가한다.

## 10. 한 줄 요약

현재 현실적인 best 설정은 `lr=1.5e-3`, `drop-rate=0.05`, `emb-dim=192`, `n-layers=4`, `context-length=64`이다.
