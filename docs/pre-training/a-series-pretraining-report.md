# A1-A4 사전학습 조합 실험 간단 보고서

작성일: 2026-06-03

## 1. 실험 목적

이번 A1-A4 실험의 목적은 기존 1차 실험에서 좋은 신호를 보였던 설정을 합쳤을 때 실제로 validation loss가 개선되는지 확인하는 것이다.

기존 실험에서는 다음 설정들이 각각 좋은 신호를 보였다.

- `emb-dim=192`
- `n-layers=4`
- `lr=5e-4`
- 짧은 학습에서는 `drop-rate=0.0`

따라서 이번 실험에서는 모델 크기와 learning rate를 고정하고, 학습 epoch와 dropout만 바꾸어 비교했다.

## 2. 공통 설정

| 항목 | 값 |
| --- | --- |
| 데이터 | NSMC language modeling 데이터 |
| vocab size | 3000 |
| context length | 64 |
| emb dim | 192 |
| n heads | 4 |
| n layers | 4 |
| batch size | 8 |
| learning rate | 5e-4 |
| eval freq | 20 |
| eval iter | 5 |

## 3. 실험 결과 요약

| 실험 | epochs | drop-rate | final step | final train loss | final val loss | best val loss | final train-val gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 4 | 0.0 | 6324 | 4.9285 | 5.2662 | 약 5.2662 | 0.3377 |
| A2 | 8 | 0.0 | 12648 | 4.1882 | 5.2562 | 약 5.2059 | 1.0680 |
| A3 | 8 | 0.05 | 12648 | 4.5097 | 5.1477 | 약 5.1326 | 0.6380 |
| A4 | 8 | 0.1 | 12648 | 4.6929 | 5.1590 | 약 5.1350 | 0.4661 |

## 4. 결과 해석

### A1: 좋은 신호 조합은 성공

A1은 기존 4 epoch 기준 실험의 validation loss `5.6948`보다 훨씬 낮은 `5.2662`를 기록했다.

따라서 `emb-dim=192`, `n-layers=4`, `lr=5e-4`, `drop-rate=0.0`을 합친 조합은 기존 기준 모델보다 명확히 좋은 후보라고 볼 수 있다.

### A2: 더 오래 학습하면 좋아지지만 과적합 신호가 커짐

A2는 A1과 같은 설정에서 8 epochs까지 학습했다.

final val loss는 `5.2662 -> 5.2562`로 소폭 개선되었고, best val loss는 약 `5.2059`까지 내려갔다. 따라서 더 오래 학습할 가치는 있었다.

다만 final train loss는 크게 낮아졌지만 validation loss 개선은 작았고, train-val gap이 `1.0680`까지 커졌다. 이는 후반부에 학습 데이터에 더 강하게 맞춰지는 과적합 신호로 해석할 수 있다.

### A3: 약한 dropout이 가장 좋은 결과

A3는 A2와 같은 8 epochs 설정에서 `drop-rate=0.05`를 적용했다.

final val loss는 `5.1477`, best val loss는 약 `5.1326`으로 A2보다 확실히 낮았다. train loss는 A2보다 높았지만 validation loss는 더 낮았기 때문에, 약한 dropout이 일반화 성능을 개선한 것으로 볼 수 있다.

### A4: dropout 0.1도 도움은 되지만 A3보다 약간 나쁨

A4는 `drop-rate=0.1`을 적용했다.

final val loss는 `5.1590`, best val loss는 약 `5.1350`으로 A2보다는 좋았지만 A3보다는 약간 나빴다. 대신 train-val gap은 A3보다 작아서 더 강한 regularization 효과는 있었다.

이 결과는 dropout 자체는 장기 학습에서 도움이 되지만, 현재 설정에서는 `0.1`보다 `0.05`가 성능과 규제 강도의 균형이 더 좋다는 뜻으로 해석할 수 있다.

## 5. 결론

이번 A1-A4 실험의 결론은 다음과 같다.

| 질문 | 답 |
| --- | --- |
| 좋은 신호를 합치면 좋아지는가? | 그렇다. A1은 기존 4 epoch 기준보다 크게 개선되었다. |
| 4 epoch보다 8 epoch가 좋은가? | 그렇다. best val loss 기준으로 A2가 A1보다 좋았다. |
| 장기 학습에서 dropout이 도움이 되는가? | 그렇다. A3와 A4 모두 A2보다 낮은 validation loss를 보였다. |
| 현재 가장 좋은 설정은 무엇인가? | `emb-dim=192`, `n-layers=4`, `lr=5e-4`, `drop-rate=0.05`, `epochs=8`인 A3이다. |

최종적으로 이번 묶음에서는 A3가 가장 좋은 후보이다.

```text
emb-dim=192
n-layers=4
lr=5e-4
drop-rate=0.05
epochs=8
```

## 6. 다음 실험 제안

다음 실험에서는 A3 설정을 임시 best로 두고 learning rate를 더 확인하는 것이 좋다.

우선순위는 다음과 같다.

1. `lr=7e-4`, `drop-rate=0.05`, `epochs=8`로 A3보다 더 좋아지는지 확인한다.
2. 시간이 부담되면 `epochs=4`로 learning rate만 먼저 비교한다.
3. 이후 best learning rate가 정해지면 `drop-rate=0.0 / 0.05 / 0.1`을 다시 한 번 재검증한다.

보고서에 쓸 현재 핵심 문장은 다음과 같다.

> 짧은 학습에서는 dropout을 끄는 것이 유리했지만, 8 epochs 장기 학습에서는 dropout을 약하게 적용한 `drop-rate=0.05`가 validation loss를 가장 낮게 만들었다. 이는 모델이 더 오래 학습하면서 학습 데이터에 과하게 맞춰지는 경향이 생겼고, 약한 dropout이 이를 완화했기 때문으로 해석할 수 있다.
