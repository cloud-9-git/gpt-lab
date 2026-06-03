# C1-C3 dropout 장기 재검증 실험 누적 보고서

작성일: 2026-06-03

## 1. 실험 목적

이번 C1-C3 실험의 목적은 B 시리즈에서 가장 좋았던 learning rate `1.5e-3`을 8 epochs 장기 학습에 사용할 때, dropout이 validation loss를 안정시키는지 확인하는 것이다.

## 2. 고정 설정

| 항목 | 값 |
| --- | --- |
| emb dim | 192 |
| n layers | 4 |
| learning rate | 1.5e-3 |
| epochs | 8 |

## 3. 실험 결과 요약

| 실험 | drop-rate | final step | final train loss | final val loss | best val loss | final train-val gap | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C1 | 0.0 | 12648 | 4.4708 | 5.0812 | 약 5.0646 | 0.6104 | 장기 학습 개선, C2에는 근소하게 밀림 |
| C2 | 0.05 | 12648 | 4.6668 | 5.0798 | 약 5.0440 | 0.4130 | 현재 1위, val loss와 gap 모두 개선 |
| C3 | 0.1 | 12648 | 4.8119 | 5.1151 | 약 5.0843 | 0.3032 | gap은 작지만 val loss가 높아 C2보다 나쁨 |

## 4. 현재 해석

C1은 `lr=1.5e-3`, `drop-rate=0.0`, `epochs=8`을 사용한 실험이다.

final val loss는 `5.0812`, best val loss는 약 `5.0646`으로, B5의 final val loss `5.1793`보다 더 낮다. 따라서 B5에서 찾은 높은 learning rate는 4 epochs뿐 아니라 8 epochs 장기 학습에서도 효과가 있었다.

다만 final train-val gap은 `0.6104`로 B5보다 커졌다. 이는 학습이 오래 진행되면서 train loss가 더 많이 내려간 결과이며, 장기 학습에서 과적합 가능성은 계속 확인해야 한다.

현재까지의 결론:

> `lr=1.5e-3`, `epochs=8` 조건에서는 `drop-rate=0.05`가 현재 가장 좋은 후보이다. C1보다 final val loss는 아주 조금 낮고, best val loss와 train-val gap은 더 분명하게 좋다.

C2는 `drop-rate=0.05`를 적용한 실험이다. final val loss는 `5.0798`, best val loss는 약 `5.0440`으로 C1보다 낮다. final train loss는 C1보다 높지만, validation loss와 train-val gap이 모두 개선되었기 때문에 약한 dropout이 일반화에 도움을 준 것으로 해석할 수 있다.

C3는 `drop-rate=0.1`을 적용한 실험이다. final val loss는 `5.1151`, best val loss는 약 `5.0843`으로 C1/C2보다 높다. train-val gap은 가장 작지만, validation loss가 나빠졌으므로 현재 설정에서는 dropout이 너무 강해 학습을 방해한 것으로 해석할 수 있다.

## 5. C 시리즈 최종 결론

| 순위 | 실험 | drop-rate | final val loss | best val loss | train-val gap |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | C2 | 0.05 | 5.0798 | 약 5.0440 | 0.4130 |
| 2 | C1 | 0.0 | 5.0812 | 약 5.0646 | 0.6104 |
| 3 | C3 | 0.1 | 5.1151 | 약 5.0843 | 0.3032 |

최종적으로 C 시리즈에서는 `drop-rate=0.05`가 가장 좋은 후보이다.

해석:

- `drop-rate=0.0`은 train loss를 가장 낮게 만들었지만 train-val gap이 가장 컸다.
- `drop-rate=0.05`는 final val loss와 best val loss가 가장 낮았고, gap도 C1보다 줄었다.
- `drop-rate=0.1`은 gap은 가장 작았지만 val loss가 높아져 과하게 regularization된 것으로 보인다.

따라서 `lr=1.5e-3`, `epochs=8` 조건에서는 `drop-rate=0.05`가 성능과 일반화 균형이 가장 좋다.

## 6. 다음 확인 포인트

- D/E 실험에서는 `lr=1.5e-3`, `drop-rate=0.05`를 고정값으로 사용하는 것이 좋다.
- A 시리즈와 C 시리즈 모두에서 `drop-rate=0.05`가 가장 좋았으므로, 현재 모델 크기에서는 약한 dropout이 가장 안정적인 후보이다.
