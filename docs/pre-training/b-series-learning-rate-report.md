# B1-B5 learning rate 탐색 실험 누적 보고서

작성일: 2026-06-03

## 1. 실험 목적

이번 B1-B5 실험의 목적은 `5e-4`보다 더 좋은 learning rate가 있는지 확인하는 것이다.

## 2. 고정 설정

| 항목 | 값 |
| --- | --- |
| emb dim | 192 |
| n layers | 4 |
| drop rate | 0.0 |
| epochs | 4 |

## 3. 실험 결과 요약

| 실험 | lr | final step | final train loss | final val loss | best val loss | final train-val gap | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B1 | 3e-4 | 6324 | 5.1312 | 5.3469 | 약 5.3446 | 0.2157 | B2보다 느리고 val loss가 높음 |
| B2 | 5e-4 | 6324 | 4.9285 | 5.2662 | 약 5.2508 | 0.3377 | 기준값, A1과 같은 조건에서 결과 확인 |
| B3 | 7e-4 | 6324 | 4.8473 | 5.2155 | 약 5.2085 | 0.3682 | B2보다 개선, B4에는 밀림 |
| B4 | 1e-3 | 6324 | 4.8937 | 5.2010 | 약 5.1947 | 0.3073 | B3보다 개선, B5에는 밀림 |
| B5 | 1.5e-3 | 6324 | 4.9818 | 5.1793 | 약 5.1793 | 0.1975 | 최종 1위, 불안정해지지 않음 |

## 4. 현재 해석

B1은 `lr=3e-4`를 사용한 실험이다. final val loss는 `5.3469`이고, best val loss는 약 `5.3446`이다.

B2의 final val loss `5.2662`, best val loss 약 `5.2508`보다 B1이 높으므로, 현재까지는 `3e-4`가 `5e-4`보다 좋지 않다. train-val gap은 B1이 더 작지만, 이는 일반화가 더 좋다기보다 learning rate가 작아서 학습 자체가 덜 진행된 결과로 보는 것이 자연스럽다.

B3은 `lr=7e-4`를 사용한 실험이다. final val loss는 `5.2155`, best val loss는 약 `5.2085`로 B2보다 낮다. train loss도 `4.9285 -> 4.8473`으로 내려갔기 때문에, `7e-4`는 단순히 validation만 우연히 좋아진 것이 아니라 학습 자체도 더 진행된 결과로 볼 수 있다.

B4는 `lr=1e-3`을 사용한 실험이다. final val loss는 `5.2010`, best val loss는 약 `5.1947`로 B3보다 더 낮다. train loss는 B3보다 약간 높지만 validation loss와 train-val gap이 모두 더 좋아서, 현재 조건에서는 `1e-3`이 가장 좋은 learning rate 후보이다.

B5는 `lr=1.5e-3`을 사용한 실험이다. final val loss는 `5.1793`, best val loss도 약 `5.1793`으로 B 시리즈 전체에서 가장 낮다. train loss는 B4보다 높지만 validation loss가 더 낮고 train-val gap도 가장 작아서, 이 조건에서는 `1.5e-3`이 불안정하지 않고 오히려 일반화 성능이 가장 좋았다.

최종 결론:

> `emb-dim=192`, `n-layers=4`, `drop-rate=0.0`, `epochs=4` 조건에서는 `lr=1.5e-3`이 가장 좋은 후보이다.

## 5. B 시리즈 결론

learning rate를 `3e-4 -> 5e-4 -> 7e-4 -> 1e-3 -> 1.5e-3`로 키울수록 final validation loss가 계속 낮아졌다.

| lr 변화 | final val loss |
| ---: | ---: |
| 3e-4 | 5.3469 |
| 5e-4 | 5.2662 |
| 7e-4 | 5.2155 |
| 1e-3 | 5.2010 |
| 1.5e-3 | 5.1793 |

따라서 이번 범위에서는 learning rate가 너무 커져서 실패하는 지점이 아직 나타나지 않았다. 다만 B5의 train loss는 B3/B4보다 높았으므로, 더 큰 lr을 바로 채택하기보다는 장기 학습과 dropout 조건에서 다시 확인하는 것이 좋다.

## 6. 다음 실험 제안

다음 C 시리즈에서는 B5의 best lr인 `1.5e-3`을 사용해 장기 학습에서 dropout을 재검증한다.

고정값:

```text
emb-dim=192
n-layers=4
lr=1.5e-3
epochs=8
```

비교할 값:

| 실험 | drop-rate | 목적 |
| --- | ---: | --- |
| C1 | 0.0 | 높은 lr에서 dropout 없이 오래 학습해도 괜찮은지 확인 |
| C2 | 0.05 | A 시리즈에서 좋았던 약한 dropout이 높은 lr에서도 좋은지 확인 |
| C3 | 0.1 | 더 강한 regularization이 장기 학습에서 유리한지 확인 |
