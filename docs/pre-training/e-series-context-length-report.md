# E1-E3 context length 재검증 실험 누적 보고서

작성일: 2026-06-03

## 1. 실험 목적

이번 E1-E3 실험의 목적은 D 시리즈에서 고른 현실적인 모델 크기인 `emb-dim=192`, `n-layers=4`를 고정하고, `context-length`를 다시 비교하는 것이다.

이전 실험에서는 `context-length=64`가 좋아 보였지만, 긴 context length에서는 epoch당 step 수가 줄어들 수 있다. 따라서 이번 실험은 완전한 공정 비교라기보다, 현재 코드와 학습 조건에서 어떤 context length가 가장 효율적인지 확인하는 실험이다.

## 2. 고정 설정

| 항목 | 값 |
| --- | --- |
| learning rate | 1.5e-3 |
| drop rate | 0.05 |
| emb dim | 192 |
| n layers | 4 |
| epochs | 4 |

## 3. 실험 결과 요약

| 실험 | context length | final step | final train loss | final val loss | best val loss | final train-val gap | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1 | 64 | 6324 | 5.0741 | 5.2254 | 약 5.2254 | 0.1513 | 기준값, 안정적인 baseline |
| E2 | 128 | 미제공 | 4.8942 | 5.3675 | 약 5.3675 | 0.4733 | train은 낮지만 val이 악화, 일반화 불리 |
| E3 | 256 | 미제공 | 4.8999 | 5.4494 | 약 5.4494 | 0.5495 | 가장 긴 문맥, val loss와 gap 모두 악화 |

## 4. 현재 해석

E1은 `context-length=64`인 기준 실험이다.

final val loss는 `5.2254`, best val loss도 약 `5.2254`이다. final train-val gap은 `0.1513`으로 작아서, D 시리즈의 D5 결과와 비교해도 안정적인 편이다.

같은 모델 크기와 최적화 설정을 사용했던 D5는 final val loss `5.2280`이었다. E1은 거의 같은 조건의 재현 실험에 가깝고, val loss도 비슷하게 나왔기 때문에 `context-length=64` 기준값은 신뢰할 만한 baseline으로 볼 수 있다.

E2는 `context-length=128`인 실험이다.

epoch 4 log 기준 train eval loss는 `4.8942`로 E1의 `5.0741`보다 낮지만, val loss는 `5.3675`로 E1의 `5.2254`보다 높다. final train-val gap도 `0.4733`으로 E1보다 크게 벌어졌다.

따라서 현재 조건에서는 context length를 `64 -> 128`로 늘리는 것이 학습 데이터에는 더 잘 맞지만 validation 성능에는 도움이 되지 않았다. 이는 긴 문맥 자체가 나쁘다기보다는, 현재 코드에서 context length가 커지며 step 수나 학습 효율이 달라진 영향까지 함께 받은 결과로 해석해야 한다.

E3는 `context-length=256`인 실험이다.

epoch 4 log 기준 train eval loss는 `4.8999`로 E2와 비슷하게 낮지만, val loss는 `5.4494`로 E 시리즈 중 가장 높다. final train-val gap도 `0.5495`로 가장 크다.

따라서 현재 학습 조건에서는 context length를 길게 늘릴수록 train loss는 낮아질 수 있지만 validation loss는 악화되는 흐름이 나타났다. E2/E3의 final step이 제공되지 않았고 context length별 step 수가 달라질 수 있으므로, 결론은 "긴 context가 본질적으로 나쁘다"가 아니라 "현재 코드와 4 epochs 조건에서는 `context-length=64`가 가장 효율적이고 안정적이다"로 적는 것이 안전하다.

현재 결론:

> E 시리즈에서는 `context-length=64`인 E1이 가장 좋은 선택이다. final val loss가 가장 낮고 train-val gap도 가장 작다. 다음 장기 재검증이나 최종 후보 실험은 `context-length=64`를 유지한다.

## 5. 다음 확인 포인트

- 최종 후보 설정은 `lr=1.5e-3`, `drop-rate=0.05`, `emb-dim=192`, `n-layers=4`, `context-length=64`로 둔다.
- 이 설정은 C2에서 이미 8 epochs 장기 학습 결과가 좋았으므로, 최종 보고서에서는 C2와 E1 결과를 함께 근거로 사용할 수 있다.
- 더 공정한 context length 비교가 필요하면 `--max-steps`를 추가한 뒤 64/128/256을 같은 update 수로 다시 비교한다.
