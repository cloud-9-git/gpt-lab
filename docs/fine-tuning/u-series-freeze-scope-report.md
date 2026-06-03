# U1-U5 freeze 범위 비교 실험 누적 보고서

작성일: 2026-06-03
업데이트: 2026-06-04

## 1. 실험 목적

이번 U1-U5 실험의 목적은 D5 사전학습 backbone을 감성 분류에 사용할 때, 어느 범위까지 unfreeze해야 validation 성능이 좋아지는지 확인하는 것이다.

F 시리즈에서 확인한 중요한 기준은 두 가지다.

- F1 random init full fine-tuning은 validation accuracy `0.8191`로 강한 기준선을 만들었다.
- F2 pretrained D5 classifier only는 validation accuracy `0.6796`으로 F1보다 크게 낮았다.

따라서 U 시리즈의 핵심 질문은 다음이다.

> D5 backbone을 완전히 고정하지 않고 일부만 조정하면 F2의 한계를 넘을 수 있는가?

## 2. 고정 설정

| 항목 | 값 |
| --- | --- |
| pretrained checkpoint | D5 계열 checkpoint |
| tokenizer | `data/tokenizer.json` |
| train data | `data/nsmc_sentiment_train.jsonl` |
| validation data | `data/nsmc_sentiment_val.jsonl` |
| max length | 64 |
| context length | 64 |
| emb dim | 192 |
| n heads | 4 |
| n layers | 4 |
| drop rate | 0.05 |
| classifier drop rate | 0.1 |
| batch size | 32 |
| epochs | 3 |
| head lr | 1e-3 |
| weight decay | 0.01 |
| seed | 42 |

## 3. 실험 설계

| 실험 | 학습 범위 | backbone lr | 가설 |
| --- | --- | ---: | --- |
| U1 | classifier only | - | 가장 안정적이지만 성능 한계가 있을 수 있다 |
| U2 | final norm + classifier | 3e-5 | 아주 작은 조정만으로 좋아질 수 있다 |
| U3 | last 1 block + final norm + classifier | 3e-5 | 교재식 기본 후보, 균형이 좋을 가능성이 크다 |
| U4 | last 2 blocks + final norm + classifier | 3e-5 | 더 많이 조정하면 좋아질 수 있지만 과적합 위험도 커진다 |
| U5 | full backbone + classifier | 1e-5 | 전체 조정이 필요할 때만 유리할 수 있다 |

## 4. 실험 결과 요약

U1은 F2와 동일한 classifier-only 설정이므로 F2 결과를 재사용한다.

| 실험 | 상태 | trainable params | final train loss | final train acc | final val loss | final val acc | acc gap | 판정 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| U1 | 완료 | 386 | 0.6017 | 0.6765 | 0.5993 | 0.6796 | -0.0031 | 안정적이지만 성능 한계가 큼 |
| U2 | 대기 |  |  |  |  |  |  | final norm만 조정하는 중간 확인 필요 |
| U3 | 완료 | 445,058 | 0.5133 | 0.7439 | 0.4915 | 0.7606 | -0.0168 | U1보다 크게 개선, F1에는 미달 |
| U4 | 완료 | 889,346 | 0.4508 | 0.7852 | 0.4356 | 0.7960 | -0.0108 | U3보다 개선, 효율 후보 |
| U5 | 완료 | 2,942,210 | 0.4408 | 0.7910 | 0.4286 | 0.7995 | -0.0084 | U4보다 소폭 개선, 현재 최고 성능 |

## 5. 가설 검증 요약

U 시리즈는 단순히 freeze 범위를 바꿔본 것이 아니라, F 시리즈에서 생긴 의문을 단계적으로 검증한 실험이다.

핵심 가설과 충족 여부는 다음과 같다.

| 가설 | 근거 실험 | 결과 | 판정 |
| --- | --- | --- | --- |
| H1. 사전학습 backbone을 완전히 고정하고 classifier만 학습하면 성능 한계가 있을 것이다. | U1/F2 | validation accuracy `0.6796`으로 F1 `0.8191`보다 크게 낮았다. | 충족 |
| H2. 마지막 block을 unfreeze하면 classifier only보다 좋아질 것이다. | U3 vs U1 | validation accuracy가 `0.6796 -> 0.7606`으로 상승했다. | 충족 |
| H3. 더 많은 block을 unfreeze하면 validation 성능이 더 좋아질 수 있다. | U4 vs U3 | validation accuracy가 `0.7606 -> 0.7960`, validation loss가 `0.4915 -> 0.4356`으로 개선됐다. | 충족 |
| H4. full fine-tuning은 과적합 위험이 있지만, 작은 lr이면 성능을 더 끌어올릴 수 있다. | U5 vs U4 | validation accuracy가 `0.7960 -> 0.7995`로 소폭 상승했고 gap도 `-0.0084`로 작았다. | 부분 충족 |
| H5. D5 사전학습 backbone을 잘 unfreeze하면 random init full fine-tuning을 이길 수 있다. | U5 vs F1 | U5 `0.7995`가 F1 `0.8191`보다 낮았다. | 미충족 |

따라서 U 시리즈가 충족한 것은 "pretrained D5는 frozen feature extractor로는 부족하지만, task에 맞게 unfreeze할수록 성능이 개선된다"는 가설이다. 반대로 아직 충족하지 못한 것은 "사전학습 기반 fine-tuning이 random init full fine-tuning보다 우수하다"는 더 강한 주장이다.

## 6. U3 상세 결과

U3는 D5 backbone에서 마지막 Transformer block, final norm, classifier head만 학습한 실험이다.

epoch별 결과는 다음과 같다.

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5819 | 0.6913 | 0.5374 | 0.7296 |
| 2 | 0.5358 | 0.7287 | 0.5091 | 0.7481 |
| 3 | 0.5133 | 0.7439 | 0.4915 | 0.7606 |

U3는 U1/F2보다 validation accuracy가 `0.6796 -> 0.7606`으로 크게 올랐다. 따라서 마지막 block을 태스크에 맞게 조정하는 것은 효과가 있다.

하지만 F1 random init full fine-tuning의 validation accuracy `0.8191`에는 아직 미치지 못한다. 현재 결과만 보면 D5 사전학습 backbone은 완전히 고정했을 때보다 일부 unfreeze했을 때 훨씬 낫지만, 아직 random init full fine-tuning보다 우수하다고 말하기는 어렵다.

## 7. U4 상세 결과

U4는 D5 backbone에서 마지막 2개 Transformer block, final norm, classifier head를 학습한 실험이다.

epoch별 결과는 다음과 같다.

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5499 | 0.7164 | 0.4869 | 0.7608 |
| 2 | 0.4813 | 0.7654 | 0.4518 | 0.7831 |
| 3 | 0.4508 | 0.7852 | 0.4356 | 0.7960 |

U4는 U3보다 validation accuracy가 `0.7606 -> 0.7960`으로 상승했고, validation loss도 `0.4915 -> 0.4356`으로 낮아졌다. train-val accuracy gap은 `-0.0108`로 작아서, 더 많이 unfreeze했지만 과적합 신호는 아직 크지 않다.

다만 F1 random init full fine-tuning의 validation accuracy `0.8191`에는 아직 `0.0231` 낮다. 따라서 U4는 pretrained D5 기반 실험 중 강한 후보지만, 사전학습 모델이 random init full fine-tuning을 넘어섰다고 결론 내리기는 이르다.

## 8. U5 상세 결과

U5는 D5 backbone 전체와 classifier head를 함께 학습한 실험이다. backbone learning rate는 더 보수적인 `1e-5`를 사용했다.

epoch별 결과는 다음과 같다.

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5484 | 0.7176 | 0.4840 | 0.7657 |
| 2 | 0.4714 | 0.7738 | 0.4457 | 0.7877 |
| 3 | 0.4408 | 0.7910 | 0.4286 | 0.7995 |

U5는 U4보다 validation accuracy가 `0.7960 -> 0.7995`로 소폭 상승했고, validation loss도 `0.4356 -> 0.4286`으로 낮아졌다. train-val accuracy gap은 `-0.0084`로 작아서, full fine-tuning을 했지만 이번 설정에서는 과적합 신호가 크지 않다.

다만 U4 대비 개선 폭은 validation accuracy 기준 `+0.0035`로 작고, 학습 파라미터는 `889,346 -> 2,942,210`으로 크게 늘어난다. 따라서 최고 성능만 보면 U5가 현재 best지만, 계산 효율과 단순성을 고려하면 U4도 여전히 현실적인 후보로 남는다.

F1 random init full fine-tuning의 validation accuracy `0.8191`에는 아직 `0.0196` 낮다. 따라서 U5까지 확인한 뒤에도, 현재 D5 사전학습 checkpoint가 random init full fine-tuning을 이겼다고 말하기는 어렵다.

## 9. 현재 결론

현재까지의 결론은 다음과 같다.

> U5는 U1/F2, U3, U4를 모두 넘어섰다. 즉, pretrained D5 representation은 고정된 feature로 쓰기보다 task에 맞게 backbone을 조정할 때 더 잘 작동한다. 현재 validation 성능 기준 U 시리즈 best는 U5다. 다만 U4 대비 개선 폭은 작고 F1 기준선보다도 낮으므로, 사전학습 효과를 최종적으로 주장하려면 U5/U4 기반 learning rate 조정이나 seed 반복이 필요하다.

## 10. 다음 확인 포인트

1. U2를 실행하면 final norm만 조정해도 U1보다 좋아지는지 확인할 수 있다.
2. 최고 validation 성능을 우선하면 U5를 기준 후보로 둔다.
3. 계산 효율과 단순성을 우선하면 U4를 현실적인 후보로 둔다.
4. L 시리즈에서는 U5의 full fine-tuning learning rate를 먼저 조정하거나, U4와 U5를 각각 한 번씩 seed 반복해 차이가 우연인지 확인한다.
5. 여전히 F1보다 낮게 유지되면 사전학습 데이터, checkpoint 품질, 파인튜닝 learning rate 전략을 다시 의심한다.
