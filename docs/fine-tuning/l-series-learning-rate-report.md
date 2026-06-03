# L1-L4 full fine-tuning learning rate 실험 누적 보고서

작성일: 2026-06-04

## 1. 실험 목적

L 시리즈의 목적은 U 시리즈에서 가장 높은 validation 성능을 낸 U5 full fine-tuning 설정을 기준으로, backbone learning rate가 적절한지 확인하는 것이다.

U 시리즈에서 확인한 기준은 다음과 같다.

- U5 full fine-tuning은 validation accuracy `0.7995`, validation loss `0.4286`으로 U 시리즈 최고 성능을 냈다.
- 하지만 F1 random init full fine-tuning의 validation accuracy `0.8191`에는 아직 낮다.

따라서 L 시리즈의 핵심 질문은 다음이다.

> D5 full fine-tuning에서 backbone learning rate를 조정하면 F1 기준선에 더 가까워질 수 있는가?

여기서 중요한 비교 대상은 단순히 train loss가 빨리 내려가는 설정이 아니다. 사전학습 checkpoint를 쓰는 목적은 이미 학습된 언어 표현을 NSMC 감성 분류에 맞게 조정하는 것이므로, L 시리즈는 "backbone을 충분히 움직이되 망가뜨리지 않는 learning rate"를 찾는 실험이다.

## 2. 고정 설정

| 항목 | 값 |
| --- | --- |
| pretrained checkpoint | D5 계열 checkpoint |
| freeze mode | full |
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
| head lr | 실험별 설정 |
| weight decay | 0.01 |
| seed | 42 |

## 3. 실험 설계

| 실험 | backbone lr | head lr | 가설 |
| --- | ---: | ---: | --- |
| L1 | 5e-6 | 1e-3 | U5보다 더 보수적으로 학습하면 안정적일 수 있다 |
| L2 | 1e-5 | 1e-3 | U5 현재 기준값 |
| L3 | 3e-5 | 1e-3 | 더 빠르게 좋아질 수 있지만 backbone 손상 위험이 있다 |
| L4 | 1e-5 | 3e-4 | head lr을 낮추면 더 안정적일 수 있다 |

각 실험의 역할은 다르다. L1은 underfitting 확인용, L2는 U5 기준선, L3는 backbone을 더 적극적으로 task에 맞추는 실험, L4는 backbone lr은 유지한 채 classifier head의 업데이트만 완만하게 만드는 실험이다. 따라서 L3가 train accuracy만 높이고 validation이 떨어지면 "lr이 과했다"고 보고, L1/L4가 느리지만 validation loss가 낮으면 "더 긴 epochs에서 재확인할 후보"로 본다.

## 4. 실험 결과 요약

L2는 U5와 동일한 설정이므로 U5 결과를 재사용한다.

| 실험 | 상태 | backbone lr | head lr | final train loss | final train acc | final val loss | final val acc | acc gap | 판정 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| L1 | 완료 | 5e-6 | 1e-3 | 0.4800 | 0.7670 | 0.4599 | 0.7831 | -0.0161 | 안정적이지만 L2보다 낮음 |
| L2 | 완료 | 1e-5 | 1e-3 | 0.4408 | 0.7910 | 0.4286 | 0.7995 | -0.0084 | 기준값, L4보다 낮음 |
| L3 | 완료 | 3e-5 | 1e-3 | 0.3885 | 0.8222 | 0.3879 | 0.8232 | -0.0009 | 현재 best, F1 초과 |
| L4 | 완료 | 1e-5 | 3e-4 | 0.4372 | 0.7935 | 0.4267 | 0.8024 | -0.0089 | L2보다 좋지만 L3보다 낮음 |

결과를 learning rate 축으로 보면 L1 -> L2 -> L3로 갈수록 train/validation 성능이 함께 좋아졌다. 이는 U5/L2의 `backbone lr=1e-5`가 pretrained backbone을 보존하기에는 안전했지만, NSMC 감성 분류에 맞게 표현을 바꾸기에는 부족했다는 신호다.

반대로 L3는 train accuracy만 단독으로 오른 것이 아니라 validation loss와 validation accuracy도 함께 개선됐다. train-val accuracy gap도 `-0.0009`로 작기 때문에, 현재 범위에서는 `3e-5`가 backbone을 망가뜨렸다기보다는 task adaptation을 충분히 만든 설정으로 해석한다.

L4는 head lr을 낮춘 효과를 확인해 준다. L4가 L2보다 좋아졌다는 점은 classifier head 업데이트가 조금 과했을 가능성을 보여주지만, L3와의 큰 차이를 보면 병목은 head lr보다 backbone lr 쪽에 더 컸다.

## 5. 가설 검증 요약

L 시리즈는 U5에서 남은 질문, 즉 "full fine-tuning은 되는데 learning rate가 너무 보수적인가?"를 검증한 실험이다.

핵심 가설과 충족 여부는 다음과 같다.

| 가설 | 근거 실험 | 결과 | 판정 |
| --- | --- | --- | --- |
| H1. backbone lr을 더 낮추면 안정적일 수 있지만, 학습이 부족할 수 있다. | L1 vs L2 | L1은 안정적으로 내려갔지만 val acc `0.7831`로 L2 `0.7995`보다 낮았다. | 충족 |
| H2. U5/L2의 `backbone lr=1e-5`, `head lr=1e-3`은 개선 여지가 있다. | L3, L4 vs L2 | L3 `0.8232`, L4 `0.8024`가 모두 L2 `0.7995`보다 높았다. | 충족 |
| H3. backbone lr을 `3e-5`로 키우면 표현을 더 잘 조정할 수 있지만 손상 위험이 있다. | L3 | val acc `0.8232`, val loss `0.3879`, gap `-0.0009`로 성능이 크게 올랐고 손상 신호는 작았다. | 충족 |
| H4. head lr을 낮추면 더 안정적으로 좋아질 수 있다. | L4 vs L2 | L4가 L2보다 acc/loss 모두 소폭 개선됐지만 L3에는 크게 못 미쳤다. | 부분 충족 |
| H5. learning rate를 조정하면 사전학습 기반 full fine-tuning이 F1 random full을 넘을 수 있다. | L3 vs F1 | L3 `0.8232`가 F1 `0.8191`을 넘었고, val loss도 `0.3879`로 F1 `0.3972`보다 낮았다. | 충족 |

가설을 종합하면, L 시리즈는 "사전학습 backbone이 효과가 없는가?"가 아니라 "얼마나 적극적으로 backbone을 업데이트해야 하는가?"의 문제였음을 보여준다. F2처럼 backbone을 고정하면 pretrained representation만으로는 부족했고, U5처럼 full fine-tuning을 해도 lr이 낮으면 F1을 넘지 못했다. L3에서 backbone lr을 높였을 때 처음으로 F1을 넘어섰기 때문에, D5 checkpoint는 적절한 fine-tuning 조건에서 downstream task에 도움이 된다고 해석할 수 있다.

따라서 L 시리즈의 핵심 결론은 "D5 사전학습 checkpoint는 full fine-tuning에서 충분히 큰 backbone lr을 줄 때 random init full fine-tuning을 넘어설 수 있다"는 것이다. 단, 이 결론은 아직 validation set의 단일 seed 결과이므로 seed 반복과 test set 최종 평가 전까지는 최종 일반화 성능으로 확정하지 않는다.

## 6. L1 상세 결과

L1은 U5보다 backbone learning rate를 낮춘 `5e-6` 설정이다.

epoch별 결과는 다음과 같다.

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5788 | 0.6942 | 0.5287 | 0.7356 |
| 2 | 0.5148 | 0.7447 | 0.4849 | 0.7653 |
| 3 | 0.4800 | 0.7670 | 0.4599 | 0.7831 |

L1은 loss가 안정적으로 내려갔지만, L2/U5보다 validation accuracy와 validation loss가 모두 나쁘다. validation accuracy는 `0.7995 -> 0.7831`로 낮아졌고, validation loss는 `0.4286 -> 0.4599`로 높아졌다.

따라서 `5e-6`은 안정적이지만 너무 보수적이어서 3 epochs 안에 충분히 적응하지 못한 설정으로 보는 것이 자연스럽다.

## 7. L3 상세 결과

L3는 backbone learning rate를 `3e-5`로 높이고, classifier head learning rate는 `1e-3`으로 유지한 설정이다.

epoch별 결과는 다음과 같다.

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4922 | 0.7572 | 0.4316 | 0.8011 |
| 2 | 0.4162 | 0.8068 | 0.3986 | 0.8162 |
| 3 | 0.3885 | 0.8222 | 0.3879 | 0.8232 |

L3는 L4보다 validation accuracy가 `0.8024 -> 0.8232`로 크게 올랐고, validation loss도 `0.4267 -> 0.3879`로 낮아졌다. train-val accuracy gap은 `-0.0009`로 매우 작아서, backbone lr을 키웠지만 과적합 신호가 크지 않다.

또한 L3는 F1 random init full fine-tuning의 validation accuracy `0.8191`을 처음으로 넘어섰다. validation loss도 F1의 `0.3972`보다 낮은 `0.3879`이므로, 현재까지는 사전학습 D5 checkpoint를 사용한 full fine-tuning이 random init full fine-tuning보다 더 좋은 설정을 찾았다고 볼 수 있다.

## 8. L4 상세 결과

L4는 L2/U5와 같은 backbone learning rate `1e-5`를 유지하고, classifier head learning rate만 `1e-3 -> 3e-4`로 낮춘 설정이다.

epoch별 결과는 다음과 같다.

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5496 | 0.7161 | 0.4831 | 0.7666 |
| 2 | 0.4678 | 0.7765 | 0.4442 | 0.7906 |
| 3 | 0.4372 | 0.7935 | 0.4267 | 0.8024 |

L4는 L2/U5보다 validation accuracy가 `0.7995 -> 0.8024`로 소폭 상승했고, validation loss도 `0.4286 -> 0.4267`로 낮아졌다. train-val accuracy gap은 `-0.0089`로 작아서 과적합 신호도 크지 않다.

따라서 head lr을 `1e-3`에서 `3e-4`로 낮추는 것은 L2 대비 약간 더 좋은 결과를 냈다. 하지만 L3가 더 높은 validation accuracy와 더 낮은 validation loss를 기록했으므로, L4는 현재 best가 아니라 안정적인 보조 후보로 둔다.

## 9. 현재 결론

현재까지의 결론은 다음과 같다.

> L1은 너무 보수적이었고, L4는 L2/U5보다 소폭 좋았다. 그러나 L3는 L4보다 크게 좋아졌고 F1 random init full fine-tuning도 넘어섰다. 따라서 현 시점의 best learning rate는 L3의 `backbone lr=3e-5`, `head lr=1e-3`이다.

이 결과는 U 시리즈 결론을 보강한다. U 시리즈에서는 "unfreeze할수록 좋아지지만 아직 F1을 넘지 못했다"가 결론이었다. L 시리즈에서는 같은 full fine-tuning이라도 learning rate를 조정하면 F1을 넘을 수 있음을 확인했다. 즉 문제는 사전학습 backbone 자체가 쓸모없다는 것이 아니라, pretrained backbone을 충분히 빠르게 task에 적응시키는 learning rate 설정이 필요했다는 쪽에 가깝다.

현재 L3의 train-val accuracy gap은 `-0.0009`로 작다. 따라서 지금 단계에서는 과적합을 줄이기 위한 강한 regularization보다, L3 설정을 유지한 채 dropout, weight decay, max length가 성능을 망치지 않는지 확인하는 쪽이 더 중요하다.

## 10. 다음 확인 포인트

1. L3를 기준 설정으로 두고 R 시리즈에서 regularization과 max length를 확인한다.
2. L3가 F1을 넘은 폭은 `+0.0041` accuracy로 크지는 않으므로, seed 반복에서 재현되는지 확인한다.
3. L3의 train-val gap이 작기 때문에 현재로서는 과적합보다 추가 regularization이 성능을 낮출 가능성도 있다.
4. L4는 더 보수적인 대안 후보로 남겨둔다.
5. L3가 seed 반복에서 흔들리면 L4 또는 더 낮은 backbone lr로 되돌아간다.
