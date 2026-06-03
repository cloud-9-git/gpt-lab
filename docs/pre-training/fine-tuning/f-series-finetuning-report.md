# F0-F2 사전학습 표현 기준선 확인 실험 최종 보고서

작성일: 2026-06-03

## 1. 실험 목적

이번 F0-F2 실험의 목적은 사전학습된 GPT backbone을 감성 분류에 연결했을 때, 고정된 표현만으로도 NSMC 분류에 도움이 되는지 확인하는 것이다.

처음 F 시리즈는 F3/F4까지 포함해 작은 unfreeze와 full fine-tuning까지 빠르게 확인하는 형태로 계획했다. 하지만 실험을 진행하면서 F 시리즈의 역할을 더 좁게 정리하는 것이 낫다고 판단했다.

> F 시리즈는 기준선과 frozen pretrained representation의 효과만 확인한다. unfreeze 범위 비교는 U 시리즈에서 다룬다.

이렇게 나누면 실험 질문이 더 명확해진다.

- F 시리즈: 사전학습 checkpoint를 고정된 feature extractor로 쓸 가치가 있는가?
- U 시리즈: pretrained backbone을 어느 범위까지 unfreeze해야 하는가?

따라서 F 시리즈는 F2까지 실행하고 종료한다. F3/F4에 해당하던 last block/full fine-tuning 질문은 U3/U5로 넘긴다.

## 2. 실험 설계

| 실험 | backbone 초기화 | 학습 범위 | backbone lr | head lr | epochs | 가설 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| F0 | 없음 | majority baseline | - | - | - | 데이터가 균형이므로 약 50% 기준선 |
| F1 | random init | full model | 1e-4 | 1e-3 | 3 | 사전학습 없이도 어느 정도 학습되는지 확인 |
| F2 | pretrained D5 | classifier only | - | 1e-3 | 3 | 사전학습 표현만으로 감성 분류가 가능한지 확인 |

### 2.1 가설 상세

F 시리즈는 최고 성능을 바로 찾기 위한 실험이 아니라, 이후 U 시리즈의 기준선을 만드는 실험이다. 따라서 각 실험의 가설도 "무엇이 최종 best인가"보다 "비교 기준이 무엇인가"에 초점을 둔다.

| 가설 | 확인 방법 | 성공 기준 | 실패 또는 한계 해석 |
| --- | --- | --- | --- |
| H0. NSMC validation split은 거의 균형이므로 majority baseline은 약 50%일 것이다. | F0 label 분포 계산 | validation accuracy가 약 `0.5` | 50%에서 크게 벗어나면 데이터 split이나 label 분포를 먼저 의심 |
| H1. Random init full fine-tuning은 강한 supervised 기준선이 될 수 있다. | F1 | F0보다 크게 높고, train-val gap이 작음 | 낮으면 classifier/pooling/train loop 문제를 의심 |
| H2. D5 pretrained backbone의 frozen representation은 majority baseline보다 높을 것이다. | F2 vs F0 | F2 validation accuracy가 F0보다 의미 있게 높음 | F0 근처면 checkpoint load, pooling, 사전학습 품질을 의심 |
| H3. Frozen representation만으로는 full fine-tuning 기준선인 F1을 넘기 어렵다. | F2 vs F1 | F2가 F1보다 낮음 | F2가 F1을 넘으면 classifier-only만으로 충분할 수 있음 |
| H4. F2의 train-val gap이 작다면, 낮은 성능은 과적합보다 표현 조정 부족으로 해석한다. | F2 train acc와 val acc 비교 | gap이 작거나 val acc가 train acc와 비슷함 | gap이 크면 head만으로도 과적합이 발생한 것 |

## 3. 고정 설정

F1-F2에서 공통으로 맞춘 설정은 다음과 같다.

| 항목 | 값 |
| --- | --- |
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
| weight decay | 0.01 |
| seed | 42 |

test set은 사용하지 않았다. F 시리즈의 판단은 validation set만으로 한다.

## 4. D5 checkpoint 확인

F2는 D5 사전학습 checkpoint가 필요하다. checkpoint는 아래 설정과 shape가 맞아야 한다.

```text
vocab-size=3000
context-length=64
emb-dim=192
n-heads=4
n-layers=4
drop-rate=0.05
```

초기 확인 시점의 `runs/` 내부 checkpoint는 F2에 바로 사용할 수 없었다.

| 경로 | 확인된 shape | 판정 |
| --- | --- | --- |
| `runs/old-smoke/checkpoint_step_*.pt` | vocab 3000, emb 128, layers 2 | D5가 아님 |
| `runs/pretrain-20260602-180456/checkpoint_step_*.pt` | vocab 800, emb 128, layers 2 | tokenizer/model shape 불일치 |

이후 D5 설정 checkpoint를 확보했고, F2 실행에서 pretrained checkpoint 로드는 성공했다.

## 5. 실험 결과 요약

| 실험 | 상태 | trainable params | final train loss | final train acc | final val loss | final val acc | best val acc | acc gap | F0 대비 | F1 대비 | 판정 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F0 | 완료 | - | - | - | - | 0.5079 | 0.5079 | - | 기준 | -0.3112 | validation majority baseline |
| F1 | 완료 | 2,942,210 | 0.3961 | 0.8202 | 0.3972 | 0.8191 | 0.8191 | 0.0011 | +0.3112 | 기준 | random init full fine-tuning 기준선이 강함 |
| F2 | 완료 | 386 | 0.6017 | 0.6765 | 0.5993 | 0.6796 | 0.6796 | -0.0031 | +0.1718 | -0.1394 | frozen pretrained representation만으로는 F1보다 크게 낮음 |

여기서 `acc gap`은 `train acc - validation acc`이다. F1과 F2 모두 gap이 매우 작으므로, 두 실험의 차이는 단순 과적합 여부보다 학습 가능한 파라미터 범위와 표현 조정 가능성의 차이로 해석하는 것이 자연스럽다.

## 6. 가설 검증 요약

| 가설 | 결과 | 판정 |
| --- | --- | --- |
| H0. validation majority baseline은 약 50%일 것이다. | label 0이 6,094개, label 1이 5,905개였고 majority accuracy는 `0.5079`였다. | 충족 |
| H1. Random init full fine-tuning은 강한 supervised 기준선이 될 수 있다. | F1은 3 epochs 후 validation accuracy `0.8191`을 기록했고 F0보다 `+0.3112` 높았다. | 충족 |
| H2. Frozen pretrained representation은 majority baseline보다 높을 것이다. | F2는 validation accuracy `0.6796`으로 F0보다 `+0.1718` 높았다. | 충족 |
| H3. Frozen representation만으로는 F1을 넘기 어렵다. | F2는 F1보다 validation accuracy가 `0.1394` 낮았다. | 충족 |
| H4. F2의 낮은 성능은 과적합보다 표현 조정 부족으로 볼 수 있다. | F2의 train acc `0.6765`, val acc `0.6796`으로 gap이 `-0.0031`에 불과했다. | 충족 |

가설 검증 결과를 한 문장으로 요약하면 다음과 같다.

> D5 사전학습 표현은 감성 분류에 완전히 무의미하지는 않지만, backbone을 고정한 상태에서는 F1 random init full fine-tuning 수준의 결정 경계를 만들기 어렵다.

## 7. F0 해석

F0은 majority baseline이다.

validation set의 label 분포는 label 0이 6,094개, label 1이 5,905개로 거의 균형이다. 따라서 validation majority accuracy는 `0.5079`이다.

이 값은 이후 실험의 최소 기준선이다. 파인튜닝 결과가 50% 근처에 머문다면 데이터 불균형 문제가 아니라 모델 연결, 학습 설정, pooling, checkpoint 로드 등을 의심해야 한다.

F0의 역할은 단순한 숫자 비교 이상이다. F1/F2가 50%를 넘는지 확인하면 train loop와 label 처리의 기본 동작을 검증할 수 있고, 이후 U 시리즈에서 pretrained 기반 실험이 적어도 무작위 추정보다 의미 있는지를 판단하는 바닥 기준이 된다.

## 8. F1 해석

F1은 사전학습 없이 `emb-dim=192`, `n-layers=4` 모델 전체를 random init 상태에서 감성 분류로 학습한 실험이다.

F1의 사전 가설은 다음과 같았다.

> NSMC 감성 분류는 label이 직접 주어지는 supervised task이므로, 작은 GPT backbone이라도 전체 파라미터를 열고 학습하면 사전학습 없이도 강한 기준선을 만들 수 있다.

epoch별 결과는 다음과 같다.

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5354 | 0.7165 | 0.4594 | 0.7779 |
| 2 | 0.4314 | 0.7976 | 0.4171 | 0.8063 |
| 3 | 0.3961 | 0.8202 | 0.3972 | 0.8191 |

F1의 final validation accuracy는 `0.8191`이고, train-val accuracy gap은 `0.0011`이다.

이는 random init full fine-tuning만으로도 NSMC 감성 분류를 안정적으로 학습할 수 있다는 뜻이다. 특히 gap이 매우 작기 때문에, F1은 단순한 과적합 기준선이 아니라 꽤 강한 비교 기준선이다.

F1의 결과는 이후 pretrained 실험을 해석할 때 중요하다. 사전학습 모델이 의미 있으려면 단순히 F0을 넘는 것만으로는 부족하고, 최소한 F1에 가까워지거나 F1을 넘는지를 봐야 한다. 이번 F 시리즈에서는 F1이 너무 낮은 기준선이 아니라 오히려 강한 기준선으로 나타났기 때문에, F2의 성능 차이를 더 엄격하게 볼 필요가 생겼다.

## 9. F2 해석

F2는 D5 사전학습 backbone을 고정하고 classifier head만 학습한 실험이다.

F2의 사전 가설은 두 갈래였다.

1. 사전학습된 D5 backbone이 한국어 리뷰 문장 구조와 자주 등장하는 표현을 어느 정도 학습했다면, classifier head만 학습해도 F0보다는 높을 것이다.
2. 하지만 next-token prediction으로 배운 표현이 감성 분류 결정 경계에 바로 맞춰진 것은 아니므로, backbone을 완전히 고정하면 F1 full fine-tuning보다는 낮을 가능성이 크다.

epoch별 결과는 다음과 같다.

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6042 | 0.6733 | 0.6009 | 0.6770 |
| 2 | 0.6015 | 0.6765 | 0.6077 | 0.6730 |
| 3 | 0.6017 | 0.6765 | 0.5993 | 0.6796 |

F2의 final validation accuracy는 `0.6796`이다. 이는 F0 baseline보다는 높지만, F1의 `0.8191`보다 크게 낮다.

train accuracy와 validation accuracy가 거의 같기 때문에 과적합 문제는 아니다. 더 자연스러운 해석은 다음과 같다.

> D5 사전학습 backbone에는 감성 분류에 어느 정도 쓸 수 있는 표현이 있지만, backbone을 완전히 고정한 상태에서는 NSMC에 필요한 결정 경계를 충분히 만들기 어렵다.

F2의 결과는 H2와 H3를 동시에 확인한다. F2가 F0보다 `+0.1718` 높다는 점은 사전학습 표현이 완전히 무작위 특징은 아니라는 뜻이다. 그러나 F1보다 `0.1394` 낮다는 점은 frozen representation만으로는 supervised full fine-tuning이 만드는 task-specific 표현을 따라가기 어렵다는 뜻이다.

epoch 흐름도 같은 결론을 지지한다. F2는 1 epoch부터 validation accuracy가 `0.6770`으로 올라왔지만 2~3 epoch에서 거의 정체했다. classifier head만 학습하는 조건에서는 초반에 분리 가능한 신호를 빠르게 주워 담은 뒤, 더 이상 성능을 크게 끌어올릴 표현 자유도가 부족해진 것으로 볼 수 있다.

## 10. F 시리즈에서 남긴 결과물

F 시리즈는 후속 실험에 다음 기준값을 남긴다.

| 기준 | 값 | 의미 |
| --- | ---: | --- |
| F0 majority baseline | 0.5079 | 최소 기준선 |
| F1 random init full fine-tuning | 0.8191 | 사전학습 없이 가능한 강한 supervised 기준선 |
| F2 pretrained classifier only | 0.6796 | frozen pretrained representation 기준선 |
| F2 - F0 | +0.1718 | 사전학습 표현이 baseline보다는 의미 있음 |
| F2 - F1 | -0.1394 | frozen representation만으로는 부족함 |

따라서 F 시리즈 이후의 핵심 질문은 "pretrained가 쓸모 있는가?"에서 "pretrained backbone을 얼마나 풀어야 task-specific 표현으로 바뀌는가?"로 이동한다.

## 11. 최종 결론

F 시리즈의 최종 결론은 다음과 같다.

> 사전학습된 D5 backbone을 frozen feature extractor로만 쓰는 것은 random init full fine-tuning보다 낮다. 따라서 F 시리즈만으로는 사전학습이 downstream 성능을 개선한다고 결론 내릴 수 없다. 다만 F2가 majority baseline은 넘었으므로, 사전학습 표현이 완전히 무의미하다고 보기는 어렵다.

즉, F 시리즈는 "frozen representation만으로 충분한가?"라는 질문에는 부정적인 답을 준다.

조금 더 엄밀하게 쓰면 다음과 같다.

> F2는 사전학습 표현의 존재 가치는 보여주었지만, 사전학습의 최종 효용을 입증하지는 못했다. 사전학습 효과를 주장하려면 classifier head만 학습하는 F2가 아니라, backbone 일부 또는 전체를 task에 맞게 조정하는 U 시리즈 결과까지 함께 봐야 한다.

## 12. 다음 실험으로 넘길 질문

F3/F4로 계획했던 unfreeze 질문은 U 시리즈에서 다룬다.

| 다음 질문 | 넘겨받는 실험 |
| --- | --- |
| classifier only보다 마지막 block을 풀면 좋아지는가? | U3 |
| 더 많은 block을 풀면 F1에 가까워지는가? | U4 |
| pretrained full fine-tuning이 random init full fine-tuning보다 나은가? | U5 |

이미 실행한 U3는 validation accuracy `0.7606`으로 F2보다 좋아졌다. 따라서 pretrained representation은 고정해서 쓰기보다는 task에 맞게 일부 조정할 때 더 잘 작동한다는 방향으로 후속 실험을 이어간다.
