# R1-R7 regularization과 길이 실험 누적 보고서

작성일: 2026-06-04

## 1. 실험 목적

R 시리즈의 목적은 L 시리즈에서 찾은 best full fine-tuning 설정을 기준으로, regularization과 입력 길이가 validation 성능에 어떤 영향을 주는지 확인하는 것이다.

L 시리즈에서 확인한 기준은 다음과 같다.

- L3는 validation accuracy `0.8232`, validation loss `0.3879`로 현재 전체 best다.
- L3는 F1 random init full fine-tuning의 validation accuracy `0.8191`을 처음으로 넘었다.
- L3의 train-val accuracy gap은 `-0.0009`로 작아서, 현재는 강한 과적합 신호가 크지 않다.

따라서 R 시리즈의 핵심 질문은 다음이다.

> L3 설정에서 regularization을 줄이거나 늘리면 성능이 더 좋아지는가?

## 2. 고정 설정

| 항목 | 값 |
| --- | --- |
| pretrained checkpoint | D5 계열 checkpoint |
| freeze mode | full |
| backbone lr | 3e-5 |
| head lr | 1e-3 |
| tokenizer | `data/tokenizer.json` |
| train data | `data/nsmc_sentiment_train.jsonl` |
| validation data | `data/nsmc_sentiment_val.jsonl` |
| context length | 64 |
| emb dim | 192 |
| n heads | 4 |
| n layers | 4 |
| drop rate | 0.05 |
| batch size | 32 |
| epochs | 3 |
| weight decay | 0.01 |
| seed | 42 |

## 3. 실험 설계

### 3.1 classifier dropout

| 실험 | classifier drop rate | 가설 |
| --- | ---: | --- |
| R1 | 0.0 | gap이 작으므로 head dropout이 없어도 될 수 있다 |
| R2 | 0.1 | L3 현재 기준값 |
| R3 | 0.2 | 더 강한 dropout이 validation 안정성을 높일 수 있다 |

classifier dropout 실험의 출발점은 L3의 train-val gap이 이미 작다는 점이다. gap이 작다면 regularization을 더 강하게 주는 것이 꼭 필요하지 않을 수 있다. 그래서 R1은 dropout을 제거해 head가 더 자유롭게 학습될 때 성능이 오르는지 확인하고, R3는 반대로 dropout을 늘려 validation 안정성이 더 좋아지는지 확인한다.

### 3.2 weight decay

| 실험 | weight decay | 가설 |
| --- | ---: | --- |
| R4 | 0.0 | 작은 파인튜닝에서는 weight decay가 없어도 될 수 있다 |
| R5 | 0.01 | L3 현재 기준값 |

weight decay는 full fine-tuning에서 backbone 전체 파라미터에 영향을 주기 때문에, classifier dropout보다 더 넓은 regularization이다. R4는 L3의 좋은 성능이 weight decay 없이도 유지되는지 확인하기 위한 제거 실험이고, R5는 L3 기준값이다.

### 3.3 max length

| 실험 | max length | 가설 |
| --- | ---: | --- |
| R6 | 32 | 짧은 리뷰 중심이면 더 빠르고 충분할 수 있다 |
| R7 | 64 | L3 현재 기준값 |

max length 실험은 regularization이라기보다는 입력 정보량과 계산 비용의 균형을 확인하는 실험이다. NSMC 리뷰가 짧은 경우가 많다면 32 토큰으로도 충분할 수 있지만, 긴 문장이나 반전 표현이 잘리는 경우에는 64 토큰이 유리할 수 있다.

## 4. 실험 결과 요약

R2는 L3와 동일한 설정이므로 L3 결과를 재사용한다.

| 실험 | 상태 | 변경점 | final train loss | final train acc | final val loss | final val acc | acc gap | 판정 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| R1 | 완료 | classifier drop `0.0` | 0.3872 | 0.8227 | 0.3885 | 0.8225 | 0.0002 | R2와 거의 같지만 소폭 낮음 |
| R2 | 완료 | classifier drop `0.1` | 0.3885 | 0.8222 | 0.3879 | 0.8232 | -0.0009 | loss 기준 R3와 동률권 |
| R3 | 완료 | classifier drop `0.2` | 0.3896 | 0.8214 | 0.3879 | 0.8234 | -0.0020 | accuracy 기준 임시 best |
| R4 | 대기 | weight decay `0.0` |  |  |  |  |  | weight decay 제거 확인 필요 |
| R5 | 완료 | weight decay `0.01` | 0.3885 | 0.8222 | 0.3879 | 0.8232 | -0.0009 | 현재 기준값 |
| R6 | 대기 | max length `32` |  |  |  |  |  | 짧은 입력 확인 필요 |
| R7 | 완료 | max length `64` | 0.3885 | 0.8222 | 0.3879 | 0.8232 | -0.0009 | 현재 기준값 |

현재까지 완료된 R1-R3는 classifier dropout만 비교한 결과다. 세 실험 모두 validation accuracy가 `0.8225-0.8234` 범위에 있어 큰 차이는 없지만, dropout을 완전히 제거한 R1이 가장 낮고 `0.1`, `0.2`는 동률권이다. 즉 head dropout은 성능을 방해하지 않았고, 약한 regularization은 유지할 가치가 있다.

R3는 train accuracy가 R2보다 낮으면서 validation accuracy는 조금 높다. 이는 dropout을 늘린 효과가 train set memorization을 아주 약하게 누르고 validation 쪽을 살짝 안정화했을 가능성을 보여준다. 다만 차이가 `0.0002`뿐이므로, 최종 설정으로 확정하려면 seed 반복이 필요하다.

## 5. 가설 검증 요약

| 가설 | 근거 실험 | 결과 | 판정 |
| --- | --- | --- | --- |
| H1. gap이 작으므로 classifier dropout을 꺼도 성능이 유지될 수 있다. | R1 vs R2 | R1은 val acc `0.8225`로 R2 `0.8232`보다 낮고, val loss도 `0.3885`로 더 높았다. | 기각 |
| H2. L3의 classifier dropout `0.1`은 안전한 기본값이다. | R2 | R2는 L3와 동일하게 val acc `0.8232`, val loss `0.3879`를 유지했다. | 충족 |
| H3. 더 강한 classifier dropout은 validation 안정성을 높일 수 있다. | R3 vs R2 | R3는 val acc `0.8234`로 가장 높고 val loss는 R2와 동률이지만, 차이는 매우 작다. | 부분 충족 |
| H4. 현재 설정은 강한 과적합 상태가 아니다. | R1-R3 gap | 세 실험 모두 train-val accuracy gap이 `0.0002`에서 `-0.0020` 사이로 작다. | 충족 |

따라서 classifier dropout 비교의 결론은 "dropout을 끄는 것은 이득이 없고, `0.1`과 `0.2`는 동률권"이다. validation accuracy만 보면 R3가 임시 best지만, R2와의 차이가 너무 작으므로 R 시리즈의 다음 실험은 dropout 값을 더 세밀하게 고르기보다 weight decay와 max length처럼 다른 축을 확인하는 쪽이 더 의미 있다.

## 6. R1 상세 결과

R1은 L3에서 classifier dropout만 `0.1 -> 0.0`으로 바꾼 실험이다.

epoch별 결과는 다음과 같다.

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4890 | 0.7591 | 0.4315 | 0.8007 |
| 2 | 0.4140 | 0.8074 | 0.3991 | 0.8157 |
| 3 | 0.3872 | 0.8227 | 0.3885 | 0.8225 |

R1은 R2/L3와 거의 같은 성능을 냈다. 하지만 validation accuracy는 `0.8232 -> 0.8225`로 아주 조금 낮아졌고, validation loss도 `0.3879 -> 0.3885`로 아주 조금 높아졌다.

따라서 현재 결과만 보면 classifier dropout을 끄는 것은 성능 개선으로 이어지지 않았다. 차이가 매우 작으므로 큰 결론을 내리기는 어렵지만, `classifier drop rate=0.0`을 채택할 근거는 없다.

## 7. R3 상세 결과

R3는 L3/R2에서 classifier dropout만 `0.1 -> 0.2`로 높인 실험이다.

epoch별 결과는 다음과 같다.

| epoch | train loss | train acc | val loss | val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4953 | 0.7545 | 0.4330 | 0.8010 |
| 2 | 0.4181 | 0.8056 | 0.3991 | 0.8162 |
| 3 | 0.3896 | 0.8214 | 0.3879 | 0.8234 |

R3는 validation accuracy 기준으로 R2/L3보다 `0.8232 -> 0.8234`로 아주 조금 높다. validation loss는 `0.3879`로 R2와 반올림 기준 동일하다. train accuracy는 R2보다 낮고 validation accuracy는 조금 높아서 gap은 `-0.0020`이며, 과적합 신호는 커지지 않았다.

따라서 classifier dropout `0.2`는 성능을 망치지 않았고, accuracy 기준으로는 현재 가장 좋은 값이다. 다만 R2와 차이가 `0.0002`뿐이라 단일 seed에서 강하게 우위라고 말하기는 어렵다.

## 8. 현재 결론

현재까지의 결론은 다음과 같다.

> classifier dropout `0.0`은 개선이 없었다. `0.1`과 `0.2`는 거의 동률이며, validation accuracy 기준으로는 R3의 `classifier drop rate=0.2`가 임시 best다.

이 결과는 L3의 성능이 head dropout 때문에 우연히 좋아진 것이라기보다는, backbone lr `3e-5`와 full fine-tuning 자체의 효과가 컸다는 쪽에 가깝다. 다만 R2와 R3의 차이가 매우 작으므로, seed 반복 전까지 classifier dropout `0.1`과 `0.2`의 최종 우열을 강하게 주장하지는 않는다.

## 9. 다음 확인 포인트

1. R4를 실행해 weight decay를 제거해도 성능이 유지되는지 확인한다.
2. R6을 실행해 max length `32`가 속도와 성능 면에서 충분한지 확인한다.
3. R2와 R3의 차이가 작으므로 seed 반복 전에는 `classifier drop rate=0.2`를 임시 best로만 둔다.
4. 현재 gap이 작으므로, regularization을 더 강하게 늘리는 실험은 성능 저하 가능성을 염두에 둔다.
