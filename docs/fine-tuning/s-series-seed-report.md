# S1-S3 seed 반복 실험 보고서

작성일: 2026-06-04

## 1. 실험 목적

S 시리즈의 목적은 R 시리즈에서 임시 best로 고른 설정이 특정 seed에서만 우연히 좋았는지 확인하는 것이다.

R 시리즈까지의 최종 후보는 다음 설정이다.

| 항목 | 값 |
| --- | --- |
| pretrained checkpoint | D5 계열 checkpoint |
| freeze mode | full |
| backbone lr | 3e-5 |
| head lr | 1e-3 |
| classifier drop rate | 0.2 |
| weight decay | 0.01 |
| max length | 64 |
| context length | 64 |
| epochs | 3 |
| batch size | 32 |

핵심 질문은 다음이다.

> R3 설정은 seed가 바뀌어도 F1 random init full fine-tuning보다 안정적으로 좋은가?

비교 기준은 F1 validation accuracy `0.8191`, validation loss `0.3972`다.

## 2. 실험 설계

| 실험 | seed | run name | 가설 |
| --- | ---: | --- | --- |
| S1 | 42 | R3 결과 재사용 | R3 단일 seed best가 기준이 된다 |
| S2 | 43 | `s2-d5-full-blr3e-5-cdrop0-2-seed43` | seed가 바뀌어도 F1보다 높아야 한다 |
| S3 | 44 | `s3-d5-full-blr3e-5-cdrop0-2-seed44` | 세 번째 seed에서도 성능이 유지되어야 한다 |

S1은 R3와 같은 설정과 seed이므로 R3 결과를 재사용한다.

## 3. 실험 결과

| 실험 | seed | final train loss | final train acc | final val loss | final val acc | acc gap | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S1 | 42 | 0.3896 | 0.8214 | 0.3879 | 0.8234 | -0.0020 | 단일 seed 최고 |
| S2 | 43 | 0.3898 | 0.8227 | 0.3908 | 0.8202 | 0.0025 | 가장 낮지만 F1 초과 |
| S3 | 44 | 0.3902 | 0.8219 | 0.3874 | 0.8222 | -0.0003 | S1과 비슷한 수준 |

요약 통계는 다음과 같다. 표준편차는 seed 3개에 대한 표본 표준편차다.

| 지표 | 평균 | 표준편차 | 최소 | 최대 |
| --- | ---: | ---: | ---: | ---: |
| train loss | 0.3899 | 0.0003 | 0.3896 | 0.3902 |
| train accuracy | 0.8220 | 0.0007 | 0.8214 | 0.8227 |
| validation loss | 0.3887 | 0.0018 | 0.3874 | 0.3908 |
| validation accuracy | 0.8219 | 0.0016 | 0.8202 | 0.8234 |

## 4. 가설 검증

| 가설 | 결과 | 판정 |
| --- | --- | --- |
| S1 단일 seed 결과는 우연히 튄 값일 수 있다. | S2/S3에서도 validation accuracy가 각각 `0.8202`, `0.8222`로 유지됐다. | 큰 흔들림 없음 |
| 최종 후보는 F1 random init full fine-tuning보다 좋아야 한다. | 세 seed 모두 F1 `0.8191`보다 높고, 평균도 `0.8219`다. | 충족 |
| train-val gap이 커지면 과적합을 의심한다. | gap은 `-0.0020`, `0.0025`, `-0.0003`으로 작다. | 과적합 신호 작음 |
| seed 표준편차가 크면 설정이 불안정하다. | validation accuracy 표준편차는 `0.0016`으로 작다. | 안정적 |

## 5. 결론

S 시리즈 결과, R3 설정은 단일 seed 우연으로 보기 어렵다. validation accuracy 평균은 `0.8219`로 F1 random init full fine-tuning의 `0.8191`보다 높고, 세 seed 모두 F1을 넘었다.

다만 개선 폭은 평균 기준 `+0.0028`로 크지 않다. 따라서 결론은 "사전학습 D5 backbone이 적절한 full fine-tuning과 learning rate 조건에서는 F1 기준선을 안정적으로 조금 넘는다"로 둔다. 최종 test set 평가는 이 설정 1개에 대해서만 진행한다.

최종 후보 설정은 다음과 같다.

```text
pretrained checkpoint = D5
freeze mode = full
backbone lr = 3e-5
head lr = 1e-3
classifier drop rate = 0.2
weight decay = 0.01
max length = 64
seed = validation에서는 42/43/44 반복, test는 대표 seed 42 또는 best validation checkpoint
```
