# mini GPT 구현 과제 보고서

작성일: 2026-06-04  
보고 범위: BPE 토크나이저 구현부터 GPT 사전학습 실험까지

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| 반 | 302 |
| 팀명 | 7팀 |
| 팀원 | 오기란, 고윤서, 구름, 이경근 |

## 1. 요약

이 프로젝트는 외부 pretrained model이나 외부 tokenizer 라이브러리 없이, PyTorch만 사용해 작은 GPT 계열 언어 모델을 구현하고 NSMC 리뷰 문장으로 language modeling 사전학습을 수행한 결과를 정리한 것이다.

구현은 `src/bpe.py`, `src/dataset.py`, `src/embeddings.py`, `src/attention.py`, `src/model.py`, `src/train.py`를 중심으로 진행했다. 전체 테스트는 `pytest tests/ -v` 기준 `34 passed`로 통과했다.

사전학습 실험은 두 단계로 정리할 수 있다.

| 단계 | 목적 | 핵심 결론 |
| --- | --- | --- |
| 초기 탐색 | epoch, batch size, drop rate, lr, context length, layer 수, embedding 차원의 영향 확인 | 학습을 오래 하고, lr을 높이고, 모델을 `emb-dim=192`, `n-layers=4`로 키우는 방향이 유효했다. |
| A-E 시리즈 | 좋은 신호를 조합한 뒤 lr, dropout, 모델 크기, context length를 순차 탐색 | 현실적인 최종 후보는 `lr=1.5e-3`, `drop-rate=0.05`, `emb-dim=192`, `n-layers=4`, `context-length=64`이다. |

최종 후보를 고른 핵심 근거는 다음과 같다.

- C2는 8 epochs 장기 학습에서 final validation loss `5.0798`, best validation loss 약 `5.0440`을 기록해 A-E 흐름 중 가장 강한 장기 학습 근거가 되었다.
- D6 `256/4`가 4 epochs 모델 크기 비교에서 순수 validation loss는 가장 낮았지만, D5 `192/4`와 차이가 매우 작고 train-val gap이 컸다. 따라서 계산량과 안정성을 고려해 D5를 현실적인 모델 크기로 선택했다.
- E 시리즈에서 `context-length=64`가 validation loss와 train-val gap 모두 가장 안정적이었다.

## 2. 구현 현황

| 단계 | 구현 내용 | 구현 파일 | 상태 |
| --- | --- | --- | --- |
| 1 | UTF-8 byte-level BPE tokenizer | `src/bpe.py` | 완료 |
| 2 | GPTDataset, create_dataloader, InputEmbedding | `src/dataset.py`, `src/embeddings.py` | 완료 |
| 3 | MultiHeadAttention, causal mask | `src/attention.py` | 완료 |
| 4 | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, greedy generation | `src/model.py` | 완료 |
| 5 | loss 계산, checkpoint 저장/복원, generation, pretraining loop, epoch logging | `src/train.py`, `scripts/pretrain_local.py` | 완료 |

감성 분류 미세 조정 구현과 로그도 존재하지만, 이 보고서는 요청 범위에 맞춰 사전학습까지의 결과만 다룬다.

## 3. 테스트 통과 현황

실행 환경에서 전체 테스트를 실행했다.

```bash
pytest tests/ -v
```

결과: `34 passed, 1 warning in 13.97s`

경고 1개는 `matplotlib`의 non-interactive canvas 관련 경고이며, 테스트 실패는 아니다.

| 테스트 파일 | 결과 |
| --- | --- |
| `tests/test_bpe.py` | 통과 |
| `tests/test_dataset.py` | 통과 |
| `tests/test_attention.py` | 통과 |
| `tests/test_model.py` | 통과 |
| `tests/test_train.py` | 통과 |
| `tests/test_pretrain_presets.py` | 통과 |
| `tests/test_finetune.py` | 통과 |
| 전체 `pytest tests/ -v` | 통과 |

## 4. 데이터

사용 데이터는 NAVER Sentiment Movie Corpus(NSMC)이다. 원본 TSV를 읽어 빈 리뷰를 제거하고 공백을 정리한 뒤, language modeling용 텍스트와 감성 분류용 JSONL로 나누었다.

| 항목 | 내용 |
| --- | --- |
| 원본 데이터 | NSMC |
| 원본 파일 | `data/ratings_train.txt`, `data/ratings_test.txt` |
| 사전학습 train | `data/nsmc_lm_train.txt` |
| 사전학습 validation | `data/nsmc_lm_val.txt` |
| 전처리 | 빈 리뷰 제거, 연속 공백 정리, seed `42`로 train/validation 분리 |
| LM char limit | `1,500,000` |
| validation ratio | `0.08` |

현재 로컬 데이터 크기는 다음과 같다.

| 파일 | line 수 | 문자 수 |
| --- | ---: | ---: |
| `data/nsmc_lm_train.txt` | 37,825 | 1,379,486 |
| `data/nsmc_lm_val.txt` | 3,289 | 120,560 |

감성 분류용 데이터도 생성되어 있지만 사전학습 결과 판단에는 사용하지 않았다.

| 파일 | line 수 |
| --- | ---: |
| `data/nsmc_sentiment_train.jsonl` | 137,996 |
| `data/nsmc_sentiment_val.jsonl` | 11,999 |
| `data/nsmc_sentiment_test.jsonl` | 49,997 |

## 5. BPE 토크나이저

토크나이저는 한국어를 글자 단위로 먼저 자르지 않고, 문자열을 UTF-8 byte sequence로 변환한 뒤 BPE merge를 적용하는 방식으로 구현했다.

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/bpe.py` |
| 방식 | UTF-8 byte-level BPE |
| 특수 토큰 | `<pad>`, `<unk>`, `<bos>`, `<eos>` |
| 특수 토큰 ID | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token ID | byte 0~255를 ID 4~259에 배치 |
| BPE merge token ID | 260 이상 |
| vocab size | 3000 |
| 실제 vocabulary 수 | 3000 |
| merge rule 수 | 2740 |
| tokenizer 저장 경로 | `data/tokenizer.json` |
| tokenizer 학습 설정 | `tokenizer_train_chars=300_000` |

구현상 중요한 점은 `decode()`에서 token을 byte까지 재귀적으로 펼친 뒤, 마지막에 한 번만 `bytes(...).decode("utf-8")`를 수행한다는 것이다. 이렇게 해야 한국어 multi-byte 문자가 중간에 깨지지 않는다.

## 6. 모델 구조

모델은 GPT 계열 decoder-only Transformer 구조다.

```text
InputEmbedding
-> TransformerBlock x N
-> final LayerNorm
-> LM head
```

각 Transformer block은 pre-norm 구조로 구성했다.

```text
x = x + attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

| 구성 요소 | 구현 내용 |
| --- | --- |
| InputEmbedding | token embedding + position embedding + dropout |
| MultiHeadAttention | Q/K/V projection, head split/merge, scaled dot-product attention, causal mask |
| FeedForward | Linear -> GELU -> Linear -> Dropout |
| TransformerBlock | causal self-attention과 FFN을 residual connection으로 연결 |
| GPTModel | block stack, final norm, LM head, cross entropy loss |
| generation | `generate_text_simple()`에서 greedy decoding |

사전학습 최종 후보 모델 설정은 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| vocab size | 3000 |
| context length | 64 |
| embedding dimension | 192 |
| attention heads | 4 |
| layers | 4 |
| drop rate | 0.05 |
| qkv bias | False |
| 총 파라미터 수 | 2,941,824 |

## 7. 사전학습 실행 설정

사전학습은 `scripts/pretrain_local.py`로 실행했다. 실행 로그는 `runs/pretrain_logs/<run-name>/` 아래에 `config.json`, `epoch_metrics.jsonl`, `epoch_metrics.csv` 형태로 저장된다.

최종 후보 기준 실행 옵션은 다음과 같다.

```text
--train-text data/nsmc_lm_train.txt
--val-text data/nsmc_lm_val.txt
--vocab-size 3000
--context-length 64
--emb-dim 192
--n-heads 4
--n-layers 4
--batch-size 8
--epochs 4 또는 8
--lr 1.5e-3
--drop-rate 0.05
--weight-decay 0.1
--eval-freq 20
--eval-iter 5
--tokenizer-path data/tokenizer.json
--train-token-cache data/train_token_ids.pt
--val-token-cache data/val_token_ids.pt
--seed 123
```

## 8. 초기 하이퍼파라미터 탐색

초기 실험은 기준 설정을 하나씩 바꾸면서 어떤 방향이 유효한지 확인하는 목적이었다.

기준 설정은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| vocab size | 3000 |
| context length | 64 |
| emb dim | 128 |
| n heads | 4 |
| n layers | 2 |
| batch size | 8 |
| epochs | 1 |
| learning rate | 3e-4 |
| drop rate | 0.1 |
| eval iter | 5 |

기준 결과:

| final step | train loss | val loss | train-val gap | perplexity |
| ---: | ---: | ---: | ---: | ---: |
| 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |

### 9.1 Epoch 변화

| epochs | final step | train loss | val loss | gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 2 | 3144 | 6.0927 | 6.2148 | 0.1221 | 500.1 |
| 4 | 6288 | 5.5249 | 5.6948 | 0.1699 | 297.3 |
| 8 | 12576 | 5.1969 | 5.4550 | 0.2581 | 233.9 |
| 16 | 25152 | 4.9453 | 5.3150 | 0.3697 | 203.4 |

epochs를 늘릴수록 validation loss가 계속 낮아졌다. 동시에 train-val gap도 커졌으므로, 장기 학습에서는 dropout 같은 regularization을 함께 확인해야 한다.

### 9.2 Batch Size 변화

| batch size | final step | train loss | val loss | gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 786 | 7.2665 | 7.2813 | 0.0148 | 1452.9 |
| 8 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 4 | 3144 | 6.6335 | 6.6469 | 0.0134 | 770.4 |
| 2 | 6289 | 6.1651 | 6.2782 | 0.1131 | 532.8 |

batch size가 작아질수록 loss가 좋아졌지만, final step도 함께 증가했다. 따라서 이 결과는 batch size 자체의 우위라기보다 update 수 증가 효과로 해석해야 한다.

### 9.3 Drop Rate 변화

| drop rate | final step | train loss | val loss | gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 1572 | 6.8437 | 6.8795 | 0.0358 | 972.1 |
| 0.1 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 0.2 | 1572 | 7.1595 | 7.1852 | 0.0257 | 1319.8 |

짧은 1 epoch 학습에서는 dropout을 끄는 것이 가장 좋았다. 그러나 장기 학습에서는 과적합 가능성이 커지므로 A/C 시리즈에서 dropout을 다시 검증했다.

### 9.4 Learning Rate 변화

| learning rate | final step | train loss | val loss | gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 1572 | 7.2740 | 7.2927 | 0.0187 | 1469.5 |
| 3e-4 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 5e-4 | 1572 | 6.3814 | 6.4468 | 0.0654 | 630.7 |

`5e-4`가 `1e-4`, `3e-4`보다 훨씬 좋았다. 이후 B 시리즈에서 더 큰 learning rate 범위를 확인했다.

### 9.5 Context Length 변화

| context length | final step | train loss | val loss | gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 128 | 786 | 7.2772 | 7.2835 | 0.0063 | 1456.1 |
| 256 | 393 | 7.3043 | 7.3071 | 0.0028 | 1490.8 |

`context-length=64`가 가장 좋았지만, 긴 context에서는 step 수가 크게 줄었다. 따라서 "긴 context가 나쁘다"가 아니라 "현재 학습량에서는 64가 가장 효율적"이라고 해석했다.

### 9.6 Layer 수와 Embedding 차원

| n-layers | final step | train loss | val loss | gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1572 | 7.0931 | 7.1236 | 0.0305 | 1240.9 |
| 2 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 4 | 1572 | 6.8880 | 6.9419 | 0.0539 | 1034.7 |

| emb-dim | final step | train loss | val loss | gap | perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1572 | 7.2964 | 7.2916 | -0.0048 | 1467.9 |
| 128 | 1572 | 7.0360 | 7.0688 | 0.0328 | 1174.7 |
| 192 | 1572 | 6.5216 | 6.5269 | 0.0053 | 683.3 |

층 수와 embedding 차원을 키웠을 때 validation loss가 내려갔다. 현재 모델은 과적합보다 학습 부족과 표현력 부족에 더 가까운 상태로 보였고, 이후 실험에서는 `emb-dim=192`, `n-layers=4`를 유력 후보로 사용했다.

## 10. A-E 사전학습 실험

A-E 시리즈는 초기 탐색 결과를 바탕으로, 한 번에 하나의 질문만 바꾸는 방식으로 진행했다.

| 시리즈 | 질문 | 결론 |
| --- | --- | --- |
| A | 좋은 신호를 합친 설정에서 epoch와 dropout은 어떤 영향을 주는가? | 8 epochs에서는 `drop-rate=0.05`가 가장 좋았다. |
| B | `5e-4`보다 더 좋은 learning rate가 있는가? | `1.5e-3`이 가장 낮은 validation loss를 보였다. |
| C | 높은 learning rate에서 장기 학습 시 dropout이 여전히 유효한가? | `drop-rate=0.05`가 best val loss와 gap 균형이 가장 좋았다. |
| D | 같은 최적화 설정에서 어느 모델 크기가 적절한가? | D6가 val loss 1위지만, D5 `192/4`가 현실적인 best였다. |
| E | 최종 후보 모델에서 context length는 64/128/256 중 무엇이 좋은가? | `context-length=64`가 가장 안정적이었다. |

![Pretraining decision trail](docs/pre-training/assets/pretraining_decision_trail.png)

주의할 점은 C 시리즈는 8 epochs 장기 학습이고, D/E 시리즈는 빠른 비교를 위해 4 epochs로 진행했다는 것이다. 따라서 서로 다른 시리즈의 loss를 단순 순위로 비교하기보다, 각 시리즈가 어떤 선택을 정당화했는지 중심으로 해석했다.

### 10.1 A 시리즈: 조합 실험과 Dropout 확인

공통 설정:

| 항목 | 값 |
| --- | --- |
| context length | 64 |
| emb dim | 192 |
| n heads | 4 |
| n layers | 4 |
| batch size | 8 |
| learning rate | 5e-4 |
| eval freq / iter | 20 / 5 |

| 실험 | epochs | drop-rate | final step | train loss | val loss | best val loss | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 4 | 0.0 | 6324 | 4.9285 | 5.2662 | 약 5.2662 | 0.3377 |
| A2 | 8 | 0.0 | 12648 | 4.1882 | 5.2562 | 약 5.2059 | 1.0680 |
| A3 | 8 | 0.05 | 12648 | 4.5097 | 5.1477 | 약 5.1326 | 0.6380 |
| A4 | 8 | 0.1 | 12648 | 4.6929 | 5.1590 | 약 5.1350 | 0.4661 |

![A series](docs/pre-training/assets/a_series_epoch_dropout.png)

A1은 기존 4 epochs 기준 validation loss `5.6948`보다 크게 개선된 `5.2662`를 기록했다. 좋은 신호를 합치는 방향은 유효했다.

A2는 8 epochs로 늘렸지만 final validation loss 개선은 작고 train-val gap이 `1.0680`까지 커졌다. A3는 `drop-rate=0.05`를 적용해 final validation loss `5.1477`, best validation loss 약 `5.1326`을 기록했다. 따라서 장기 학습에서는 약한 dropout이 일반화에 도움이 된다고 판단했다.

### 10.2 B 시리즈: Learning Rate 탐색

공통 설정:

| 항목 | 값 |
| --- | --- |
| emb dim | 192 |
| n layers | 4 |
| drop rate | 0.0 |
| epochs | 4 |

| 실험 | lr | final step | train loss | val loss | best val loss | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 3e-4 | 6324 | 5.1312 | 5.3469 | 약 5.3446 | 0.2157 |
| B2 | 5e-4 | 6324 | 4.9285 | 5.2662 | 약 5.2508 | 0.3377 |
| B3 | 7e-4 | 6324 | 4.8473 | 5.2155 | 약 5.2085 | 0.3682 |
| B4 | 1e-3 | 6324 | 4.8937 | 5.2010 | 약 5.1947 | 0.3073 |
| B5 | 1.5e-3 | 6324 | 4.9818 | 5.1793 | 약 5.1793 | 0.1975 |

![B series](docs/pre-training/assets/b_series_learning_rate.png)

learning rate를 `3e-4 -> 1.5e-3`로 키울수록 validation loss가 계속 낮아졌다. `1.5e-3`에서도 발산이나 loss 튐이 나타나지 않았으므로, C 시리즈에서는 이 값을 고정하고 dropout을 재검증했다.

### 10.3 C 시리즈: 높은 Learning Rate에서 Dropout 장기 재검증

공통 설정:

| 항목 | 값 |
| --- | --- |
| emb dim | 192 |
| n layers | 4 |
| learning rate | 1.5e-3 |
| epochs | 8 |

| 실험 | drop-rate | final step | train loss | val loss | best val loss | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | 0.0 | 12648 | 4.4708 | 5.0812 | 약 5.0646 | 0.6104 |
| C2 | 0.05 | 12648 | 4.6668 | 5.0798 | 약 5.0440 | 0.4130 |
| C3 | 0.1 | 12648 | 4.8119 | 5.1151 | 약 5.0843 | 0.3032 |

![C series](docs/pre-training/assets/c_series_dropout.png)

C1은 dropout 없이도 B5보다 validation loss가 낮아져 `lr=1.5e-3`이 8 epochs에서도 유효함을 보여주었다. 하지만 gap이 `0.6104`로 컸다.

C2는 final validation loss `5.0798`, best validation loss 약 `5.0440`으로 C 시리즈에서 가장 좋았고, gap도 C1보다 줄었다. C3는 gap은 작았지만 validation loss가 높아져 `drop-rate=0.1`은 현재 조건에서 다소 강한 regularization으로 해석했다.

### 10.4 D 시리즈: 모델 크기 비교

공통 설정:

| 항목 | 값 |
| --- | --- |
| learning rate | 1.5e-3 |
| drop rate | 0.05 |
| epochs | 4 |
| context length | 64 |

| 실험 | emb-dim | n-layers | final step | train loss | val loss | best val loss | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | 128 | 2 | 6324 | 5.1549 | 5.3315 | 약 5.3262 | 0.1766 |
| D2 | 192 | 2 | 6324 | 4.9702 | 5.2748 | 약 5.2632 | 0.3046 |
| D3 | 256 | 2 | 6324 | 4.8166 | 5.2257 | 약 5.2257 | 0.4091 |
| D4 | 128 | 4 | 6324 | 5.0195 | 5.3227 | 약 5.3161 | 0.3032 |
| D5 | 192 | 4 | 6324 | 5.0585 | 5.2280 | 약 5.2232 | 0.1695 |
| D6 | 256 | 4 | 6324 | 4.7450 | 5.2234 | 약 5.2204 | 0.4784 |

![D series](docs/pre-training/assets/d_series_model_size.png)

2-layer 조건에서는 embedding 차원을 키울수록 validation loss가 낮아졌다. 4-layer 조건에서도 D6 `256/4`가 순수 validation loss 1위였지만, D5 `192/4`와 final validation loss 차이는 약 `0.0046`에 불과했다.

반면 D6의 train-val gap은 `0.4784`로 D5의 `0.1695`보다 훨씬 컸다. 따라서 계산량과 안정성을 고려해 D5 `emb-dim=192`, `n-layers=4`를 현실적인 모델 크기로 선택했다.

### 10.5 E 시리즈: Context Length 재검증

공통 설정:

| 항목 | 값 |
| --- | --- |
| learning rate | 1.5e-3 |
| drop rate | 0.05 |
| emb dim | 192 |
| n layers | 4 |
| epochs | 4 |

| 실험 | context length | final step | train loss | val loss | best val loss | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | 64 | 6324 | 5.0741 | 5.2254 | 약 5.2254 | 0.1513 |
| E2 | 128 | 3160 | 4.8942 | 5.3675 | 약 5.3675 | 0.4733 |
| E3 | 256 | 1580 | 4.8999 | 5.4494 | 약 5.4494 | 0.5495 |

![E series](docs/pre-training/assets/e_series_context_length.png)

E1은 D5와 거의 같은 조건의 재현 실험에 가깝고, validation loss도 `5.2254`로 안정적으로 나왔다. E2/E3는 train loss는 낮았지만 validation loss와 gap이 모두 악화되었다.

현재 코드와 4 epochs 조건에서는 `context-length=64`가 가장 효율적이고 안정적이다. 다만 context length가 길어질수록 final step이 달라지므로, 긴 context 자체가 나쁘다는 결론은 아니다.

## 11. 최종 사전학습 후보

현재까지의 실험 흐름에서 선택한 사전학습 후보는 다음과 같다.

| 항목 | 선택값 | 근거 |
| --- | --- | --- |
| learning rate | `1.5e-3` | B5가 B 시리즈 최저 validation loss |
| drop rate | `0.05` | A3/C2에서 장기 학습 일반화 균형이 가장 좋음 |
| emb dim | `192` | D5가 D6와 거의 동률이면서 gap과 계산량이 더 현실적 |
| n layers | `4` | 초기 탐색과 D 시리즈 모두에서 4-layer가 유효 |
| context length | `64` | E1이 E 시리즈 최저 validation loss와 최저 gap |
| batch size | `8` | 대부분의 A-E 실험에서 고정한 비교 기준 |
| weight decay | `0.1` | 사전학습 스크립트 기본 최적화 설정 |

최종 후보 명령 예시는 다음과 같다.

```bash
python scripts/pretrain_local.py \
  --train-text data/nsmc_lm_train.txt \
  --val-text data/nsmc_lm_val.txt \
  --vocab-size 3000 \
  --context-length 64 \
  --emb-dim 192 \
  --n-heads 4 \
  --n-layers 4 \
  --batch-size 8 \
  --epochs 8 \
  --lr 1.5e-3 \
  --drop-rate 0.05 \
  --eval-freq 20 \
  --eval-iter 5 \
  --tokenizer-path data/tokenizer.json \
  --train-token-cache data/train_token_ids.pt \
  --val-token-cache data/val_token_ids.pt
```

## 12. 실험 환경

| 항목 | 내용 |
| --- | --- |
| Python | Python 3.12.0에서 테스트 실행 |
| PyTorch | 2.12.0 |
| 실행 장치 | 로컬 macOS, MPS 로그 확인 |
| 주요 로그 경로 | `runs/pretrain_logs/` |
| 그래프 경로 | `docs/pre-training/assets/` |
| 테스트 명령 | `pytest tests/ -v` |

README의 권장 환경은 Python 3.11이지만, 현재 로컬 테스트는 Python 3.12.0에서도 통과했다.

## 13. 한계와 개선점

이번 결과에서 가장 조심해야 할 점은 step 수 통제다. batch size와 context length가 바뀌면 epoch당 update 수가 달라진다. 초기 batch size 실험과 context length 실험은 그래서 완전한 공정 비교가 아니라, 현재 코드와 학습량에서의 효율 비교로 해석해야 한다.

D/E 시리즈는 빠른 탐색을 위해 4 epochs로 진행했다. C2는 8 epochs 장기 학습 근거로 강하지만, 최종 후보인 `lr=1.5e-3`, `drop-rate=0.05`, `emb-dim=192`, `n-layers=4`, `context-length=64`를 더 확실히 제출하려면 8 epochs 이상으로 한 번 더 재현하는 것이 좋다.

향후 개선 방향은 다음과 같다.

| 개선점 | 이유 |
| --- | --- |
| `--max-steps` 옵션 추가 | batch size/context length를 같은 update 수로 공정 비교하기 위해 필요 |
| 최종 후보 8~16 epochs 재검증 | C2 수준의 validation loss가 재현되는지 확인 |
| D6 장기 재검증 | 계산 여유가 있다면 `emb-dim=256`, `n-layers=4`가 실제로 더 좋은지 확인 |
| generation sample 체계적 기록 | loss 외에 정성적 생성 품질 변화를 함께 비교하기 위해 필요 |

## 14. 결론

사전학습 실험의 전체 흐름은 "작은 기준 모델이 아직 충분히 학습하지 못하고 있다"는 관찰에서 출발했다. epoch를 늘리고, learning rate를 높이고, 모델 크기를 키웠을 때 validation loss가 낮아졌기 때문이다.

다만 오래 학습할수록 train-val gap이 커졌고, dropout 없이 장기 학습할 때 과적합 신호가 나타났다. A/C 시리즈는 `drop-rate=0.05`가 이 문제를 가장 잘 완화한다는 근거를 주었다.

현재 프로젝트의 사전학습 best 후보는 다음 설정이다.

```text
vocab_size = 3000
context_length = 64
emb_dim = 192
n_heads = 4
n_layers = 4
drop_rate = 0.05
learning_rate = 1.5e-3
batch_size = 8
```

이 설정은 성능, 일반화 gap, 계산량을 모두 고려했을 때 현재 조건에서 가장 현실적인 선택이다.
