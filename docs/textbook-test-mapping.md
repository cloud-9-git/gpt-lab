# 교재-테스트 매핑 및 팀 스터디 가이드

이 문서는 팀이 『밑바닥부터 만들면서 배우는 LLM』을 읽으면서 `gpt-lab` 과제를 단계별로 구현하기 위한 가이드입니다. 교재의 장/절 소제목과 짧은 연결 설명만 정리하며, 긴 발췌나 페이지 재현은 넣지 않습니다.

## 사용 방법

1. 각 Step의 **교재 범위**를 먼저 읽습니다.
2. 스터디 시간에 **팀 토론 질문**을 함께 확인합니다.
3. **구현 목표**를 기준으로 관련 `src/` 파일을 구현합니다.
4. **함수 단위 테스트**로 작은 동작부터 확인합니다.
5. Step의 **파일 단위 테스트**가 통과하면 다음 Step으로 넘어갑니다.
6. 모든 Step이 끝나면 전체 테스트를 실행합니다.

```bash
pytest tests/ -v
```

## 전체 흐름

| Step | 구현 주제 | 교재 범위 | 완료 테스트 |
| --- | --- | --- | --- |
| 1 | BPE Tokenizer | 2.3, 2.4, 2.5 | `pytest tests/test_bpe.py -v` |
| 2 | Dataset / Embedding | 2.6, 2.7, 2.8 | `pytest tests/test_dataset.py -v` |
| 3 | Causal Multi-Head Attention | 3.5, 3.6 | `pytest tests/test_attention.py -v` |
| 4 | GPT Model | 4.2, 4.3, 4.4, 4.5, 4.6, 4.7 | `pytest tests/test_model.py -v` |
| 5 | Pretraining Utilities | 5.1, 5.2, 5.3, 5.4 | `pytest tests/test_train.py -v` |
| 6 | Sentiment Fine-tuning | 6.2, 6.3, 6.5, 6.6, 6.7 | `pytest tests/test_finetune.py -v` |

## Step 1. BPE Tokenizer

### 교재 범위

- 2.3 토큰을 토큰 ID로 변환하기
- 2.4 특수 문맥 토큰 추가하기
- 2.5 바이트 페어 인코딩

### 공부 목표

- 텍스트가 토큰, 토큰 ID, 다시 텍스트로 바뀌는 흐름을 설명할 수 있다.
- 특수 토큰이 모델 입력과 padding에서 어떤 역할을 하는지 설명할 수 있다.
- BPE가 처음 보는 단어를 더 작은 단위로 처리하는 이유를 설명할 수 있다.
- 한국어에서는 문자 단위보다 UTF-8 byte 기반 처리가 안전한 이유를 이해한다.

### 구현 목표

- `<pad>`, `<unk>`, `<bos>`, `<eos>`를 고정 ID로 등록한다.
- byte 0~255를 ID 4~259 범위에 등록한다.
- corpus에서 자주 등장하는 이웃 pair를 merge rule로 학습한다.
- 학습한 vocab과 merge rule을 JSON으로 저장하고 다시 불러온다.
- `encode()`와 `decode()`가 한국어/영어 혼합 텍스트를 깨뜨리지 않고 복원한다.

### 관련 파일

- `src/bpe.py`
- `tests/test_bpe.py`

### 함수 단위 테스트

```bash
pytest tests/test_bpe.py::TestSpecialTokens::test_special_ids_fixed -v
pytest tests/test_bpe.py::TestBPETokenizer::test_init_special_tokens -v
pytest tests/test_bpe.py::TestBPETokenizer::test_save_load_restores_vocab -v
pytest tests/test_bpe.py::TestBPETokenizer::test_encode_decode_restores_original_text -v
pytest tests/test_bpe.py::TestBPETokenizer::test_get_special_ids -v
pytest tests/test_bpe.py::TestBPETrain::test_train_increases_vocab -v
```

### 파일 단위 테스트

```bash
pytest tests/test_bpe.py -v
```

### 완료 기준

- `tests/test_bpe.py` 전체가 통과한다.
- `decode(encode(text, add_bos_eos=True), skip_special=True)`가 원래 문자열을 복원한다.

### 팀 토론 질문

- 왜 이 과제에서는 외부 tokenizer를 쓰지 않고 직접 BPE를 구현할까?
- byte token ID를 4부터 시작하게 만든 이유는 무엇일까?
- merge token을 decode할 때 재귀적으로 byte까지 펼쳐야 하는 이유는 무엇일까?
- `skip_special=True`일 때 `<bos>`와 `<eos>`는 어떻게 처리해야 할까?

### 주의점

- 교재는 BPE 부분에서 외부 tokenizer 사용법을 보여 주지만, 과제는 직접 구현해야 한다.
- 한국어 byte를 하나씩 decode하면 깨질 수 있다. byte 리스트를 모두 모은 뒤 마지막에 한 번 decode해야 한다.
- JSON은 `bytes`나 `tuple`을 그대로 저장하지 못하므로 저장 형식을 명확히 정해야 한다.

## Step 2. Dataset / Embedding

### 교재 범위

- 2.6 슬라이딩 윈도로 데이터 샘플링하기
- 2.7 토큰 임베딩 만들기
- 2.8 단어 위치 인코딩하기

### 공부 목표

- 다음 토큰 예측에서 input과 target이 한 칸 shift되는 이유를 설명할 수 있다.
- `context_length`와 `stride`가 학습 샘플 수에 어떤 영향을 주는지 이해한다.
- token embedding과 position embedding을 더하는 이유를 설명할 수 있다.
- embedding 출력 shape `(batch, seq_len, emb_dim)`을 읽을 수 있다.

### 구현 목표

- token ID 리스트를 `context_length` 길이의 input/target 쌍으로 자른다.
- `stride` 간격으로 sliding window 샘플을 만든다.
- `GPTDataset`을 `DataLoader`로 감싼다.
- token embedding, position embedding, dropout을 적용하는 `InputEmbedding`을 만든다.

### 관련 파일

- `src/dataset.py`
- `src/embeddings.py`
- `tests/test_dataset.py`

### 함수 단위 테스트

```bash
pytest tests/test_dataset.py::TestGPTDataset::test_dataset_length -v
pytest tests/test_dataset.py::TestGPTDataset::test_dataset_getitem_shape -v
pytest tests/test_dataset.py::TestCreateDataloader::test_dataloader_batch_shape -v
pytest tests/test_dataset.py::TestInputEmbedding::test_input_embedding_shape -v
```

### 파일 단위 테스트

```bash
pytest tests/test_dataset.py -v
```

### 완료 기준

- `tests/test_dataset.py` 전체가 통과한다.
- target tensor가 input tensor보다 한 token 앞선 값을 가진다.
- embedding 출력 shape가 `(batch_size, seq_len, emb_dim)`이다.

### 팀 토론 질문

- input 길이가 `context_length`이면 target을 만들기 위해 왜 token이 하나 더 필요할까?
- `stride=context_length`와 `stride=1`은 데이터 중복과 학습에 어떤 차이를 만들까?
- Transformer에 position embedding이 없으면 어떤 정보를 잃을까?
- dropout은 학습과 평가에서 왜 다르게 동작해야 할까?

### 주의점

- 샘플 개수 계산에서 마지막 target token 1개를 고려해야 한다.
- position ID는 입력 tensor와 같은 device에 있어야 GPU 학습에서 문제가 없다.
- `shuffle=False` 테스트에서는 배치 순서가 예측 가능해야 한다.

## Step 3. Causal Multi-Head Attention

### 교재 범위

- 3.5 코잘 어텐션으로 미래의 단어를 감추기
- 3.6 싱글 헤드 어텐션을 멀티 헤드 어텐션으로 확장하기

### 공부 목표

- causal mask가 autoregressive generation에서 왜 필요한지 설명할 수 있다.
- Q/K/V projection의 역할을 구분할 수 있다.
- `(B, T, C)` 텐서를 `(B, heads, T, head_dim)`으로 바꾸는 이유를 설명할 수 있다.
- attention weight shape `(B, heads, T, T)`의 의미를 읽을 수 있다.

### 구현 목표

- Q, K, V projection과 output projection을 만든다.
- `d_model`을 `n_heads`개의 head로 나눈다.
- attention score를 `sqrt(head_dim)`으로 scale한다.
- 미래 위치를 볼 수 없도록 upper-triangular mask를 적용한다.
- 요청 시 attention weight도 함께 반환한다.

### 관련 파일

- `src/attention.py`
- `tests/test_attention.py`

### 함수 단위 테스트

```bash
pytest tests/test_attention.py::TestMultiHeadAttention::test_mha_output_shape -v
pytest tests/test_attention.py::TestMultiHeadAttention::test_mha_causal_mask_future_zero -v
```

### 파일 단위 테스트

```bash
pytest tests/test_attention.py -v
```

### 완료 기준

- `tests/test_attention.py` 전체가 통과한다.
- output shape가 input shape와 같은 `(batch, seq_len, d_model)`이다.
- causal mask 적용 시 attention weight의 미래 위치 값이 0이다.

### 팀 토론 질문

- mask를 softmax 전 score에 적용해야 하는 이유는 무엇일까?
- `d_model % n_heads == 0` 조건이 필요한 이유는 무엇일까?
- `transpose` 이후 `contiguous()`가 필요한 상황은 언제일까?
- causal mask가 없다면 다음 토큰 예측 학습에서 어떤 정보 누수가 생길까?

### 주의점

- mask shape는 batch와 head 차원에 broadcast될 수 있어야 한다.
- dropout은 attention weight에 적용하는 것이 일반적이다.
- 테스트는 `return_attention_weights=True` 동작까지 확인한다.

## Step 4. GPT Model

### 교재 범위

- 4.2 층 정규화로 활성화 정규화하기
- 4.3 GELU 활성화 함수를 사용하는 피드 포워드 네트워크 구현하기
- 4.4 숏컷 연결 추가하기
- 4.5 어텐션과 선형 층을 트랜스포머 블록에 연결하기
- 4.6 GPT 모델 만들기
- 4.7 텍스트 생성하기

### 공부 목표

- LayerNorm이 마지막 차원 기준으로 평균과 분산을 맞추는 이유를 설명할 수 있다.
- FeedForward가 임베딩 차원을 확장했다가 다시 줄이는 구조를 이해한다.
- residual connection이 깊은 모델 학습에 왜 필요한지 설명할 수 있다.
- GPT 전체 구조가 embedding, transformer blocks, final norm, LM head로 이어지는 흐름을 설명할 수 있다.

### 구현 목표

- `LayerNorm`, `GELU`, `FeedForward`를 구현한다.
- Pre-LayerNorm 방식의 `TransformerBlock`을 구현한다.
- `GPTModel`이 logits를 만들고, target이 있으면 cross entropy loss도 반환한다.
- greedy 방식의 `generate_text_simple()`을 구현한다.

### 관련 파일

- `src/model.py`
- `src/attention.py`
- `src/embeddings.py`
- `tests/test_model.py`

### 함수 단위 테스트

```bash
pytest tests/test_model.py::TestLayerNorm::test_layernorm_shape -v
pytest tests/test_model.py::TestGELU::test_gelu_shape -v
pytest tests/test_model.py::TestFeedForward::test_feedforward_shape -v
pytest tests/test_model.py::TestTransformerBlock::test_transformer_block_shape -v
pytest tests/test_model.py::TestGPTModel::test_gpt_forward_shape -v
pytest tests/test_model.py::TestGPTModel::test_gpt_forward_with_targets_returns_loss -v
pytest tests/test_model.py::TestGenerateTextSimple::test_generate_text_simple_shape -v
```

### 파일 단위 테스트

```bash
pytest tests/test_model.py -v
```

### 완료 기준

- `tests/test_model.py` 전체가 통과한다.
- `GPTModel(idx)`는 `(batch, seq_len, vocab_size)` logits를 반환한다.
- `GPTModel(idx, targets)`는 scalar loss와 logits를 반환한다.
- generation 결과 길이가 `start_len + max_new_tokens`가 된다.

### 팀 토론 질문

- LayerNorm에서 batch 차원이 아니라 마지막 차원을 정규화하는 이유는 무엇일까?
- GELU가 ReLU보다 LLM에서 자주 쓰이는 이유는 무엇일까?
- attention과 feedforward 각각에 residual connection을 두는 이유는 무엇일까?
- LM head의 출력 차원이 왜 vocab size와 같아야 할까?

### 주의점

- 교재의 `GPTModel.forward()`는 주로 logits 반환을 설명하지만, 과제는 target이 있을 때 loss 반환까지 요구한다.
- cross entropy 계산은 logits와 targets를 flatten해서 token 단위 예측으로 계산해야 한다.
- generation에서는 현재 문맥이 `context_size`보다 길면 마지막 token들만 사용해야 한다.

## Step 5. Pretraining Utilities

### 교재 범위

- 5.1 텍스트 생성 모델 평가하기
- 5.2 LLM 훈련하기
- 5.3 무작위성을 제어하기 위한 디코딩 전략
- 5.4 파이토치로 모델 로드하고 저장하기

### 공부 목표

- 다음 토큰 예측 loss가 cross entropy로 계산되는 흐름을 설명할 수 있다.
- train loss와 validation loss를 나누어 보는 이유를 이해한다.
- temperature와 top-k sampling이 generation 결과에 주는 영향을 설명할 수 있다.
- model state와 optimizer state를 함께 저장해야 하는 이유를 이해한다.

### 구현 목표

- 한 batch와 loader 전체의 평균 loss를 계산한다.
- checkpoint에 model, optimizer, epoch, global step을 저장하고 복원한다.
- temperature와 top-k를 지원하는 `generate()`를 구현한다.
- 사전 학습 loop에서 학습, 평가, 샘플 생성을 수행한다.

### 관련 파일

- `src/train.py`
- `src/model.py`
- `tests/test_train.py`

### 함수 단위 테스트

```bash
pytest tests/test_train.py::TestCalcLossBatch::test_calc_loss_batch_returns_scalar -v
pytest tests/test_train.py::TestCalcLossLoader::test_calc_loss_loader_returns_float -v
pytest tests/test_train.py::TestCheckpoint::test_save_load_checkpoint_restores_epoch_and_step -v
pytest tests/test_train.py::TestGenerate::test_generate_shape -v
pytest tests/test_train.py::TestPlotLosses::test_plot_losses_callable -v
```

### 파일 단위 테스트

```bash
pytest tests/test_train.py -v
```

### 완료 기준

- `tests/test_train.py` 전체가 통과한다.
- loss 함수가 scalar를 반환한다.
- checkpoint load 후 epoch와 global step이 저장 전 값으로 복원된다.
- `generate()` 결과 길이가 요청한 token 수만큼 늘어난다.

### 팀 토론 질문

- logits shape와 target shape를 cross entropy에 맞추려면 왜 flatten이 필요할까?
- validation에서는 왜 `torch.no_grad()`와 `model.eval()`이 필요할까?
- temperature가 0 또는 낮은 값일 때 generation은 어떻게 달라질까?
- top-k를 적용하면 왜 낮은 확률 token이 제거될까?

### 주의점

- `calc_loss_loader()`는 비어 있는 loader를 만났을 때 안전하게 처리해야 한다.
- checkpoint는 optimizer 없이도 load할 수 있어야 한다.
- `eos_id`가 생성되면 생성을 중단하는 동작을 고려해야 한다.
- 학습 loop는 테스트보다 실제 노트북 실행에서 더 중요하게 검증된다.

## Step 6. Sentiment Fine-tuning

### 교재 범위

- 6.2 데이터셋 준비
- 6.3 데이터 로더 만들기
- 6.5 분류 헤드 추가하기
- 6.6 분류 손실과 정확도 계산하기
- 6.7 지도 학습 데이터로 모델 미세 튜닝하기

### 공부 목표

- language modeling과 sequence classification의 target 차이를 설명할 수 있다.
- 텍스트 분류에서 padding과 truncation이 필요한 이유를 이해한다.
- GPT backbone 위에 classifier head를 붙이는 구조를 설명할 수 있다.
- 마지막 token hidden state를 문장 대표 벡터로 쓰는 이유를 이해한다.

### 구현 목표

- NSMC TSV를 읽어 빈 리뷰를 제거하고 train/validation/test 데이터를 만든다.
- 리뷰를 token ID로 바꾼 뒤 `max_length`에 맞게 자르거나 padding한다.
- GPT backbone 위에 dropout과 linear classifier를 붙인다.
- classification loss와 accuracy를 계산하는 train/eval 함수를 만든다.

### 관련 파일

- `src/finetune.py`
- `src/model.py`
- `tests/test_finetune.py`

### 함수 단위 테스트

```bash
pytest tests/test_finetune.py::TestMakeSentimentDataset::test_make_sentiment_dataset_splits_rows -v
pytest tests/test_finetune.py::TestReviewSentimentDataset::test_review_sentiment_dataset_getitem -v
pytest tests/test_finetune.py::TestGPTForSequenceClassification::test_sequence_classification_shape -v
pytest tests/test_finetune.py::TestSentimentTrainEval::test_train_eval_functions_exist -v
```

### 파일 단위 테스트

```bash
pytest tests/test_finetune.py -v
```

### 완료 기준

- `tests/test_finetune.py` 전체가 통과한다.
- dataset item은 `(max_length,)` shape의 LongTensor와 label을 반환한다.
- classifier logits shape는 `(batch_size, num_labels)`이다.

### 팀 토론 질문

- 교재의 스팸 분류 예제를 NSMC 감성 분류로 바꿀 때 무엇이 달라질까?
- classifier가 vocab size가 아니라 label 개수만큼 출력해야 하는 이유는 무엇일까?
- 마지막 token 표현을 쓰는 방식과 mean pooling 방식은 어떤 차이가 있을까?
- backbone 전체를 학습할지 일부만 freeze할지는 어떤 기준으로 정하면 좋을까?

### 주의점

- 교재는 스팸/스팸 아님 분류를 다루지만 과제는 긍정/부정 감성 분류를 다룬다.
- 테스트는 실제 BPE 학습 없이 dummy tokenizer로 dataset 동작도 확인한다.
- `GPTModel`이 hidden state를 직접 반환하지 않는다면 classifier 구현에서 backbone 내부 모듈을 재사용하는 방식이 필요하다.
- label dtype은 cross entropy에 맞게 `torch.long`이어야 한다.

## 단계별 실행 명령 모음

```bash
pytest tests/test_bpe.py -v
pytest tests/test_dataset.py -v
pytest tests/test_attention.py -v
pytest tests/test_model.py -v
pytest tests/test_train.py -v
pytest tests/test_finetune.py -v
pytest tests/ -v
```

## 팀 운영 체크리스트

- 각 Step 시작 전에 담당자가 교재 범위의 핵심 개념을 5분 안에 설명한다.
- 구현자는 테스트 함수 하나를 먼저 고르고, 그 테스트가 요구하는 shape와 반환값을 말로 설명한다.
- 리뷰어는 구현 코드를 보기 전에 테스트 명령을 먼저 실행한다.
- PR 설명에는 통과한 테스트 명령과 남은 실패 테스트를 적는다.
- Step이 끝날 때 `REPORT.md`의 구현 현황과 테스트 통과 현황을 갱신한다.

## 교재와 과제의 차이

- 교재의 BPE 설명은 외부 tokenizer 사용 중심이지만, 과제는 UTF-8 byte-level BPE를 직접 구현한다.
- 교재의 분류 예제는 스팸 분류이고, 과제의 미세 조정 목표는 NSMC 영화 리뷰 긍정/부정 분류다.
- 교재의 GPT forward 예제는 logits 반환 중심이고, 과제는 target이 있을 때 loss 반환까지 요구한다.
- 교재의 대형 설정은 학습 비용이 크므로, 과제 테스트는 작은 config와 synthetic input으로 동작을 검증한다.

## 문서 관리 규칙

- 교재의 장/절 번호와 소제목, 짧은 요약만 기록한다.
- 긴 발췌, 페이지 이미지, 전체 텍스트 경로는 기록하지 않는다.
- 테스트 이름이 바뀌면 먼저 `pytest --collect-only tests -q`로 현재 이름을 확인한 뒤 이 문서를 갱신한다.
- 데이터 파일, checkpoint, vocabulary 파일은 문서에 경로 예시만 적고 Git에는 올리지 않는다.
