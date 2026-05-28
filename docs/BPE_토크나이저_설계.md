``` python
    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        # raise NotImplementedError("BPETokenizer.train을 구현하세요.")
        0. _init_special_tokens()를 먼저 호출
        1. `corpus.encode("utf-8")`값에 BYTE_OFFSET을 더하는걸 반복하여 byte ID 시퀀스를 만듭니다.
        2. 시퀀스에 for zip 두 개씩 묶어서 해당 묶음의 카운트 값을 1씩 올려준다. 끝까지 다 돌고나면 그 중 카운트 값이 가장 큰 페어를 best pair로 선정한다.
        new_id = len(self.id_to_token)
        3. self.token_to_id 갱신:  best pair(tuple)를 key값에 새 token ID를 value 값에 넣어줍니다.
        4. self.id_to_token 갱신: 새 token ID를 key값에 best pair(tuple)를 새 ID의 value값에 넣어줍니다.
        5. best pair를 이용하여 새로운 시퀀스로 갱신: 왼쪽부터 보면서 현재 위치와 다음 위치가 best pair와 같은 pair이면 new_ids 리스트에 new_id를 append하고 두 칸 이동합니다. 아니면 현재 token을 그대로 넣고 한 칸 이동합니다. 기존 시퀀스를 new_ids 리스트로 갱신합니다.
        6. self.merges 갱신: merges에 best pair(tuple)를 append
        7. vocab_size에 도달하거나, 더 이상 pair가 없을 때까지 2~6번을 반복. 한 번 merge하면 sequence가 바뀌므로, 그 다음 pair 빈도는 다시 세야 합니다.

        방어 용도 구문
        if best_pair in self.token_to_id:
            break
        보통 현재 sequence에서 새로 고른 pair는 아직 merge되지 않은 pair일 가능성이 높지만, 구현 실수나 edge case 방어용으로 생각할 수 있어.
    
    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        1. 8바이트 리스트를 만듭니다. text.encode("utf-8")?
        2. merge는 추후 구현
        3. add_bos_eos=True이면 list의 앞뒤에 bos/eos ID를 붙입니다.
        4. list를 반환
        raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
```