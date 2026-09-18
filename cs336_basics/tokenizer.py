from .bpe import pretokenize
import re
from collections.abc import Iterable

class Tokenizer:

    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab=vocab
        self.merges=merges

        self.bytes_to_token_id={}
        for item in vocab.items():
            self.bytes_to_token_id[item[1]]=item[0]

        self.merge_ranks={}
        for i,merge in enumerate(merges):
            self.merge_ranks[merge]=i

        if not special_tokens:
            self.special_tokens=[]
        if special_tokens:
            self.special_tokens=special_tokens
            for special_token in self.special_tokens:
                special_token_byte=special_token.encode('utf-8')
                if special_token_byte not in self.bytes_to_token_id:
                    self.bytes_to_token_id[special_token_byte]=len(vocab)
                    self.vocab[len(vocab)]=special_token_byte


    def decode(self,ids:list[int])->str:
        bb=b''
        for i in ids:
            bb+=self.vocab[i]

        return bb.decode('utf-8',errors='replace')


    def encode(self,text:str)->list[int]:
        # 处理special tokens
        if self.special_tokens:
            self.special_tokens=sorted(self.special_tokens,key=len,reverse=True)
            escaped_special_tokens=[]
            for special_token in self.special_tokens:
                escaped_special_tokens.append(re.escape(special_token))
            pattern="|".join(escaped_special_tokens)
            chunks=re.split(f"({pattern})",text)

        else:
            chunks=[text]

        final_token_ids=[]
        # 预分词
        for chunk in chunks:
            if chunk in self.special_tokens:
                bytes_token=chunk.encode("utf-8")
                final_token_ids.append(self.bytes_to_token_id[bytes_token])
            else:
                pretokens=pretokenize(chunk)
                for pretoken in pretokens:
                    byte_tokens=tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))

                    # Continue with BPE merging logic here
                    while True:
                        rank=float('inf')
                        best_pair=None

                        for pair in zip(byte_tokens,byte_tokens[1:]):
                            if pair in self.merge_ranks:
                                pair_rank=self.merge_ranks[pair]
                                if pair_rank<rank:
                                    rank=pair_rank
                                    best_pair=pair

                        if best_pair is None:
                            break

                        else:
                            # Merge the best pair
                            merged_token=b''.join(best_pair)
                            byte_tokens=list(byte_tokens)
                            i=0
                            new_byte_tokens=[]
                            while i<len(byte_tokens)-1:
                                if byte_tokens[i]==best_pair[0] and byte_tokens[i+1]==best_pair[1]:
                                    new_byte_tokens.append(merged_token)
                                    i+=2
                                else:
                                    new_byte_tokens.append(byte_tokens[i])
                                    i+=1
                            if i==len(byte_tokens)-1:
                                new_byte_tokens.append(byte_tokens[i])

                        byte_tokens=new_byte_tokens

                    for byte_token in byte_tokens:
                        byte_token_id=self.bytes_to_token_id[byte_token]
                        final_token_ids.append(byte_token_id)

        return final_token_ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        # 只排序局部变量，不在读取每段文本时反复修改实例属性。
        special_tokens = sorted(
            self.special_tokens or [],
            key=len,
            reverse=True,
        )

        if "" in special_tokens:
            raise ValueError("special_tokens 不能包含空字符串")

        pattern = "|".join(re.escape(s) for s in special_tokens)
        max_special_len = max(
            (len(s) for s in special_tokens),
            default=0,
        )

        buffer = ""

        for text in iterable:
            buffer += text

            # 1. 在整个 buffer 上查找需要暂存的 special 前缀。
            safe_end = len(buffer)
            max_prefix_len = min(
                len(buffer),
                max(0, max_special_len - 1),
            )

            for length in range(max_prefix_len, 0, -1):
                suffix = buffer[-length:]

                if any(
                    len(special) > length and special.startswith(suffix)
                    for special in special_tokens
                ):
                    safe_end = len(buffer) - length
                    break

            # 2. 暂存边界不能切断已经匹配到的完整 special。
            if special_tokens and 0 < safe_end < len(buffer):
                for match in re.finditer(pattern, buffer):
                    if match.start() < safe_end < match.end():
                        safe_end = match.start()
                        break

            safe_text = buffer[:safe_end]
            special_tail = buffer[safe_end:]

            # 3. 只在 special 边界已经安全的区域内分割。
            if special_tokens:
                chunks = re.split(f"({pattern})", safe_text)
            else:
                chunks = [safe_text]

            # 除最后一段普通文本外，其余部分都可以处理。
            for chunk in chunks[:-1]:
                yield from self.encode(chunk)

            # 4. 最后一段普通文本，继续保留最后两个 pretokens。
            last_chunk = chunks[-1]
            pretokens = pretokenize(last_chunk)

            for pretoken in pretokens[:-2]:
                yield from self.encode(pretoken)

            buffer = "".join(pretokens[-2:]) + special_tail

        # 输入真正结束后，剩余内容不再等待。
        yield from self.encode(buffer)
