import regex as re
from collections import Counter,defaultdict
import os
import json
import multiprocessing

Token=tuple[bytes,...]
Pair=tuple[bytes,bytes]

#处理special token的核心原则是：它是不可拆分的边界，不参与BPE pair统计，也不能在它两侧形成pair。
def remove_special(text:str,special_tokens:list[str])->list[str]:
    if special_tokens:
        special_pattern="|".join(re.escape(token)
                                 for token in sorted(special_tokens,key=len,reverse=True,)
                                 )
        chunks=re.split(special_pattern,text,)
    else:
        chunks=[text]
    return chunks

#得到pretoken
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pretokenize(chunk:str)->list[str]:
    return [match.group(0) for match in re.finditer(PAT,chunk)]

#pretoken计数
def count_pretokens(text:str,special_tokens:list[str])->dict[Token,int]:
    chunks=remove_special(text,special_tokens)
    counts:Counter[Token]=Counter()
    for chunk in chunks:
        pretokens=pretokenize(chunk)
        for pretoken in pretokens:
            byte_tokens=tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))
            counts[byte_tokens]+=1
    return dict(counts)


def build_pair_data(sequences:list[Token],frequencies:list[int]
                    )->dict[Counter[Pair,dict[Pair,set[int]]]]:
    pair_counts:Counter[Pair]=Counter()
    pair_to_token_ids:defaultdict[Pair,set[int]]=defaultdict(set)

    for token_id,sequence in enumerate(sequences):
        frequency=frequencies[token_id]

        for pair in zip(sequence,sequence[1:]):
            pair_counts[pair]+=frequency
            pair_to_token_ids[pair].add(token_id)

    return pair_counts,pair_to_token_ids

#对单个pretoken进行合并
def merge_pair(token:Token,pair:Pair)->Token:
    result:list[bytes]=[]
    index=0
    while index<len(token):
        if index+1<len(token) and token[index]==pair[0] and token[index+1]==pair[1]:
            result.append(token[index]+token[index+1])
            index+=2
        else:
            result.append(token[index])
            index+=1
    return tuple(result)


def train_bpe(input_path,vocab_size:int,special_tokens:list[str]
              )->tuple[dict[int,bytes],list[Pair]]:
    
    with open(input_path,"r",encoding="utf-8") as file:
        text=file.read()

    pretoken_counts=count_pretokens(text,special_tokens)
    sequences:list[Token]=list(pretoken_counts.keys())
    frequencies:list[int]=list(pretoken_counts.values())

    vocab:dict[int,bytes]={index:bytes([index]) for index in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)]=special_token.encode("utf-8")
    merges:list[Pair]=[]

    pair_counts,pair_to_token_ids=build_pair_data(sequences,frequencies)
    while len(vocab)<vocab_size:
        if not pair_counts:
            break
        best_pair=max(pair_counts,key=lambda pair:(pair_counts[pair],pair))

        affected_ids=list(pair_to_token_ids.get(best_pair,set()))
        touched_pairs:set[Pair]=set()

        for token_id in affected_ids:
            frequency=frequencies[token_id]
            sequence=sequences[token_id]

            old_pairs=list(zip(sequence,sequence[1:]))

            for old_pair in old_pairs:
                pair_counts[old_pair]-=frequency
                touched_pairs.add(old_pair)

            for old_pair in set(old_pairs):
                token_ids=pair_to_token_ids.get(old_pair)
                if token_ids is not None:
                    token_ids.discard(token_id)
                    if not token_ids:
                        del pair_to_token_ids[old_pair]

            new_sequence=merge_pair(sequence,best_pair)
            sequences[token_id]=new_sequence

            new_pairs=list(zip(new_sequence,new_sequence[1:]))

            for new_pair in new_pairs:
                pair_counts[new_pair]+=frequency
                touched_pairs.add(new_pair)

            for new_pair in set(new_pairs):
                pair_to_token_ids.setdefault(new_pair,set()).add(token_id)

        for pair in touched_pairs:
            if pair_counts[pair]==0:
                del pair_counts[pair]
            elif pair_counts[pair]<0:
                raise RuntimeError(f"pair count became negative:{pair}")
        
        new_token=best_pair[0]+best_pair[1]
        vocab[len(vocab)]=new_token
        merges.append(best_pair)

    return vocab,merges

'''
#初版，speed不通过

#处理special token的核心原则是：它是不可拆分的边界，不参与BPE pair统计，也不能在它两侧形成pair。
def remove_special(text:str,special_tokens:list[str])->list[str]:
    if special_tokens:
        special_pattern="|".join(re.escape(token)
                                 for token in sorted(special_tokens,key=len,reverse=True,)
                                 )
        chunks=re.split(special_pattern,text,)
    else:
        chunks=[text]
    return chunks

#得到pretoken
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pretokenize(chunk:str)->list[str]:
    return [match.group(0) for match in re.finditer(PAT,chunk)]

#pretoken计数
def count_pretokens(text:str,special_tokens:list[str])->dict[tuple[bytes,...],int]:
    chunks=remove_special(text,special_tokens)
    counts:Counter[tuple[bytes,...]]=Counter()
    for chunk in chunks:
        pretokens=pretokenize(chunk)
        for pretoken in pretokens:
            byte_tokens=tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))
            counts[byte_tokens]+=1
    return dict(counts)

#计算pair数量
def count_pairs(pretoken_counts:dict[tuple[bytes,...],int])->dict[tuple[bytes,bytes],int]:
    counts:Counter[tuple[bytes,bytes]]=Counter()
    for token,count in pretoken_counts.items():
        for left,right in zip(token,token[1:]):
            counts[(left,right)]+=count
    return dict(counts)

#对单个pretoken进行合并
def merge_pair(token:tuple[bytes,...],pair:tuple[bytes,bytes])->tuple[bytes,...]:
    result:list[bytes]=[]
    index=0
    while index<len(token):
        if index+1<len(token) and token[index]==pair[0] and token[index+1]==pair[1]:
            result.append(token[index]+token[index+1])
            index+=2
        else:
            result.append(token[index])
            index+=1
    return tuple(result)

#合并后的新计数
def apply_merge(pretoken_counts:dict[tuple[bytes,...],int],pair:tuple[bytes,bytes]
                )->dict[tuple[bytes,...],int]:
    new_counts:Counter[tuple[bytes,...]]=Counter()
    for token,count in pretoken_counts.items():
        merged_token=merge_pair(token,pair)
        new_counts[merged_token]+=count
    return new_counts

def train_bpe(input_path,vocab_size:int,special_tokens:list[str]
              )->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    
    with open(input_path,"r",encoding="utf-8") as file:
        text=file.read()

    pretoken_counts=count_pretokens(text,special_tokens)
    vocab:dict[int,bytes]={index:bytes([index]) for index in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)]=special_token.encode("utf-8")
    merges:list[tuple[bytes,bytes]]=[]

    while len(vocab)<vocab_size:
        pair_counts=count_pairs(pretoken_counts)
        if not pair_counts:
            break
        best_pair=max(pair_counts,key=lambda pair:(pair_counts[pair],pair))
        new_token=best_pair[0]+best_pair[1]
        vocab[len(vocab)]=new_token
        merges.append(best_pair)
        pretoken_counts=apply_merge(pretoken_counts,best_pair)

    return vocab,merges

'''