import regex as re
from collections import Counter,defaultdict
import os
import json
import multiprocessing
import time
import heapq
from dataclasses import dataclass

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

def count_chunk(input_path,start:int,end:int,special_tokens:list[str]):
    with open(input_path,"rb") as file:
        #文件指针移动到start
        file.seek(start)
        chunk_bytes=file.read(end-start)
    chunk_text=chunk_bytes.decode("utf-8")
    return count_pretokens(chunk_text,special_tokens)

def find_chunk_boundaries(input_path,num_processes:int,special_token:bytes
                          )->list[int]:
    
    with open(input_path,"rb") as file:
        file.seek(0,os.SEEK_END) #seek(offset,whence)
        file_size=file.tell() #返回指针位置
        file.seek(0)

        #理论边界
        boundaries=[
            file_size*i//num_processes
            for i in range(num_processes+1)
        ]

        mini_chunk_size=4096

        for bi in range(1,len(boundaries)-1):
            initial_position=boundaries[bi]
            file.seek(initial_position)

            while True:
                mini_chunk=file.read(mini_chunk_size)

                if mini_chunk==b"":
                    boundaries[bi]=file_size
                    break

                found_at=mini_chunk.find(special_token)
                if found_at!=-1:
                    boundaries[bi]=initial_position+found_at
                    break
                initial_position+=mini_chunk_size

    return sorted(set(boundaries))


def parallel_count_pretokens(input_path,boundaries:list[int],special_tokens,num_processes
                             )->Counter[Token]:
    #对 Counter 来说，update() 的含义是加上计数，不是普通 dict.update() 那种直接覆盖。
    tasks=[
        (input_path,start,end,special_tokens)
        for start,end in zip(boundaries[:-1],boundaries[1:])
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results=pool.starmap(count_chunk,tasks)

    total_counts=Counter()
    for counts in results:
        total_counts.update(counts)

    return total_counts
         

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


@dataclass(slots=True)
class HeapEntry:
    pair:Pair
    count_snapshot:int
    #自定义less than
    def __lt__(self,other):
        return (self.count_snapshot,self.pair)>(other.count_snapshot,other.pair)

def train_bpe(input_path,vocab_size:int,special_tokens:list[str]
              )->tuple[dict[int,bytes],list[Pair]]:

    num_processes=16

    pretokenization_start=time.perf_counter()

    boundaries=find_chunk_boundaries(input_path,num_processes,b"<|endoftext|>")
    pretoken_counts=parallel_count_pretokens(input_path,boundaries,special_tokens,num_processes)

    pretokenization_end=time.perf_counter()

    BPE_build_merge_start=time.perf_counter()

    sequences:list[Token]=list(pretoken_counts.keys())
    frequencies:list[int]=list(pretoken_counts.values())

    vocab:dict[int,bytes]={index:bytes([index]) for index in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)]=special_token.encode("utf-8")
    merges:list[Pair]=[]

    pair_counts,pair_to_token_ids=build_pair_data(sequences,frequencies)
    heap=[
        HeapEntry(pair=pair,count_snapshot=count)
        for pair,count in pair_counts.items()
    ]
    heapq.heapify(heap)

    while len(vocab)<vocab_size:
        if not pair_counts:
            break

        while heap:
            entry=heapq.heappop(heap)
            #lazy invalidation
            if entry.pair not in pair_counts:
                continue
            if entry.count_snapshot!=pair_counts[entry.pair]:
                continue
            best_pair=entry.pair
            break
        else:
            break

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
            current_count=pair_counts[pair]
            if current_count<0:
                raise RuntimeError(f"pair count became negative:{pair}")
            elif current_count==0:
                del pair_counts[pair]
            else:
                new_entry=HeapEntry(pair=pair,count_snapshot=current_count)
                heapq.heappush(heap,new_entry)
        
        new_token=best_pair[0]+best_pair[1]
        vocab[len(vocab)]=new_token
        merges.append(best_pair)

    BPE_build_merge_end=time.perf_counter()

    print(f"pretokenization time: {pretokenization_end - pretokenization_start:.2f} seconds")
    print(f"BPE build merge time: {BPE_build_merge_end - BPE_build_merge_start:.2f} seconds")

    return vocab,merges
