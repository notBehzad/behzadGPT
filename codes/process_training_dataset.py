import sentencepiece as sp
import multiprocessing as mp
import numpy as np

NUM_CORES=mp.cpu_count()
MODEL_PATH="./tokenizer/unigram_8k.model"
BATCH_SIZE=100_000
INPUT_FILE="./dataset/train/fine_tune.txt"
OUTPUT_FILE="./dataset/train/fine_tune.bin"

def tokenize_lines(MODEL_PATH, lines):
    tokenizer=sp.SentencePieceProcessor()
    tokenizer.load(MODEL_PATH)
    tokens=[]
    for line in lines:
        if line.strip():
            tokens.extend(tokenizer.encode_as_ids(line))
    return np.array(tokens, dtype=np.int16)

def main():
    with open(INPUT_FILE, "r",encoding="utf-8") as infile,\
        open(OUTPUT_FILE,"wb") as outfile:
        pool=mp.Pool(NUM_CORES)
        total_tokens=0
        batch=[]
        for line in infile:
            batch.append(line)
            if len(batch)>=BATCH_SIZE:
                chunks=[batch[i::NUM_CORES] for i in range(NUM_CORES)]
                results=pool.starmap(tokenize_lines,[(MODEL_PATH,chunk) for chunk in chunks])
                results=np.concatenate(results)

                total_tokens+=len(results)
                outfile.write(results.tobytes())
                print(f"{total_tokens} tokens processed",end="\r")
                batch=[]

        if batch:
            chunks=[batch[i::NUM_CORES] for i in range(NUM_CORES)]
            results=pool.starmap(tokenize_lines,[(MODEL_PATH,chunk) for chunk in chunks])
            results=np.concatenate(results)

            total_tokens+=len(results)
            outfile.write(results.tobytes())
            print(f"{total_tokens} tokens processed",end="\r")
            batch=[]

        pool.close()
        pool.join()

        print(f"\nTokenized binary file saved successfully at path: {OUTPUT_FILE}")


if __name__=="__main__":
    main()