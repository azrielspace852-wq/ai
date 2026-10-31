import json, os, sys
from collections import Counter

SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>']

def parse_line(line):
    try:
        o = json.loads(line)
    except Exception:
        return None
    if not isinstance(o, dict):
        return None

    # 1) {"text":"..."}
    if isinstance(o.get('text'), str):
        return o['text']

    # 2) {"question":"...","answer":"<think>..</think>\n<output>..</output>"}
    if 'question' in o and 'answer' in o:
        return f"Question: {o['question']}\nAnswer: {o['answer']}"

    # 3) {"prompt":"...","response":"<think>..</think>\n<output>..</output>"}
    if 'prompt' in o and 'response' in o:
        return f"Prompt: {o['prompt']}\nResponse: {o['response']}"

    # 4) {"instructions":"...","messages":[{role,content},...]}
    if 'instructions' in o and isinstance(o.get('messages'), list):
        parts = [f"Instructions: {o['instructions']}"]
        for m in o['messages']:
            if isinstance(m, dict) and 'role' in m and 'content' in m:
                parts.append(f"{m['role']}: {m['content']}")
        return '\n'.join(parts)

    return None

def main():
    paths = [p.strip() for p in open('jsonl_files.txt') if p.strip()]
    texts, skipped = [], 0
    for p in paths:
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = parse_line(line)
                if t: texts.append(t)
                else: skipped += 1

    print(f"[info] parsed={len(texts)} skipped={skipped}")
    if not texts:
        sys.exit("tidak ada sample valid")

    vocab_size = int(os.environ.get('VOCAB_SIZE', '2048'))
    seq_len    = int(os.environ.get('SEQ_LEN',    '256'))

    counter = Counter()
    for t in texts: counter.update(t)

    vocab  = {tok: i for i, tok in enumerate(SPECIAL)}
    budget = max(0, vocab_size - len(SPECIAL))
    for ch, _ in counter.most_common(budget):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    print(f"[info] vocab={len(vocab)}")

    unk, bos, eos = vocab['<unk>'], vocab['<bos>'], vocab['<eos>']

    seqs = []
    for t in texts:
        ids = [bos] + [vocab.get(ch, unk) for ch in t] + [eos]
        for i in range(0, len(ids), seq_len):
            chunk = ids[i:i + seq_len]
            if len(chunk) >= 2:
                seqs.append(chunk)
    print(f"[info] sequences={len(seqs)}")

    json.dump(seqs, open('dataset.json', 'w'))
    json.dump(vocab, open('vocab.json', 'w', encoding='utf-8'), ensure_ascii=False)

if __name__ == '__main__':
    main()
