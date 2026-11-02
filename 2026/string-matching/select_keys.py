"""Two independent log-frequency distributions; deterministic sampling per bucket."""
import hashlib
import math

def select(words, bytefreq, total_words, byte_count):
    selections = []
    anchors={w:bytefreq[ord(w[0])] for w in words}
    for distribution, counts, denominator in [('first_byte',anchors,byte_count),('whole_word',words,total_words)]:
        pool={w:c for w,c in counts.items() if w.isascii() and w.isalpha() and 3<=len(w)<=16 and c>0}
        low=min(pool.values()); top=max(pool.values())
        buckets=[[] for _ in range(10)]
        for w,c in pool.items():
            bucket=min(9,int(math.log(c/low)/math.log(top/low)*10))
            buckets[bucket].append(w)
        for b,members in enumerate(buckets):
            assert members, (distribution,b)
            ranked=sorted(members,key=lambda w:hashlib.sha256(f'hn-frequency-v1:{distribution}:{b}:{w}'.encode()).digest())
            chosen=ranked[:3]
            for rank,w in enumerate(chosen,1):
                selections.append(dict(distribution=distribution,bucket=b+1,
                    lower_count=low*(top/low)**(b/10),upper_count=low*(top/low)**((b+1)/10),
                    lower_probability=low*(top/low)**(b/10)/denominator,upper_probability=low*(top/low)**((b+1)/10)/denominator,
                    eligible_words=len(members),sample_rank=rank,needle=w,count=pool[w],
                    probability=pool[w]/denominator))
    # Explicit original-task and mechanism controls, kept separate from bucket sampling.
    controls=['youtube','YouTube','you','yes']
    keys=list(dict.fromkeys(controls+[r['needle'] for r in selections]))
    return keys,selections
