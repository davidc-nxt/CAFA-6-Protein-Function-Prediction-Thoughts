---
description: Run overnight data fetching jobs (InterPro, UniProt text)
---

# Long-Running Data Fetch Jobs

## InterPro Domain Fetch (Domain2GO)

This fetches InterPro domains for all test proteins from UniProt API.

### Start Job
```bash
nohup venv/bin/python cafa_v37_domain2go_overnight.py > domain2go_overnight_output.log 2>&1 &
```

### Monitor Progress
```bash
# Check if running
ps aux | grep cafa_v37 | grep -v grep

# Check checkpoint (proteins processed)
cat interpro_checkpoint.txt

# Check logs
tail -20 domain2go_overnight.log
```

### Resume After Interruption
Simply run the same command - it resumes from checkpoint.

### Expected Duration
~18-24 hours for 224k proteins

---

## UniProt Text Mining Fetch

This fetches protein descriptions for SBERT embeddings.

### Start Job
```bash
nohup venv/bin/python cafa_v39_text_mining.py > text_mining_output.log 2>&1 &
```

### Monitor Progress
```bash
# Check if running
ps aux | grep cafa_v39 | grep -v grep

# Check checkpoint
cat text_mining_checkpoint.txt

# Check logs
tail -20 text_mining.log
```

### Resume After Interruption
Simply run the same command - it resumes from checkpoint.

### Expected Duration
~18-24 hours for 224k proteins

---

## Notes
- Both scripts use checkpointing every 5000 proteins
- Cache files are saved incrementally
- UniProt API is free but rate-limited
