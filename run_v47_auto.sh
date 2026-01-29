# Automation script for V47 Pipeline (Fast Version)
echo "=== V47 Automation Strategy (FAST) ==="

echo "1. Checking/Launching Fast Text Mining..."
# Check if fast mining is already running, if not start it
if ! pgrep -f "cafa_v39_fast_text_mining.py" > /dev/null; then
    echo "   Starting cafa_v39_fast_text_mining.py..."
    nohup venv/bin/python cafa_v39_fast_text_mining.py > text_mining_fast.log 2>&1 &
fi

echo "2. Monitoring Fast Text Mining process..."
while pgrep -f "cafa_v39_fast_text_mining.py" > /dev/null; do
    echo "   Fast Mining running... checking again in 5 minutes"
    sleep 300
done

echo "3. Fast Mining finished. Running original V39 to generate SBERT embeddings..."
# This will skip fetching (since cache is full) and do embeddings + prediction
venv/bin/python cafa_v39_text_mining.py

echo "4. V39 complete. Verifying output..."
if [ -f "submission_v39_text_only.tsv" ]; then
    echo "   Success: submission_v39_text_only.tsv found."
    
    echo "5. Running V47 GORetriever Reranker..."
    venv/bin/python cafa_v47_goretriever_rerank.py
    
    if [ -f "submission_v47_text_rerank.tsv" ]; then
        echo "6. Submitting V47 to Kaggle..."
        # IMPORTANT: Do NOT hardcode tokens in scripts.
        # Set your Kaggle token in the environment before running, e.g.:
        # export KAGGLE_API_TOKEN=<your_token>
        cp submission_v47_text_rerank.tsv submission.tsv
        venv/bin/kaggle competitions submit -c cafa-6-protein-function-prediction -f submission.tsv -m "V47: Text-First Pivot (GORetriever Rerank)"
        echo "   DONE! V47 Submitted."
    else
        echo "   Error: V47 script failed to generate output."
    fi
else
    echo "   Error: V39 finished but output file missing!"
fi
