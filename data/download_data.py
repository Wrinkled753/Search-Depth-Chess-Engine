import os
import json
import requests
import zstandard as zstd

LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.json.zst"
OUTPUT_FILE = "data/raw_evals.jsonl"
TARGET_SAMPLES = 300000

def download_and_extract(url, output_path, target_samples):
    print(f"Downloading stream from {url}...")
    
    # We use stream=True to process the zst file as it downloads
    response = requests.get(url, stream=True)
    response.raise_for_status()

    dctx = zstd.ZstdDecompressor()
    
    samples_collected = 0
    buffer = ""

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        # stream_reader lets us read decompressed text line by line
        with dctx.stream_reader(response.raw) as reader:
            print("Decompressing and parsing JSON lines...")
            while samples_collected < target_samples:
                # Read chunks and split into lines
                chunk = reader.read(65536).decode("utf-8", errors="ignore")
                if not chunk:
                    break
                
                buffer += chunk
                lines = buffer.split("\n")
                
                # The last item might be an incomplete line, keep it in the buffer
                buffer = lines.pop()
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        fen = data.get("fen")
                        evals = data.get("evals", [])
                        
                        if not fen or not evals:
                            continue
                            
                        # We want the evaluation at the highest depth.
                        # Usually, evals is a list. The Lichess format often has 
                        # 'evals': [{'pvs': [{'cp': 14}]}, ...] 
                        # Sometimes multiple items in 'evals', we'll take the last one 
                        # which typically corresponds to the highest depth searched.
                        last_eval = evals[-1]
                        pvs = last_eval.get("pvs", [])
                        if not pvs:
                            continue
                            
                        # The principal variation (first item in pvs) contains the score
                        score_obj = pvs[0]
                        
                        if "cp" in score_obj:
                            score_type = "cp"
                            score_val = score_obj["cp"]
                        elif "mate" in score_obj:
                            score_type = "mate"
                            score_val = score_obj["mate"]
                        else:
                            continue

                        # Save clean format
                        clean_data = {
                            "fen": fen,
                            "type": score_type,
                            "value": score_val
                        }
                        out_f.write(json.dumps(clean_data) + "\n")
                        samples_collected += 1
                        
                        if samples_collected % 50000 == 0:
                            print(f"Collected {samples_collected}/{target_samples} samples...")
                            
                        if samples_collected >= target_samples:
                            break
                    except json.JSONDecodeError:
                        continue

    print(f"Finished collecting {samples_collected} samples to {output_path}")

if __name__ == "__main__":
    download_and_extract(LICHESS_EVAL_URL, OUTPUT_FILE, TARGET_SAMPLES)
