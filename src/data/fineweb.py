import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

tknzr = tiktoken.get_encoding("gpt2")

def get_fineweb_data(datasets_base_dir, num_proc=40):
    """Processes and stores FineWeb-Edu dataset for training."""
    FW_DATA_PATH = os.path.join(datasets_base_dir, "fineweb_edu/")
    
    if not os.path.exists(os.path.join(FW_DATA_PATH, "train.bin")):
        os.makedirs(FW_DATA_PATH, exist_ok=True)
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", ame="sample-10BT")

        # Split dataset into train and validation sets
        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(example["text"])  # Tokenize text
            ids.append(tknzr.eot_token)  # Append End-of-Text token
            return {"ids": ids, "len": len(ids)}

        # Tokenize dataset in parallel
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # Save tokenized dataset as .bin files
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(FW_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # Since GPT-2 token IDs < 65536

            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))  # Adapt batch size

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch  # Store tokens
                idx += len(arr_batch)
            arr.flush()  # Save to disk

    return {
        "train": os.path.join(FW_DATA_PATH, "train.bin"),
        "val": os.path.join(FW_DATA_PATH, "val.bin"),
    }