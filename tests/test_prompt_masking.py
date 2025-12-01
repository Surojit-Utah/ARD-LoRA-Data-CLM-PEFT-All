import json
import os
import tempfile
from dataloader.bayesian_peft import load_bayesian_peft_dataset


class DummyTokenizer:
    """A tiny tokenizer that splits on spaces and maps tokens to incremental ids.
    Designed for testing masking logic only.
    """
    def __init__(self):
        self.vocab = {"": 0}

    def _encode(self, text):
        if text == "":
            return []
        toks = text.split()
        ids = []
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
            ids.append(self.vocab[t])
        return ids

    def __call__(self, texts, truncation=True, max_length=128, padding=None):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        input_ids = [self._encode(t)[:max_length] for t in texts]
        # mimic padding to max_length if requested
        if padding == "max_length":
            input_ids = [ids + [0] * (max_length - len(ids)) for ids in input_ids]
        return {"input_ids": input_ids}


def test_prompt_masking_tmpfiles():
    # create temporary dataset directory with a small train.jsonl
    with tempfile.TemporaryDirectory() as tmpdir:
        ds_dir = os.path.join(tmpdir, "dataset")
        os.makedirs(ds_dir, exist_ok=True)
        train_path = os.path.join(ds_dir, "train.jsonl")
        example = {"instruction": "Say hello", "input": "to the world", "output": "Hello world"}
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(example) + '\n')

        tokenizer = DummyTokenizer()
        train_ds, val_ds = load_bayesian_peft_dataset(tmpdir, "dataset", tokenizer=tokenizer, max_len=20, sample_size=1)
        assert train_ds is not None
        # get the first tokenized item
        item = train_ds[0]
        input_ids = item['input_ids']
        labels = item['labels']
        # Ensure labels and input_ids same length
        assert len(input_ids) == len(labels)
        # prompt = 'Say hello to the world' -> tokens count > 0 and should be masked (-100)
        # response = 'Hello world' -> its tokens should not be -100
        # find non-zero positions in labels
        masked_positions = [i for i, v in enumerate(labels) if v == -100]
        assert len(masked_positions) > 0
        # response tail should contain non -100 values
        nonmasked_positions = [i for i, v in enumerate(labels) if v != -100 and v != 0]
        assert len(nonmasked_positions) > 0
