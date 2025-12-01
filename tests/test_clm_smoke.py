import tempfile
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trainer.trainer_clm import build_clm_trainer


class SimpleTensorDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in {"input_ids": self.inputs[idx], "labels": self.labels[idx]}.items()}


def test_clm_smoke_train_one_step():
    # load a small model for smoke test
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    texts = ["Hello world.", "Goodbye world."]
    max_len = 32
    toks = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len)
    input_ids = toks["input_ids"]
    labels = [ids.copy() for ids in input_ids]

    ds = SimpleTensorDataset(input_ids, labels)

    cfg = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "train_epochs": 1,
        "fp16": False,
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "beta": 0.0,
    }

    with tempfile.TemporaryDirectory() as outdir:
        trainer = build_clm_trainer(model, tokenizer, ds, None, cfg, outdir)
        # run a short training (will process small dataset quickly)
        result = trainer.train()
        assert result.global_step >= 1
