import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from functools import partial
from typing import Callable

from flax.training import train_state
from flax.training.common_utils import shard
from dataclasses import dataclass
from datasets import load_dataset

from transformers import (
    FlaxBigBirdForQuestionAnswering,
    FlaxBigBirdForQuestionAnsweringModule,
    BigBirdTokenizerFast,
    BigBirdConfig
)

##########################################################
# How can we inherit from HuggingFace Flax Transformers ?
# explore it below 👇👇
##########################################################

class FlaxBigBirdForNaturalQuestionsModule(FlaxBigBirdForQuestionAnsweringModule):
    """
    BigBirdForQuestionAnswering with CLS Head over the top for predicting category

    This way we can load its weights with FlaxBigBirdForQuestionAnswering
    """
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    add_pooling_layer: bool = True

    def setup(self):
        super().setup()
        self.cls = nn.Dense(5)

    def __call__(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        cls_out = self.cls(outputs["pooler_output"]) # TODO: implement this before merge :-)
        return outputs["start_logits"], outputs["end_logits"], cls_out

class FlaxBigBirdForNaturalQuestions(FlaxBigBirdForQuestionAnswering):
    module_class = FlaxBigBirdForNaturalQuestionsModule


@dataclass
class Args:
    model_id: str = "google/bigbird-base-trivia-itc"
    lr: float = 1e-5

    batch_size: int = 8

    train_data: str = "data/nq-training.jsonl"
    val_data: str = "data/nq-validation.jsonl"

@dataclass
class DataCollator:

    pad_id: int
    max_length: int = 4096 # no dynamic padding on TPUs

    def __call__(self, batch):
        batch = self.collate_fn(batch)
        return batch

    def collate_fn(self, features):
        input_ids, attention_mask = self.fetch_inputs([x["input_ids"] for x in features])
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": [x['start_token'] for x in features],
            "end_positions": [x['end_token'] for x in features],
            "pooler_label": [x["category"] for x in features],
        }
        batch = jax.tree_map(partial(jnp.array, dtype=jnp.int32), batch)
        return batch

    def fetch_inputs(self, input_ids: list):
        inputs = [self._fetch_inputs(ids) for ids in input_ids]
        return zip(*inputs)

    def _fetch_inputs(self, input_ids: list):
        attention_mask = [1 for _ in range(len(input_ids))]
        while len(input_ids) < self.max_length:
            input_ids.append(self.pad_id)
            attention_mask.append(0)
        return input_ids, attention_mask


def get_batched_dataset(file_path, batch_size, seed=None):
    dataset = load_dataset("json", data_files=file_path)["train"]
    num_samples = len(dataset)
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    # dropping last batch
    dataset = dataset.select(range(num_samples - num_samples % batch_size))
    for i in range(num_samples):
        batch = dataset[i*batch_size: (i+1)*batch_size]
        batch = shard(batch)
        yield batch


def train_step(state, batch, drp_rng):
    def loss_fn(**kwargs):
        pred = state.apply(**kwargs, params=state.params, drp_rng=drp_rng, train=True)
        return state.loss_fn(labels, pred["logits"])

    return


@dataclass
class Trainer:
    args: Args
    model: nn.Module
    data_collator: Callable
    training_step: Callable
    validation_step: Callable

    def __post_init__(self):
        self.training_step = jax.pmap(self.training_step)
        self.validation_step = jax.pmap(self.validation_step)

    def setup(self):

        return state

    def train(self, state):
        args = self.args
        val_dataset = get_batched_dataset(args.train_data, args.batch_size)

        for epoch in self.epochs:

            for batch in get_batched_dataset(args.train_data, args.batch_size, seed=epoch):
                batch = self.data_collator(batch)
                state, metrics = self.training_step(state, batch)

        return

    def evaluate(self, dataset):

        return metrics

if __name__ == '__main__':
    args = Args()

    model = FlaxBigBirdForQuestionAnswering.from_pretrained(args.model_id)
    tokenizer = BigBirdTokenizerFast.from_pretrained(args.model_id)

    state = train_state.TrainState.create()
    collate_fn = DataCollator()

    

