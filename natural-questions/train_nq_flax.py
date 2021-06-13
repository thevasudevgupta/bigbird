import json
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import joblib
import optax
from datasets import load_dataset
from flax import traverse_util
from flax.serialization import from_bytes, to_bytes
from flax.training import train_state
from flax.training.common_utils import shard
from tqdm.auto import tqdm

from transformers import (BigBirdConfig, BigBirdTokenizerFast,
                          FlaxBigBirdForQuestionAnswering)
from transformers.models.big_bird.modeling_flax_big_bird import \
    FlaxBigBirdForQuestionAnsweringModule

##########################################################
# How can we inherit from HuggingFace Flax Transformers?
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
        self.cls = nn.Dense(5, dtype=self.dtype)

    def __call__(self, *args, **kwargs):
        kwargs["return_dict"] = True
        outputs = super().__call__(*args, **kwargs)
        cls_out = self.cls(outputs.pooled_output)
        return outputs.start_logits, outputs.end_logits, cls_out


class FlaxBigBirdForNaturalQuestions(FlaxBigBirdForQuestionAnswering):
    module_class = FlaxBigBirdForNaturalQuestionsModule


def calculate_loss_for_nq(
    start_logits, start_labels, end_logits, end_labels, pooled_logits, pooler_labels
):
    def cross_entropy(logits, labels, reduction=None):
        """
        Args:
            logits: bsz, seqlen, vocab_size
            labels: bsz, seqlen
        """
        vocab_size = logits.shape[-1]
        labels = (labels[..., None] == jnp.arange(vocab_size)[None]).astype("f4")
        logits = jax.nn.log_sigmoid(logits, axis=-1)
        loss = -jnp.sum(labels * logits, axis=-1)
        if reduction is not None:
            loss = reduction(loss)
        return loss

    cross_entropy = partial(cross_entropy, reduction=jnp.mean)
    start_loss = cross_entropy(start_logits, start_labels)
    end_loss = cross_entropy(end_logits, end_labels)
    pooled_loss = cross_entropy(pooled_logits, pooler_labels)
    return (start_loss + end_loss + pooled_loss) / 3


@dataclass
class Args:
    model_id: str = "google/bigbird-base-trivia-itc"
    lr: float = 1e-5
    eval_steps: int = 4
    save_steps: int = 6

    batch_size: int = 8
    max_epochs: int = 2

    # tx_args
    lr: float = 1e-4
    init_lr: float = 0.0
    warmup_steps: int = 10
    weight_decay: float = 1e-3

    save_dir: str = "bigbird-roberta-natural-questions"
    base_dir: str = "training-expt"
    tr_data_path: str = "data/nq-training.jsonl"
    val_data_path: str = "data/nq-validation.jsonl"

    def __post_init__(self):
        os.path.mkdir(self.base_dir, exist_ok=True)
        self.save_dir = os.path.join(self.base_dir, self.save_dir)


@dataclass
class DataCollator:

    pad_id: int
    max_length: int = 4096  # no dynamic padding on TPUs

    def __call__(self, batch):
        batch = self.collate_fn(batch)
        return batch

    def collate_fn(self, features):
        input_ids, attention_mask = self.fetch_inputs(
            [x["input_ids"] for x in features]
        )
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_labels": [x["start_token"] for x in features],
            "end_labels": [x["end_token"] for x in features],
            "pooled_labels": [x["category"] for x in features],
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
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        batch = shard(batch)
        yield batch


def train_step(state, drp_rng, **model_inputs):
    def loss_fn(params):
        start_labels = model_inputs.pop("start_labels")
        end_labels = model_inputs.pop("end_labels")
        pooled_labels = model_inputs.pop("pooled_labels")

        outputs = state.apply(
            **model_inputs, params=params, drp_rng=drp_rng, train=True
        )
        start_logits, end_logits, pooled_logits = outputs

        return calculate_loss_for_nq(
            start_logits,
            start_labels,
            end_logits,
            end_labels,
            pooled_logits,
            pooled_labels,
        )

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def val_step(state, drp_rng, **model_inputs):
    start_labels = model_inputs.pop("start_labels")
    end_labels = model_inputs.pop("end_labels")
    pooled_labels = model_inputs.pop("pooled_labels")

    outputs = state.apply(
        **model_inputs, params=state.params, drp_rng=drp_rng, train=False
    )
    start_logits, end_logits, pooled_logits = outputs

    loss = calculate_loss_for_nq(
        start_logits, start_labels, end_logits, end_labels, pooled_logits, pooled_labels
    )
    return loss


@dataclass
class Trainer:
    args: Args
    data_collator: Callable
    train_step_fn: Callable
    val_step_fn: Callable

    # def __post_init__(self):
    #     self.training_step = jax.pmap(self.training_step)
    #     self.validation_step = jax.pmap(self.validation_step)

    def create_state(self, model, tx, ckpt_dir=None):
        params = model.params
        state = train_state.TrainState.create(
            apply_fn=model.__call__, params=params, tx=tx
        )
        if ckpt_dir is not None:
            params, opt_state, step, args, data_collator = restore_checkpoint(
                ckpt_dir, state
            )
            tx_args = {
                "lr": args.lr,
                "init_lr": args.init_lr,
                "warmup_steps": args.warmup_steps,
                "num_train_steps": calcuate_num_steps(
                    args.tr_data_path, args.batch_size, args.max_epochs
                ),
                "weight_decay": args.weight_decay,
            }
            tx = build_tx(**tx_args)
            state = train_state.TrainState(
                step=step,
                apply_fn=model.__call__,
                params=params,
                tx=tx,
                opt_state=opt_state,
            )
            self.args = args
            self.data_collator = data_collator
            model.params = params
        return state

    def train(self, state, save_fn):
        args = self.args
        val_dataset = get_batched_dataset(args.val_data_path, args.batch_size)

        for epoch in args.max_epochs:
            running_loss = jnp.array(0, dtype=jnp.float32)
            tr_dataset = get_batched_dataset(
                args.tr_data_path, args.batch_size, seed=epoch
            )
            for batch in tqdm(tr_dataset, desc=f"Running EPOCH-{epoch}"):
                batch = self.data_collator(batch)
                state, loss = self.train_step_fn(state, batch)
                running_loss += loss
                tr_loss = running_loss / (state.step + 1)

                if state.step % args.eval_steps == 0:
                    eval_loss = self.evaluate(state, val_dataset)

                if state.step % args.save_steps == 0:
                    print("SAVING MODEL AFTER STEP:", state.step, end="... ")
                    save_fn(args.save_dir + f"-epoch-{epoch}", params=state.params)
                    print("DONE")

        return tr_loss, eval_loss

    def evaluate(self, state, drp_rng, dataset):
        running_loss = jnp.array(0, dtype=jnp.float32)
        for i, batch in tqdm(enumerate(dataset), desc="Evaluating ... "):
            batch = self.data_collator(batch)
            loss = self.val_step_fn(state, drp_rng, **batch)
            running_loss += loss
        return running_loss / (i + 1)

    def save_checkpoint(self, save_dir, state):
        print(f"SAVING CHECKPOINT IN {save_dir}", end=" ... ")
        self.model.save_pretrained(save_dir, params=state.params)
        with open(os.path.join(save_dir, "opt_state.msgpack"), "wb") as f:
            f.write(to_bytes(state.opt_state))
        joblib.dump(args, os.path.join(save_dir, "args.joblib"))
        joblib.dump(self.data_collator, os.path.join(save_dir, "data_collator.joblib"))
        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
            json.dump({"step": state.step}, f)
        print("DONE")


def restore_checkpoint(save_dir, state):
    print(f"RESTORING CHECKPOINT FROM {save_dir}", end=" ... ")
    with open(os.path.join(save_dir, "flax_model.msgpack"), "rb") as f:
        params = from_bytes(state.params, f.read())

    with open(os.path.join(save_dir, "opt_state.msgpack"), "rb") as f:
        opt_state = from_bytes(state.opt_state, f.read())

    args = joblib.load(os.path.join(save_dir, "args.joblib"))
    data_collator = joblib.load(os.path.join(save_dir, "data_collator.joblib"))

    with open(os.path.join(save_dir, "training_state.json"), "r") as f:
        training_state = json.load(f)
    step = training_state["step"]

    print("DONE")
    return params, opt_state, step, args, data_collator


def build_tx(lr, init_lr, warmup_steps, num_train_steps, weight_decay):
    def weight_decay_mask(params):
        params = traverse_util.unflatten(params)
        mask = {
            k: (v[-1] != "bias" and v[-2:] != ("LayerNorm", "scale"))
            for k, v in params.items()
        }
        return traverse_util.unflatten_dict(mask)

    decay_steps = num_train_steps - warmup_steps
    warmup_fn = optax.linear_schedule(
        init_value=init_lr, end_value=lr, transition_steps=warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=lr, end_value=1e-7, transition_steps=decay_steps
    )
    lr = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
    )

    return optax.adamw(
        learning_rate=lr, weight_decay=weight_decay, mask=weight_decay_mask
    )


def calcuate_num_steps(data_path, batch_size, max_epochs):
    steps = 0
    for _ in get_batched_dataset(data_path, batch_size):
        steps += 1
    return steps * max_epochs


if __name__ == "__main__":
    args = Args()
    print(args)

    model = FlaxBigBirdForQuestionAnswering.from_pretrained(args.model_id)
    tokenizer = BigBirdTokenizerFast.from_pretrained(args.model_id)
    data_collator = DataCollator(pad_id=tokenizer.pad_token_id, max_length=4096)

    trainer = Trainer(
        args=args,
        data_collator=data_collator,
        train_step_fn=train_step,
        val_step_fn=val_step,
    )

    num_steps = calcuate_num_steps(args.tr_data_path, args.batch_size, args.max_epochs)
    tx_args = {
        "lr": args.lr,
        "init_lr": args.init_lr,
        "warmup_steps": args.warmup_steps,
        "num_train_steps": num_steps,
        "weight_decay": args.weight_decay,
    }
    tx = build_tx(**tx_args)

    state = trainer.create_state(model, tx, ckpt_dir=None)
    trainer.train(state, save_fn=model.save_pretrained)

    model.save_pretrained(args.save_dir, params=state.params)
