
DOC_STRIDE = 2048
MAX_LENGTH = 4096
SEED = 42
# BATCH_SIZE = 32

GROUP_BY_LENGTH = True
LEARNING_RATE = 1.e-4
WARMUP_STEPS = 100
MAX_EPOCHS = 3
FP16 = False
SCHEDULER = "linear"

MODEL_ID = "google/bigbird-roberta-base"

CATEGORY_MAPPING = {
    "null": 0,
    "short": 1,
    "long": 2,
    "yes": 3,
    "no": 4,
}
