import os

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
MODEL_ID = "HiTZ/Latxa-Llama-3.1-8B-Instruct"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


DATASET_ID = "Iker/calibration-dataset"
DATASET_SPLIT = "train"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 2048
MAX_SEQUENCE_LENGTH = 8192

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split="train")
ds = ds.shuffle(seed=42)
ds = ds.select(range(NUM_CALIBRATION_SAMPLES))


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
model_input = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Who are you?"}],
    tokenize=True,
    return_tensors="pt",
)
output = model.generate(
    model_input.to(model.device), max_new_tokens=100, temperature=0.6, do_sample=True
)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")


# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-w8a8_fp8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)


model.push_to_hub("Iker/Latxa-Llama-3.1-8B-Instruct-w8a8_fp8")
tokenizer.push_to_hub("Iker/Latxa-Llama-3.1-8B-Instruct-w8a8_fp8")
