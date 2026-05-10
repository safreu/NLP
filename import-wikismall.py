# Import the load_dataset function from our own wikismall_dataset module.
# This keeps the loading logic separate from the inspection/analysis code.
from load_wikismall_dataset import load_dataset

# Call load_dataset() to read all three splits (train/validation/test)
# from disk and return them as a single DatasetDict object.
dataset = load_dataset()


# ================== Structure ==================

# Print the full DatasetDict object.
# Shows each split name, column names (features), and number of rows.
print("Dataset structure: ")
print(dataset)
print()

# Print the data type of each column in the training split.
# Confirms both 'Normal' and 'Simple' are plain text (Value('string')).
print("Features: ")
print(dataset["train"].features)
print()


# ================== Split sizes ==================

# Loop over each split name ("train", "validation", "test")
# and print how many sentence pairs it contains.
print("Split sizes: ")
for split in dataset:
    print(f"    {split}: {len(dataset[split])} entries")
print()


# ================== Average sentence length ==================

# Count the total number of words across all training sentences
# for both the Normal and Simple columns.
print("Average sentence length: ")
normal_sum = 0
simple_sum = 0

# Iterate over every row in the training split.
# Each 'data' is a dict: {"Normal": "...", "Simple": "..."}.
for data in dataset["train"]:
    # .split() breaks the sentence into a list of words by whitespace.
    # len() counts how many words are in that list.
    normal_sum += len(data["Complex"].split())
    simple_sum += len(data["Simple"].split())

# Divide total word count by number of training rows to get the average.
# :.2f formats the number to 2 decimal places.
print(f"    Complex: {normal_sum / len(dataset['train']):.2f} words")
print(f"    Simple: {simple_sum / len(dataset['train']):.2f} words")
print()


# ================== Examples ==================

# Print one sentence pair from each split to visually verify the data.
# Index [1] fetches the second row (index 0 would be the first).
print("Examples: ")

print("   Training sample: ")
sample = dataset["train"][1]
print(f"    Complex: {sample['Complex']}")
print(f"    Simple: {sample['Simple']}")
print()

print("   Validation sample: ")
sample = dataset["validation"][1]
print(f"    Complex: {sample['Complex']}")
print(f"    Simple: {sample['Simple']}")
print()

print("   Testing sample: ")
sample = dataset["test"][1]
print(f"    Complex: {sample['Complex']}")
print(f"    Simple: {sample['Simple']}")
print()