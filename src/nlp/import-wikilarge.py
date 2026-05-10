from datasets import load_dataset

dataset = load_dataset("an-atlas/wikilarge")

print("Dataset structure: ")
print(dataset)
print()

print("Features: ")
print(dataset["train"].features)
print()

print("Split sizes: ")
for split in dataset:
    print(f"    {split}: {len(dataset[split])} entries")
print()

print("Average sentence length: ")
normal_sum = 0
simple_sum = 0
for data in dataset["train"]:
    normal_sum += len(data["Normal"].split())
    simple_sum += len(data["Simple"].split())

print(f"    Normal: {normal_sum / len(dataset["train"]):.2f} words")
print(f"    Simple: {simple_sum / len(dataset["train"]):.2f} words")
print()

print("Examples: ")
print("   Trainging sample: ")
sample = dataset["train"][1]
print(f"    Normal: {sample['Normal']}")
print(f"    Simple: {sample['Simple']}")
print()

print("   Validation sample: ")
sample = dataset["validation"][1]
print(f"    Normal: {sample['Normal']}")
print(f"    Simple: {sample['Simple']}")
print()

print("   Testing sample: ")
sample = dataset["test"][1]
print(f"    Normal: {sample['Normal']}")
print(f"    Simple: {sample['Simple']}")
print()
