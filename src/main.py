
from data.one_stop_english_corpus import OneStopEnglish
from training.dataset_builder import split_pairs, to_dataset
from training.trainer import train_model
from evaluation.evaluate import evaluate_model

def main() -> None:
    print("setting up")
    onestop: OneStopEnglish = OneStopEnglish.load_from_disk();    

    pairs = onestop.as_training_pairs()
    
    train, valid, test = split_pairs(pairs)
    
    train_dataset = to_dataset(train)
    valid_dataset = to_dataset(valid)
    
    #train_model(train=train_dataset, valid=valid_dataset)

    results = evaluate_model(test)
    
    print(f"BLEU: {results['bleu']:.4f}")
    print(f"ROUGE-L: {results['rouge-l']:.4f}")
    print(f"SARI: {results['sari']:.4f}")

    print(f"BERTScore F1: {results['bert']['f1_mean']:.4f}")

    print(f"Token F1: {results['f1']['f1_mean']:.4f}")

    print(f"Flesch-Kincaid: {results['flesch']['mean']:.4f}")

if __name__ == "__main__":
    main()
