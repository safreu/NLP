
from data.one_stop_english_corpus import OneStopEnglish
from training.dataset_builder import split_pairs, to_dataset
from training.trainer import train_model

def main() -> None:
    print("setting up")
    onestop: OneStopEnglish = OneStopEnglish.load_from_disk();    

    pairs = onestop.as_training_pairs()
    
    train, valid, test = split_pairs(pairs)
    
    train_dataset = to_dataset(train)
    valid_dataset = to_dataset(valid)
    
    train_model(train=train_dataset, valid=valid_dataset)



if __name__ == "__main__":
    main()
