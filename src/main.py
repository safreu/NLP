
from data.one_stop_english_corpus import OneStopEnglish
from preprocessing.dataset_builder import split_pairs, to_dataset
from training.trainer import train_model
from evaluation.evaluate import evaluate_model
from evaluation.file_writer import write_results, create_run_dir, write_stats

def main() -> None:
    print("setting up")
    
    run_dir = create_run_dir()
    model_dir = run_dir / "model"
    results_path = run_dir / "scores.json"
    predictions_path = run_dir / "predictions.json"
    stats_path = run_dir / "stats.json"
    
    print("Data loading")
    onestop: OneStopEnglish = OneStopEnglish.load_from_disk();   
    print("Data loaded")
    
    pairs = onestop.as_training_pairs()
    
    train, valid, test = split_pairs(pairs)
     
    train_dataset = to_dataset(train)
    valid_dataset = to_dataset(valid)
    
    print("printing stats")
    write_stats(onestop.stats, stats_path)

    
    print("starting training")
    train_model(train=train_dataset, valid=valid_dataset, path=model_dir)
    print("finished training")
    
    print("starting evaluation")
    results = evaluate_model(test, model_dir, predictions_path)
    print("finished evaluation")
    
    #write_results(results, results_path)
        

if __name__ == "__main__":
    main()
