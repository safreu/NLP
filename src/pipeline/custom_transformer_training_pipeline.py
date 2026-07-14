
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

from configuration.custom_transformer_config import CustomTransformerTrainingConfig
from data.dataset_loader import DatasetLoader
from models.custom_transformer import Transformer
from pipeline.custom_transformer_evaluation_pipeline import CustomTransformerGenerationPipeline
from storage.paths import RunPaths
from training.custom_transformer_trainer import create_data_loader, train_model


class CustomTransformerTrainingPipeline:
    def __init__(
        self,
        name: str,
        dataset_loader: DatasetLoader,
        training_config: CustomTransformerTrainingConfig,
        run_paths: RunPaths,
        evaluation_pipeline: CustomTransformerGenerationPipeline,
    ) -> None:
        self.name = name
        self.dataset_loader = dataset_loader
        self.training_config = training_config
        self.run_paths = run_paths
        self.evaluation_pipeline = evaluation_pipeline
       
    
    def _create_model(self, tokenizer, device) -> Transformer:
        model = Transformer(
            trg_vocab_size=tokenizer.vocab_size,
            trg_pad_idx=tokenizer.pad_token_id,
            device=str(device),
            max_length=self.training_config.max_length,
        ).to(device)
        
        if self.training_config.freeze_encoder:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False

        return model
    
    def run(self) -> None:
        print(f"Running pipeline {self.name}")
        
        self.run_paths.pipeline_dir = self.name
        
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.training_config.tokenizer_name
        )
        
            
        train_pairs, valid_pairs, test_pairs = self.dataset_loader.load_pairs(False)
        
        train_loader = create_data_loader(
            pairs=train_pairs,
            tokenizer=tokenizer,
            max_length=self.training_config.max_length,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )
        
        model = self._create_model(
            tokenizer=tokenizer,
            device=device,
        )
        
        optimizer = optim.AdamW(
            model.decoder.parameters(),
            lr=self.training_config.decoder_learning_rate,
        )
        
        criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id
        )
        
        train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=self.training_config.num_epochs,
        )
        
        results = self.evaluation_pipeline.run(
            model=model,
            tokenizer=tokenizer,
            device=device,
            test_pairs=test_pairs,
        )
        
        print(f"Finished pipeline {self.name}")
        
        return results
        
        