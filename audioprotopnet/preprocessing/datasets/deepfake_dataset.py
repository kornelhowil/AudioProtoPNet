import os
import glob
import random
from typing import Optional, List, Dict
import datasets
from datasets import Dataset, DatasetDict, Audio
from birdset.datamodule.base_datamodule import (
    BaseDataModuleHF,
    DatasetConfig,
    LoadersConfig,
)
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from birdset.utils import pylogger
import numpy as np

log = pylogger.get_pylogger(__name__)

class DeepfakeDataModule(BaseDataModuleHF):
    def __init__(
        self,
        dataset: DatasetConfig = DatasetConfig(),
        loaders: LoadersConfig = LoadersConfig(),
        transforms: BirdSetTransformsWrapper = BirdSetTransformsWrapper(),
        mapper: None = None,
    ):
        super().__init__(
            dataset=dataset, loaders=loaders, transforms=transforms
        )
        self.mapper = mapper
        
        # We need to set the task if not already set, defaulting to multiclass (binary classification here)
        if not self.dataset_config.task:
            self.dataset_config.task = "multiclass"
            
    @property
    def num_classes(self):
        return 2

    def _load_data(self, decode: bool = True) -> DatasetDict:
        """
        Load audio dataset from local LJSpeech and WaveFake directories.
        """
        project_root = self.dataset_config.data_dir 
        
        ljspeech_dir = os.path.join(project_root, "datasets", "LJSpeech-1.1", "wavs")
        wavefake_base_dir = os.path.join(project_root, "datasets", "wavefake")
        
        wavefake_method = self.dataset_config.hf_name 
        if not wavefake_method:
             wavefake_method = "ljspeech_melgan" 
        
        wavefake_dir = os.path.join(wavefake_base_dir, wavefake_method)
        
        if not os.path.exists(ljspeech_dir):
            raise FileNotFoundError(f"LJSpeech directory not found at {ljspeech_dir}")
        if not os.path.exists(wavefake_dir):
            raise FileNotFoundError(f"WaveFake directory not found at {wavefake_dir}")

        real_files = glob.glob(os.path.join(ljspeech_dir, "*.wav"))
        real_ids = [os.path.splitext(os.path.basename(f))[0] for f in real_files]
        real_ids.sort() # Ensure deterministic order

        test_ids = real_ids[:1000]
        val_ids = real_ids[1000:2000]
        train_ids = real_ids[2000:]
        
        data_splits = {
            "train": train_ids,
            "valid": val_ids,
            "test": test_ids
        }
        
        dataset_dict = {}
        
        for split_name, ids in data_splits.items():
            audio_paths = []
            labels = []
            for file_id in ids:
                real_path = os.path.join(ljspeech_dir, f"{file_id}.wav")
                audio_paths.append(real_path)
                labels.append(1) # 1 for Real
                
                if os.path.exists(os.path.join(wavefake_dir, f"{file_id}_gen.wav")):
                    fake_path = os.path.join(wavefake_dir, f"{file_id}_gen.wav")
                elif os.path.exists(os.path.join(wavefake_dir, f"{file_id}_generated.wav")):
                    fake_path = os.path.join(wavefake_dir, f"{file_id}_generated.wav")
                else:
                    fake_path = os.path.join(wavefake_dir, f"{file_id}.wav")
                audio_paths.append(fake_path)
                labels.append(0) # 0 for Fake
            
            # Create Dataset
            dataset_dict[split_name] = Dataset.from_dict({
                "audio": audio_paths,
                "labels": labels
            })
            
            print(f"Loaded {len(audio_paths)} samples for {split_name} split.")
            
            if split_name == "train":
                self.num_train_labels = self._count_labels(labels)
            
            # Cast audio column
            dataset_dict[split_name] = dataset_dict[split_name].cast_column(
                column="audio",
                feature=Audio(
                    sampling_rate=self.dataset_config.sampling_rate,
                    mono=True,
                    decode=decode,
                ),
            )

        return DatasetDict(dataset_dict)

    def _preprocess_data(self, dataset):
        if self.dataset_config.task == "multilabel":
            log.info(">> One-hot-encode classes")
            dataset = dataset.map(
                self._classes_one_hot,
                batched=True,
                batch_size=500,
                load_from_cache_file=True,
                num_proc=1,
                desc="One-hot-encoding"
            )
        return dataset
