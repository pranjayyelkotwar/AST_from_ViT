# Speech Command Classification with ViT

This repository contains a PyTorch implementation for classifying speech commands using a Vision Transformer (ViT) model. The model processes audio data by converting waveforms into mel spectrograms and then classifies the spoken commands.

## Features
- Uses the `SpeechCommands` dataset from `torchaudio`
- Converts audio waveforms into mel spectrograms
- Processes spectrograms using a Vision Transformer (ViT)
- Implements training, validation, and early stopping
- Logs training metrics and visualizes performance

## Installation
Ensure you have Python 3.8+ and install the required dependencies:
```bash
pip install torch torchaudio timm tqdm matplotlib
```

## Usage
Run the script to train the model:
```bash
python script.py
```

## Dataset
The dataset is automatically downloaded from `torchaudio.datasets.SPEECHCOMMANDS` when running the script.

## Model Architecture
The model is based on ViT and fine-tuned for speech command classification. It adapts an audio spectrogram transformer (AST) approach to process mel spectrograms.

## Spectrogram Example
The model converts speech waveforms into mel spectrograms before feeding them into the ViT.

![Spectrogram Example](spectrogram.png)

## Training Metrics
The following plot shows the training and validation loss/accuracy over epochs.

![Training Metrics](training_metrics.png)

## Results
- The trained model is saved as `best_model.pth`.
- Training metrics are logged in `training_metrics.csv`.

## Acknowledgments
- The Speech Commands dataset: [Google Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands)
- Audio Spectrogram Transformer (AST) model inspiration

## License
This project is licensed under the MIT License.

