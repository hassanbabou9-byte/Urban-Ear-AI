# Urban Ear AI

Urban Ear AI classifies short urban audio clips into the ten UrbanSound8K
classes. It includes a Streamlit upload-and-predict app, a command-line
predictor, and scripts for preprocessing and training.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt`
- The included model at `models/urban_ear_model.h5` to run predictions
- The UrbanSound8K audio and metadata files to preprocess or train

TensorFlow's supported installation options vary by operating system. On
Windows, use a TensorFlow-supported Python/OS combination or a supported Linux
environment such as WSL.

## Install

Run these commands from the project root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the virtual environment with
`source .venv/bin/activate`.

## Run the app

```powershell
streamlit run app.py
```

The app accepts WAV audio and uses the first three seconds for classification.
The included model is approximately 11 MB and is kept in the repository; it is
small enough for a regular GitHub file.

For command-line prediction:

```powershell
python -m src.predict path\to\audio.wav
```

## Preprocess and train

Obtain the UrbanSound8K dataset from its authors and review its license and
attribution requirements. The dataset is not included in this repository.
Arrange the extracted files as follows:

```text
data/
  audio/
    fold1/
    ...
    fold10/
  metadata/
    UrbanSound8K.csv
```

Then run, from the project root:

```powershell
python -m src.preprocess
python -m src.train
```

Preprocessing writes `processed_data.npz` in the project root. It records the
official fold for each sample; training uses folds 1-8 for training, fold 9
for validation, and fold 10 for the final evaluation. The generated archive and
dataset files are excluded from Git because the archive is large and the
dataset must be obtained separately.

The model expects 128-by-128 mel-spectrograms with three repeated channels.
Training and prediction share the same audio loading, spectrogram, and padding
code so their inputs stay consistent.
The bundled checkpoint can be loaded for inference, but this repository does
not include a measured held-out accuracy result. Treat its confidence scores as
model outputs, not calibrated probabilities, until evaluated on representative
test audio.

## Repository notes

- `.gitignore` excludes virtual environments, local dataset files, caches, and
  generated preprocessing output.
- No project license is included. Add a license before redistributing the code
  or model if you want to grant reuse rights.