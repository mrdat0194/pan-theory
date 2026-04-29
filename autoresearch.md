# Goal
Maximize the `accuracy` of the Audio-JEPA model on the IEMOCAP dataset (few-shot learning task).

# Optimization Target
Higher `accuracy` is better. The metric is expressed as a percentage. 

# Files in Scope
- `MLModel/AIModel/run/main_audio_jepa.py`
- `MLModel/AIModel/model/jepa_backbone.py`

# Context
We are trying to achieve or beat performance parity with the legacy Wav2Vec 2.0 pipeline, which achieves ~79.41% accuracy. 
Note: The execution script (`autoresearch.sh`) runs a faster "proxy" pipeline using fewer epochs (`--epochs 10`) to speed up iteration times. Keep this in mind, as the absolute accuracy numbers will be lower than a full 40-epoch run, but improvements here should transfer.

# Ideas to Try
- Adjust hyperparameters: learning rates (backbone vs. head), batch size, and latent dimensions.
- Feature extraction: Experiment with different `N_MFCC` values, max frames, or normalization techniques.
- Architecture: Tweak the JEPA encoder parameters in `jepa_backbone.py` or modify the pooling strategy before the linear head.
- Augmentation: Add simple data augmentations during the unsupervised pre-training phase.
