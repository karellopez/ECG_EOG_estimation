# ECG_EOG_estimation

## MEGNet-based ICA classification (optional)

The ECG/EOG ICA selection utilities can optionally use a MEGNet model to score ICs. To enable
this path, set `ICA_UNSUP_MODE` to `"megnet"` (MEGNet only) or `"hybrid"` (weighted MEGNet +
heuristics), and point `MEGNET_MODEL_PATH` to a TensorFlow/Keras model on disk. The scripts will
resample each IC to `MEGNET_INPUT_SAMPLES` and use `MEGNET_OUTPUT_INDEX` from the model output to
derive the score. TensorFlow is required only when MEGNet mode is enabled. 
