# Sara Brain Project Instructions

## Model Training Standards

### Sara Extractor
- **Production Architecture**: 115M parameters.
  - `d_model`: 768
  - `enc_layers`: 8
  - `dec_layers`: 6
  - `n_heads`: 12
- **Training Expectation**: Approximately 2 hours on an RTX 3070 for 50k steps.
- **Mandate**: NEVER use the smaller 6.9M test architecture for anything other than local syntax smoke tests. All "clean" or "production" retrains MUST use the 115M configuration.

## Workspace Conventions
- **Tool Use**: Always prioritize `run_shell_command` for background training.
- **Verification**: Always verify `nvidia-smi` and log output (Params count) immediately after launching a training session.
