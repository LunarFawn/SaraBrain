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

## Strict Authorization Mandate
- **Wait for "Go"**: NEVER execute a state-modifying command (e.g., `write_file`, `replace`, or a `run_shell_command` that alters files, launches background jobs, or kills processes) in the same turn that you ask for permission. You MUST propose the action, provide the technical rationale, and then STOP and wait for an explicit user response (e.g., "Go", "Do it", "Yes").
- **Process Integrity**: NEVER kill, terminate, or restart a background process unless explicitly instructed to do so by the user. If a process appears hung or inefficient, report the status and wait for direction.
- **Unilateral Reverts**: NEVER revert a code change or undo a previous action (even if it caused an error) unless the user specifically asks you to "revert" or "undo".

## Workspace Conventions
- **Database Persistence**: ALWAYS store ingested brain databases in persistent directories (e.g., `data/` or the project root). NEVER use `/tmp/` for brain storage, as these files are lost on reboot or session reset.
- **Tool Use**: Always prioritize `run_shell_command` for background training.
- **Verification**: Always verify `nvidia-smi` and log output (Params count) immediately after launching a training session.
