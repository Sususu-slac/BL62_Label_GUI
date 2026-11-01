# Theme release validation — 2026-11-01

All four Windows GUIs passed real desktop Light/Dark switching and preference
persistence checks. Light uses blue accents; Dark retains each GUI's original
green accents. The button names its destination: Switch to Light / Switch to Dark.

Existing regression suites passed: Label 17, Fine-tune 36, MomoBot 78, and
Reconstruction theme checks 6 (137 total). Additional native GUI checks preserved
image pixels, painted/erased masks including holes, histogram control points,
training values/logs, conversations and unsent drafts across repeated switches.
The real MomoBot language-model server processes remained unchanged during theme
switching. No new scientific training or full-volume reconstruction was run for
this presentation change.

Reconstruction 1.2.0 and Fine-tune 1.1.0 packaged startup checks passed. Fine-tune
also passed its packaged dataset export/reopen and lossless-mask roundtrip.
External scientific environments, checkpoints and model files are required for
GPU work; they are not included in source archives or GUI installers.
