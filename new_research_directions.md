- [ ] add skip from raw input (before unshuffle) to output and extra processing layers, for the fine grained representation.

- [ ] patching e.g. 128x128 for training and inference.
- [ ] ensemble of models
- [ ] rotations / test time augmentation for more consistent output

- [x] bigger batch sizes, fix validation set being only 2 categories out of 4


- [ ] implementing transformers here? (or some other way to give more computing power, for free gains)

- [ ] dropout didn't prove too useful yet, I think this is because the training/val/test sets are very similar to each other (coming from the same image set data, this is given by the kaggle challenge).


- [ ] adding extra loss terms (like ergas and SID) for training.



- [x] Model is probably too big for only 180 training images, can probably reduce.



architectures:
- [ ] **HSCCN-R/D**
- [ ] **AWAN**
- [ ] **MST/MST++**
- [ ] **EDSR-style Residual Networks**









