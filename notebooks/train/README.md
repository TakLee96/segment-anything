## The Game Plan

1. (Done) Study again how SAM pre/post processing works, especially for our mask gt
2. (Done) Study how SAM PromptDecoder works; try to make use of their 
    - remember to experiment with num_workers and prefetch factor --> takes forever to initialize
    - maybe experiment with not learning background (0) or padding (-1) --> tested in v3
    - maybe experiment with learning rate schedule (warm-up + decay) --> tested in v3
3. Write model prediction code, in a loop, in a batch
4. If you have time, write a new predict.ipynb for zero-shot method
