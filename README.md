To start training the model, This code is only working for single coil right now. runs:

```bash
python train_pix2pix_demo.py
```

If you run with no arguments, the script will create a `fastmri_dirs.yaml` file
in the root directory that you can use to store paths for your system. You can
also pass options at the command line:

```bash
python train_pix2pix_demo.py \
    --challenge CHALLENGE \
    --data_path DATA \
    --mask_type MASK_TYPE
```

where `CHALLENGE` is `singlecoil` and `MASK_TYPE` is
either `random` (for knee) or `equispaced` (for brain). Training logs and
checkpoints are saved in the current working directory by default.

To run the model on test data:

```bash
python train_pix2pix_demo.py \
    --mode test \
    --test_split TESTSPLIT \
    --challenge CHALLENGE \
    --data_path DATA \
    --resume_from_checkpoint MODEL
```

where `MODEL` is the path to the model checkpoint.`TESTSPLIT` should specify
the test split you want to run on - either `test` or `challenge`.

The outputs will be saved to `reconstructions` directory which can be uploaded
for submission.
