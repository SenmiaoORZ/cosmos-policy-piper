# Raw Piper sponge dataset

The full canonical20 raw collection is stored separately in
[`SenmiaoORZ/piper-sponge-canonical20-data`](https://github.com/SenmiaoORZ/piper-sponge-canonical20-data)
through Git LFS. Keeping it outside this public fork is required because
GitHub does not allow public forks to upload new LFS objects.

```bash
git clone https://github.com/SenmiaoORZ/piper-sponge-canonical20-data.git
cd piper-sponge-canonical20-data
git lfs pull
mkdir -p ../datasets
tar --use-compress-program=unzstd -xf raw/piper_sponge_canonical20_20260831.tar.zst -C ../datasets
```

The extracted folder is the `--source` input for
`cosmos_policy.experiments.robot.piper.prepare_piper_dataset`.
