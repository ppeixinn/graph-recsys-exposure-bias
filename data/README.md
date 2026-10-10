# Data

Not included in the code zip or the repository. Download into this folder.

## KuaiRand-Pure (about 47 MB archive)
- Source: https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz (project page: https://github.com/chongminggao/KuaiRand)
- Extract so that `data/KuaiRand-Pure/data/log_*.csv` exists.
- The published MD5 in the KuaiRand repository does not match the file from this link, but the archive extracts cleanly and the random-exposure log has the documented 1,186,059 rows.

## KuaiRec (412 MB archive)
- Source: https://zenodo.org/records/18164998/files/KuaiRec.zip (project page: https://kuairec.com)
- Save it as `data/KuaiRec.zip`. Do not extract it: the loader reads `KuaiRec 2.0/data/big_matrix.csv` and `small_matrix.csv` directly from the archive (they are 1.5 GB unpacked).

## Processed datasets
Notebook 02 writes `data/processed/kuairand_seed{0,1,2}.npz` and `kuairec_seed{0,1,2}.npz` (about 2-2.5 MB each). Read one with `data.load_processed("kuairand", seed=0)`. Instead of running notebook 02, a member can copy these files into `data/processed/`; the raw files are then only needed to rerun notebooks 01 and 02.
