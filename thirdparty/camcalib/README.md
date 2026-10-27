# SPEC/camcalib upstream component

This directory contains the SPEC/camcalib source distributed with the TRAM
checkout. The source-file headers and Max-Planck license terms are preserved
verbatim. These files are not relicensed under BeyondRetarget Plus's CC BY-NC
license; use them only under a valid upstream license agreement.

The trained checkpoint is intentionally kept external because it is large and
subject to the same upstream terms. Obtain
`camcalib_sa_biased_l2.ckpt` through `scripts/download_models.sh` after
accepting the upstream terms, or place a legally obtained copy in
`data/pretrain/` before running the moving-camera frontend.
