# SPEC/camcalib upstream component

This directory contains the SPEC/camcalib source distributed with the TRAM
checkout. The source-file headers and Max-Planck license terms are preserved
verbatim. These files are not relicensed under BeyondRetarget Plus's CC BY-NC
license; use them only under a valid upstream license agreement.

The trained checkpoint is intentionally kept external because it is large and
subject to the same upstream terms. `scripts/download_models.sh` intentionally
does not fetch it. After accepting the upstream terms, obtain
`camcalib_sa_biased_l2.ckpt` from the
[official TRAM resource](https://drive.google.com/file/d/1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO/view?usp=sharing)
and place a legally obtained copy in `data/pretrain/` before running the
moving-camera frontend.
