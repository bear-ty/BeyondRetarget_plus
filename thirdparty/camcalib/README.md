# User-supplied SPEC/camcalib

This project does not have permission to redistribute this component, so its
source is not included. Obtain `__init__.py`, `model.py`, `cam_utils.py`, and
`resnet.py` from [TRAM's camcalib directory](https://github.com/yufu-wang/tram/tree/main/thirdparty/camcalib)
under the [SPEC terms](https://github.com/mkocabas/SPEC/blob/master/LICENSE).
Keep the original license notices and put those four files directly in this
directory; no path registration or code changes are needed.

Obtain `camcalib_sa_biased_l2.ckpt` separately from the
[official TRAM resource](https://drive.google.com/file/d/1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO/view?usp=sharing)
under its terms and place it at `data/pretrain/camcalib_sa_biased_l2.ckpt`.
Both source and checkpoint are required before generating world-camera results
in any camera mode, including static-camera and EMDB preprocessing.

See [the setup instructions](../../README.md#spec-and-pare-components) for
the frontend installation and checks. User-supplied files are Git-ignored.
