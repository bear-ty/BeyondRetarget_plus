#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# HMR2b checkpoint used to initialize the official VIMO training path.
mkdir -p data/pretrain/hmr2b
gdown --fuzzy -O ./data/pretrain/hmr2b/epoch=35-step=1000000.ckpt https://drive.google.com/file/d/1W4fcp8mwS19Rg_A7MoTS1lc7JafqTGu-/view?usp=sharing
