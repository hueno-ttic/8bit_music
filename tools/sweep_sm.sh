#!/bin/sh
# 使い方: tools/sweep_sm.sh "<引数>" → 6 曲の合計を出す
cd /Users/super_reader/8bit_music
for s in greenlights aisarenakutemo sand_planet 39music hand_in_hand bless_mv; do
  .venv/bin/python tools/eval_sm.py $s --no_baseline $1 2>&1 | grep score_model | sed "s/^/$s /"
done
