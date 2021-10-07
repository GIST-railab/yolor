#!/bin/bash

WEIGHTS=yolor_p6.pt
CFG=cfg/yolor_p6.cfg
SIZE=1280

SOURCE_FOLDER=/HDD/accident_anticipation/Data/DoTA_P650N700/Test_Frames/*

for VID in $SOURCE_FOLDER
do
  OUTPUT=$(echo $VID | sed 's/Test_Frames/Test_Bbox/g')
  echo $OUTPUT
  python detect_1005.py --weights $WEIGHTS --source $VID --output $OUTPUT --cfg $CFG --img-size $SIZE --save-txt
done
