#!/bin/sh

ncall=1000
itmx=10
nblk=256

echo run: gVegasSingle ${ncall} ${itmx} ${nblk}

gVegasSingle -n=${ncall} -i=${itmx} -b=${nblk}

