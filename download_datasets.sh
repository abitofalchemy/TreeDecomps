#!/bin/bash
echo "Download the following datasets"
cat tdec/maindatasets | parallel basename {}

echo ""
echo "Get these from dsg1:/data/saguinag/datasets or download them from http://konect.uni-koblenz.de/networks/"
