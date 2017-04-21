#!/bin/bash
find /data/saguinag/datasets -name 'out*' -type f | parallel python netinfo.py {}
