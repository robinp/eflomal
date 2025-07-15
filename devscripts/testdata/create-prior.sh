#!/bin/env sh
eflomal-align -i my-align.txt -f my-align.fwd -r my-align.rev
eflomal-makepriors -i my-align.txt  -f my-align.fwd -r my-align.rev -p my-align.pri
