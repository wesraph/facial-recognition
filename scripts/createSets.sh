#!/bin/bash
set -e

echo "Creating positive probes"
mkdir -p data/dataset1/{positive,negative}

echo $(pwd)

IFS=$'\n'
i=1
find  data/dataset1/images -type f | shuf | while read -r f
do
  mv "$f" data/dataset1/positive
  i=$((i + 1))
  [ $i -gt 100  ] && break
done

echo "Creating negative probes"

i=1
find  data/dataset2/images -type f | shuf | while read -r f
do
  mv "$f" data/dataset1/negative
  i=$((i + 1))
  [ $i -gt 100  ] && break
done
