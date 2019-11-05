#!/bin/bash
set -e

echo "Creating positive probes"
mkdir -p dataset1/{positive,negative}

mv data/dataset1/images/*.1.jpg data/dataset1/positive

i=0
for image in dataset1/positive/*:
do
  [ $i -gt 100  ] && {
    rm "$image"
  }
  i=$((i + 1))
done

echo "Creating negative probes"
cp data/dataset2/images/*.1.jpg data/dataset1/negative

i=0
for image in dataset1/negative/*:
do
  [ $i -gt 100  ] && {
    rm "$image"
  }
  i=$((i + 1))
done
