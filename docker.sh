#!/bin/sh

# Build the image from the Dockerfile (jekyll/jekyll plus npm). Docker caches it, so this is
# only slow the first time. No build context is sent, since the Dockerfile copies no files.
docker build \
  --platform linux/amd64 \
  --tag mortendahl-blog \
  - < Dockerfile || exit 1

docker run \
  -it \
  --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  --publish 4000:4000 \
  --platform linux/amd64 \
  mortendahl-blog \
  bash
