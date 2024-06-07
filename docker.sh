#!/bin/sh

docker run \
  -it \
  --rm \
  --volume="$PWD:/srv/jekyll:Z" \
  --publish 4000:4000 \
  --platform linux/amd64 \
  jekyll/jekyll \
  bash
