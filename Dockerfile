# The jekyll/jekyll image plus npm, which `make install` uses to install the Node packages
# for rendering math at build time and for rebuilding the theme's CSS. The image already
# has Node itself. Built and started by docker.sh.
FROM jekyll/jekyll

RUN apt-get update \
  && apt-get install -y --no-install-recommends npm \
  && rm -rf /var/lib/apt/lists/*
