# Blog

## Running locally

Start a Jekyll container (on host):

```sh
./docker.sh
```

Then in the container:

```sh
make install
make serve
```

The site is served at http://localhost:4000. Use `make serve-drafts` instead to include drafts.

The Gemfile uses the `github-pages` gem, so the local build uses the same gem versions as the live GitHub Pages build.
