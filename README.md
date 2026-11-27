# Blog

The local build uses the `github-pages` gem, so it matches the live GitHub Pages build.

## Running locally

```sh
make install
make serve
```

Then open http://localhost:4000. Use `make serve-drafts` to include drafts.

## Running in Docker

On host:

```sh
./docker.sh
```

Then in container:

```sh
make install
make serve
```
