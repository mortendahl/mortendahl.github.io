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

## Math

MathJax is loaded only on pages that ask for it. Add `math: true` to a post's front matter, then write TeX between `$$ ... $$` (inline or as its own paragraph) or `$ ... $` (inline).

MathJax is self-hosted in `js/mathjax/` (version in `js/mathjax/VERSION`) and includes all TeX extensions, so math pages don't depend on any external service.

## Third-party libraries

jQuery and Bootstrap are in `js/` and `css/`. The pages load the minified `.min` files, and the readable files next to them are from the same releases. To patch a library, edit the readable file and point `_includes/head.html` or `_includes/footer.html` at it instead of the `.min` file.

The theme's own CSS and JS are built from `less/clean-blog.less` and `js/clean-blog.js` with `grunt` (after `npm install`). To change styles or behaviour, edit those rather than the library files.
