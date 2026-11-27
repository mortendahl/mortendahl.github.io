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

`make install` also runs `npm install`, which needs Node.js. Node is used to render math for publishing (see "Rendering" below) and to rebuild the theme's CSS.

To check the site as it would be published, with math rendered into the pages, run:

```sh
make preview
```

This runs `make build` and serves the result at http://localhost:4000 without rebuilding, so rerun it after changes. Use `make build` alone to only build into `_site/`.

The Gemfile uses the `github-pages` gem, so the local build uses the same gem versions as the live GitHub Pages build.

## Math

MathJax is loaded only on pages that ask for it. Add `math: true` to a post's front matter, then write TeX between `$$ ... $$` (inline or as its own paragraph) or `$ ... $` (inline).

### Rendering

Math is rendered in one of two ways, both using MathJax 3.2.2 with the same settings:

- `make serve` (development): MathJax renders the math in the browser, so changes show up as soon as you save.
- `make build` and `make preview` (production): Jekyll builds with `JEKYLL_ENV=production`, then `_scripts/prerender-math.js` renders the math into each page as SVG, with hidden MathML for screen readers. Readers don't load or run MathJax, and math shows even with JavaScript off. A TeX error, including an undefined macro, fails the build instead of showing up in red on the page.

The TeX settings (delimiters and a post's `math_macros`) are written once into each math page by `_includes/mathjax.html`, and both ways read them from there. The RSS feed is not rendered and still contains the raw TeX.

A post can define its own TeX macros under `math_macros` in its front matter, written without the leading backslash and in single quotes so YAML keeps the backslashes, e.g. `x: '\textcolor{#60BA46}{x}'` lets the post write `\x` for a green x. See the Paillier part 1 post for an example.

### Code or math?

The blog is code-first: use backticks when pointing at the code, and MathJax when stating something about the values. The test is "would I copy this into the Python code?" If yes, use backticks.

1. Backticks for anything from the post's code: functions, variables, attributes and expressions that run as written, e.g. `add_cipher`, `ek.nn`, `c == enc(ek, x, r)`.
2. MathJax for notation Python can't express directly or expresses badly: modular inverses and powers ($$x^{-1}$$, $$r^n$$), sets ($$\mathbb{Z}_n^*$$), magnitudes ($$2^{2048}$$), indexed families ($$r_1, r_2$$), and derivations where `=` means "equals" rather than assignment or `==`.
3. Never mix the two within one expression or statement; pick one form for all of it.
4. When math describes something the code implements, link them once, e.g. "the order of $$n$$, i.e. $$(p - 1)(q - 1)$$ (`order_of_n` in the code)".
5. A name mentioned on its own in the prose keeps the same form throughout the post. Math names use the same letter and case as the code, so a fixed constant `Q` is $$Q$$ while a function argument `q` that is still being searched for is $$q$$. A name with a meaning in the math (such as $$n$$, $$x$$ or $$T$$) is always math, and a name that only exists in the code (such as `ek.nn`, `A_values` or `omega4`) is always code. Code expressions can still contain math names, e.g. `pow(r, ek.n, ek.nn)`, and vice versa.
6. Stick to one form per sentence where possible. If a sentence needs both, move the code part into its own sentence ("In the code this is `A_values[i]`."). A short parenthetical link as in rule 4 is fine.
7. When in doubt, use backticks.
8. Colors only appear in math, so they always mean "the same quantity as in the pictures". Code is never colored.

The difference between the two forms is visible on its own: in code you call `enc(ek, x, r)`, while in math the key is dropped and it's $$\mathsf{enc}(x, r)$$; likewise `*` vs $$\cdot$$, `==` vs $$=$$, and `r1` vs $$r_1$$.

### Self-hosting

MathJax is self-hosted in `js/mathjax/` (version in `js/mathjax/VERSION`) and includes all TeX extensions, so math pages don't depend on any external service.

## Third-party libraries

jQuery and Bootstrap are in `js/` and `css/`. The pages load the minified `.min` files, and the readable files next to them are from the same releases. To patch a library, edit the readable file and point `_includes/head.html` or `_includes/footer.html` at it instead of the `.min` file.

The theme's own CSS and JS are built from `less/clean-blog.less` and `js/clean-blog.js` with `grunt` (after `npm install`). To change styles or behaviour, edit those rather than the library files.
