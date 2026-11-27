# Blog

## Running locally

Start a Jekyll container (on host):

```sh
./docker.sh
```

This builds an image from `Dockerfile` (the `jekyll/jekyll` image plus npm) and starts it. Docker caches the image, so only the first run is slow.

Then in the container:

```sh
make install
make serve
```

The site is served at http://localhost:4000. Use `make serve-drafts` instead to include drafts.

`make install` also runs `npm install` when npm is available. Node is used to render math for publishing (see "Rendering" below) and to rebuild the theme's CSS. Without npm, `make serve` still works, but `make build` and `make preview` don't.

To check the site as it would be published, with math rendered into the pages, run:

```sh
make preview
```

This runs `make build` and serves the result at http://localhost:4000 without rebuilding, so rerun it after changes. Use `make build` alone to only build into `_site/`.

The Gemfile uses the `github-pages` gem, so the local build uses the same gem versions as the live GitHub Pages build.

## Math

MathJax is loaded only on pages that ask for it. Add `math: true` to a post's front matter, then write TeX between `$$ ... $$` (inline or as its own paragraph) or `$ ... $` (inline).

### Rendering

Math is rendered in one of two ways, both using MathJax 3.2.2's CHTML output (HTML and CSS, with MathJax's own web fonts) and the same settings:

- `make serve` (development): MathJax renders the math in the browser, so changes show up as soon as you save. GitHub Pages' own build also works this way.
- `make build` and `make preview` (production): Jekyll builds with `_config.prerender.yml` added, which leaves out the MathJax script, then `_scripts/prerender-math.js` renders the math into each page, with hidden MathML for screen readers. Readers don't load or run MathJax, only its fonts, and math shows even with JavaScript off. A TeX error, including an undefined macro, fails the build instead of showing up in red on the page.

The settings (TeX delimiters, a post's `math_macros`, where MathJax's fonts are, and whether the page has handwritten math) are written once into each math page by `_includes/mathjax.html`, and both ways read them from there. On posts with handwritten figures, both also draw the math in those figures in the handwriting font (see "Handwritten figures"); everything else is plain MathJax. The RSS feed is not rendered and still contains the raw TeX.

A post can define its own TeX macros under `math_macros` in its front matter, written without the leading backslash and in single quotes so YAML keeps the backslashes, e.g. `x: '\class{math-green}{x}'` lets the post write `\x` for a green x. A macro with arguments is a list of the definition and the number of arguments, e.g. `odd: ['\class{math-brown}{#1}', 1]` for `\odd{2x}`. See the Paillier part 1 and Secret Sharing part 2 posts for examples.

### Colors

Math is colored with CSS classes, `\class{math-blue}{...}` in TeX and `class="math-blue"` in handwritten figures, so the shades are defined once in `less/clean-blog.less` and are the same in every post and figure. They're the colors of the original hand-drawn equation images, slightly darkened, and are used by role:

- `math-blue`: fixed public parameters, such as the modulus and generator ($$n$$ and $$g$$ in Paillier, $$Q$$ and $$\omega$$ in Secret Sharing).
- `math-green` and `math-brown`: a first and a second value that the explanation keeps apart, such as plaintexts $$x_1$$ and $$x_2$$, or the even and odd halves $$B$$ and $$C$$ of a polynomial.
- `math-red` and `math-orange`: a first and a second piece of randomness, such as $$r_1$$ and $$r_2$$.
- `math-purple`: a third value, such as the third part $$D$$ of a polynomial.

In a post about several parties, such as secure aggregation, the colors instead tell the parties apart.

Color only what helps the explanation, and give a symbol the same color everywhere in a post.

### Handwritten figures

Derivations and other math shown on its own line are drawn as handwritten figures, in the Indie Flower font and styled like the original drawings. Add `handwritten: true` to the post's front matter. That loads the handwriting font and the styles for diagrams (`css/handwritten.css`) and, unless handwritten math is turned off (see below), the styles and code that draw math in the handwriting font (`css/handwritten-math.css` and `js/handwritten-math.js`). The font is self-hosted in `assets/fonts/indie-flower/` (SIL Open Font License).

Write an equation, derivation, vector or matrix in TeX, as display math inside a `<div class="hw">`, using the post's color macros like the rest of its math:

```html
<div class="hw">
\[
\begin{aligned}
\mathsf{enc}(\xa, \r) \cdot \g^{\xb} &= (\g^{\xa} \cdot \r^{\n}) \cdot \g^{\xb} \\
&= \g^{\xa + \xb} \cdot \r^{\n} \quad = \mathsf{enc}(\xa + \xb, \r)
\end{aligned}
\]
</div>
```

MathJax lays it out, and `js/handwritten-math.js` draws every symbol in the handwriting font instead of MathJax's fonts, with two exceptions: brackets that grow with their contents (matrices, `\left[ ... \right]`) and arrows stay in MathJax's style, since they have to stretch. Exponents stack over subscripts as in typeset math ($$r_1^n$$). Put a note such as "where t = r · s" in `\class{hw-note}{...}` to make it smaller. See Paillier part 1 for derivations.

When a post shows the same values in many figures, define each once as a macro and build the figures from those. Secure Aggregation part 1 does this for its example: `\vect{blue}{s_{01}}{644\\280}` is a labeled blue vector, defined once as `\sab`, so a figure reads `\xa \arr{blue} \saa , \sab , \sac`. See its front matter for the helpers (labeled vectors, colored arrows, weights, bracketed columns).

Keep spacing out of CSS: MathJax sizes brackets from its own layout, so padding added with CSS makes contents stick out of them. Use TeX instead, such as `\\[0.6em]` for more room after a row. Derivations (`\begin{aligned}`) already get extra room between rows.

Diagrams that aren't equations, such as networks, are drawn as inline SVG (`<div class="hw-svg">`). See Secure Aggregation part 1.

MathJax without handwriting is for math inside sentences. Don't also show a derivation as MathJax display math when a figure shows it.

### Turning off handwritten math

To fall back to plain MathJax, set `handwritten_math: false` in `_config.yml`. Figures then render as ordinary MathJax display math, from the same TeX and in the same colors, and no code that relies on MathJax internals runs, in `make serve` or `make build`. Diagrams stay handwritten SVG, and nothing in the posts needs to change.

This works because the handwriting is a separate layer on top of standard TeX. To keep it that way, write figures in standard TeX with `\class{...}` for colors, as now; the only handwriting-specific markup is `\class{hw-note}{...}`, which in plain MathJax just shows at normal size.

Turn it off if a MathJax upgrade breaks the handwritten figures (see "Pinned MathJax version"). Plain MathJax is configured with documented options only, so for a newer 3.x version it keeps working as is. MathJax 4 is a bigger change even without handwriting: it packages its fonts differently, so the self-hosted files in `js/mathjax/` and the font location in `_includes/mathjax.html` need updating, and its Node package (`@mathjax/src` instead of `mathjax-full`) may need changes to `_scripts/prerender-math.js`.

### Code or math?

The blog is code-first: use backticks when pointing at the code, and MathJax when stating something about the values. The test is "would I copy this into the Python code?" If yes, use backticks.

1. Backticks for anything from the post's code: functions, variables, attributes and expressions that run as written, e.g. `add_cipher`, `ek.nn`, `c == enc(ek, x, r)`.
2. MathJax for notation Python can't express directly or expresses badly: modular inverses and powers ($$x^{-1}$$, $$r^n$$), sets ($$\mathbb{Z}_n^*$$), magnitudes ($$2^{2048}$$), indexed families ($$r_1, r_2$$), and derivations where `=` means "equals" rather than assignment or `==`.
3. Never mix the two within one expression or statement; pick one form for all of it.
4. When math describes something the code implements, link them once, e.g. "the order of $$n$$, i.e. $$(p - 1)(q - 1)$$ (`order_of_n` in the code)".
5. A name mentioned on its own in the prose keeps the same form throughout the post. Math names use the same letter and case as the code, so a fixed constant `Q` is $$Q$$ while a function argument `q` that is still being searched for is $$q$$. A name with a meaning in the math (such as $$n$$, $$x$$ or $$T$$) is always math, and a name that only exists in the code (such as `ek.nn`, `A_values` or `omega4`) is always code. Code expressions can still contain math names, e.g. `pow(r, ek.n, ek.nn)`, and vice versa.
6. Stick to one form per sentence where possible. If a sentence needs both, move the code part into its own sentence ("In the code this is `A_values[i]`."). A short parenthetical link as in rule 4 is fine.
7. When in doubt, use backticks.
8. Colors only appear in math and handwritten figures, and follow the roles under "Colors" above. Code is never colored.

The difference between the two forms is visible on its own: in code you call `enc(ek, x, r)`, while in math the key is dropped and it's $$\mathsf{enc}(x, r)$$; likewise `*` vs $$\cdot$$, `==` vs $$=$$, and `r1` vs $$r_1$$.

### Self-hosting

MathJax is self-hosted in `js/mathjax/` (version in `js/mathjax/VERSION`): the browser script `tex-chtml-full.js`, which includes all TeX extensions, and its fonts in `js/mathjax/output/chtml/fonts/woff-v2/`, where the script expects them. The handwriting font is self-hosted too, so math pages don't depend on any external service.

### Pinned MathJax version

MathJax is pinned to 3.2.2, and must be the same version in both places:

- `js/mathjax/`: the browser script and its fonts, used by `make serve` (the version is in `js/mathjax/VERSION`);
- `mathjax-full` in `package.json`: an exact version, with no `^`, used by `make build`.

The handwritten figures rely on MathJax internals that a new version may change. `js/handwritten-math.js` adds a TeX post-filter that rewrites the parsed formula, and overrides the font metrics MathJax measured. `_scripts/prerender-math.js` and `_includes/mathjax.html` remove text and table widths MathJax wrote into the page. So treat an upgrade, in particular to MathJax 4, as a small project rather than a file copy:

1. Copy `tex-chtml-full.js` and `output/chtml/fonts/woff-v2/` from the `es5` folder of the `mathjax-full` npm package at the new version into `js/mathjax/`, and update `js/mathjax/VERSION` and `package.json`.
2. Run `make build`: it fails on TeX errors, including errors from the filter.
3. Look at the handwritten figures in Paillier part 1 and Secure Aggregation part 1 with both `make serve` and `make preview`: text in the handwriting font, brackets that cover their contents, and the same result in both.

If the handwritten figures don't work with the new version and fixing them can wait, set `handwritten_math: false` (see "Turning off handwritten math") and upgrade anyway.

## Third-party libraries

jQuery and Bootstrap are in `js/` and `css/`. The pages load the minified `.min` files, and the readable files next to them are from the same releases. To patch a library, edit the readable file and point `_includes/head.html` or `_includes/footer.html` at it instead of the `.min` file.

The theme's own CSS and JS are built from `less/clean-blog.less` and `js/clean-blog.js` with `grunt` (after `npm install`). To change styles or behaviour, edit those rather than the library files.
