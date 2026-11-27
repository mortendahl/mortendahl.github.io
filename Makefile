# Read files as UTF-8 even when the shell has no UTF-8 locale (e.g. in containers)
export RUBYOPT += -Eutf-8

.PHONY: serve
serve:
	bundle exec jekyll serve \
		--host 0.0.0.0 \
		--incremental

.PHONY: serve-drafts
serve-drafts:
	bundle exec jekyll serve \
		--host 0.0.0.0 \
		--incremental \
		--drafts

# Build the site as it would be published: math is rendered into the pages (see README)
.PHONY: build
build:
	@command -v node >/dev/null || { echo "make build needs Node.js (node) to render the math"; exit 1; }
	JEKYLL_ENV=production bundle exec jekyll build
	node _scripts/prerender-math.js _site

# Serve the result of `make build` as is, without rebuilding
.PHONY: preview
preview: build
	bundle exec jekyll serve \
		--host 0.0.0.0 \
		--skip-initial-build \
		--no-watch

.PHONY: install
install:
	bundle install
	npm install

.PHONY: clean
clean:
	rm -rf _site
