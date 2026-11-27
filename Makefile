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

.PHONY: install
install:
	bundle install

.PHONY: clean
clean:
	rm -rf _site
