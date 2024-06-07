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
