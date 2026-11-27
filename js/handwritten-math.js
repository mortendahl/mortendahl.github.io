/*
 * Handwritten figures written in TeX.
 *
 * Inside an element with class "hw" (see css/handwritten.css), every symbol of the math is drawn as
 * text in the surrounding handwriting font, while MathJax still does the layout: superscripts,
 * subscripts, aligned "=" signs and so on. Colors from \class{math-...}{...} are kept.
 *
 * This is a MathJax TeX post-filter: it swaps each identifier, number and operator in the parsed
 * formula for a text node. Used by the browser during `make serve` (see _includes/mathjax.html) and
 * by _scripts/prerender-math.js during `make build`.
 */
(function (root) {
  // operators get a class so css/handwritten.css can space them like the drawings
  var OPERATOR_CLASS = {
    '=': 'hw-rel', '≡': 'hw-rel',
    '+': 'hw-bin', '−': 'hw-bin', '-': 'hw-bin', '⋅': 'hw-bin', '·': 'hw-bin',
    ',': 'hw-punct'
  };
  // the handwriting font has the middle dot but not the math dot operator that \cdot produces
  var REPLACE = { '⋅': '·' };

  function inFigure(node, adaptor) {
    for (var n = node; n; n = adaptor.parent(n)) {
      // only elements have classes: skip text and comment nodes ("#text", "#comment") and the
      // document itself (its kind is "" in the browser)
      var kind = adaptor.kind(n);
      if (kind && kind.charAt(0) !== '#' && adaptor.hasClass(n, 'hw')) return true;
    }
    return false;
  }

  function filter(args) {
    var math = args.math, data = args.data, adaptor = args.document.adaptor;
    if (!math.start || !math.start.node || !inFigure(math.start.node, adaptor)) return;
    var tokens = [];
    data.root.walkTree(function (node) {
      if (node.isKind('mi') || node.isKind('mn') || node.isKind('mo')) tokens.push(node);
    });
    tokens.forEach(function (token) {
      var text = token.getText();
      if (!text) return;
      var mtext = data.nodeFactory.create('token', 'mtext', {}, REPLACE[text] || text);
      // keep the symbol's own attributes, such as the color class that \class{math-green}{x} puts
      // on a single symbol, but not its font variant, so it's drawn in the handwriting font
      var attributes = token.attributes.getAllAttributes();
      Object.keys(attributes).forEach(function (name) {
        if (name !== 'mathvariant') mtext.attributes.set(name, attributes[name]);
      });
      var classes = [attributes['class'], token.isKind('mo') && OPERATOR_CLASS[text]].filter(Boolean);
      if (classes.length) mtext.attributes.set('class', classes.join(' '));
      token.parent.replaceChild(mtext, token);
    });
  }

  // Run before MathJax's own clean-up filters (priorities -6 to -1), which among other things mark each
  // node with its script level: that's what makes the new text in superscripts and subscripts smaller.
  var api = {
    filter: filter,
    priority: -10,
    inFigure: inFigure,
    // x-height of the handwriting font as a fraction of its size, as the browser measures it. MathJax
    // scales math to the x-height of the font around it; at build time it can't measure, so
    // _scripts/prerender-math.js uses this value for math in handwritten figures.
    X_HEIGHT: 0.365
  };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.HandwrittenMath = api;
})(this);
