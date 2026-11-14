/* Derive TOC badges from rendered API headings; no per-page annotations. */
(() => {
  const labels = {
    function: ["fn", "Function"],
    method: ["mth", "Method"],
    class: ["cls", "Class"],
    dataclass: ["dat", "Dataclass"],
    schema: ["sch", "Schema"],
    "type-alias": ["typ", "Type alias"],
    property: ["prp", "Property"],
    "typed-dict": ["dct", "Typed dict"],
    protocol: ["prt", "Protocol"],
    constant: ["con", "Constant"],
    attribute: ["atr", "Attribute"],
  };

  function splitIdentifier(text) {
    return text
      .replace(/([a-z0-9])([A-Z])/g, "$1\u200b$2")
      .replace(/([A-Z])([A-Z][a-z])/g, "$1\u200b$2")
      .replace(/(_+)(?=[A-Za-z0-9])/g, "$1\u200b")
      .split("\u200b");
  }

  function addWordBreaks(element) {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const parts = splitIdentifier(node.textContent);
      if (parts.length < 2) continue;
      const fragment = document.createDocumentFragment();
      parts.forEach((part, index) => {
        if (index) fragment.append(document.createElement("wbr"));
        const word = document.createElement("span");
        word.className = "br-toc-word";
        word.textContent = part;
        fragment.append(word);
      });
      node.replaceWith(fragment);
    }
  }

  function decorateReferenceNavigation() {
    document.querySelectorAll(
      '.md-sidebar--secondary .md-nav__link[href^="#"]'
    ).forEach((link) => {
      if (link.classList.contains("br-toc-entry")) return;

      let id;
      try {
        id = decodeURIComponent(link.getAttribute("href").slice(1));
      } catch {
        return;
      }
      const heading = document.getElementById(id);
      if (!heading?.matches(".doc-heading")) {
        // Style organizational headings only on pages with API entries.
        if (
          document.querySelector(".doc-heading > .br-api-badge") &&
          heading?.matches("h2, h3, h4, h5, h6")
        ) {
          link.classList.add(
            heading.matches("h2") ? "br-toc-section" : "br-toc-subsection"
          );
        }
        return;
      }
      const source = heading.querySelector(":scope > .br-api-badge");
      if (!source) return;

      // Typed dicts and protocols also carry a class color fallback, so use
      // the displayed category rather than whichever CSS class comes first.
      const kind = source.textContent.trim().replace(/\s+/g, "-");
      const label = labels[kind];
      if (!label) return;

      const name = document.createElement("span");
      name.className = "br-toc-name";
      while (link.firstChild) name.append(link.firstChild);

      addWordBreaks(name);

      const badge = document.createElement("span");
      badge.className = "br-toc-kind";
      for (const className of source.classList) {
        if (className.startsWith("br-api-badge--")) {
          badge.classList.add(className);
        }
      }
      badge.textContent = label[0];
      badge.title = label[1];
      badge.setAttribute("aria-hidden", "true");

      const accessibleKind = document.createElement("span");
      accessibleKind.className = "md-visually-hidden";
      accessibleKind.textContent = ` (${label[1]})`;
      name.append(accessibleKind);

      link.classList.add("br-toc-entry");
      link.append(badge, name);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", decorateReferenceNavigation);
  } else {
    decorateReferenceNavigation();
  }
  // Also support the theme's instant navigation if enabled later.
  if (typeof document$ !== "undefined") {
    document$.subscribe(decorateReferenceNavigation);
  }
})();
