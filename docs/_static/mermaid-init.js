window.addEventListener("load", function () {
  if (typeof mermaid === "undefined") {
    return;
  }

  mermaid.initialize({ startOnLoad: false, theme: "neutral" });

  document.querySelectorAll("div.highlight-mermaid pre").forEach(function (pre) {
    const source = pre.textContent;
    const wrapper = document.createElement("div");
    wrapper.className = "mermaid";
    wrapper.textContent = source;
    const outer = pre.closest("div.highlight-mermaid");
    if (outer) {
      outer.parentElement.replaceChild(wrapper, outer);
    }
  });

  mermaid.run();
});
