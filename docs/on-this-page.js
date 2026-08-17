(() => {
    "use strict";

    const links = Array.from(
        document.querySelectorAll('.page-toc a[href^="#"], .page-toc-mobile a[href^="#"]')
    );
    if (!links.length) return;

    const linksById = new Map();
    for (const link of links) {
        const id = decodeURIComponent(link.hash.slice(1));
        if (!linksById.has(id)) linksById.set(id, []);
        linksById.get(id).push(link);
    }

    const headings = Array.from(linksById.keys())
        .map((id) => document.getElementById(id))
        .filter(Boolean);
    if (!headings.length) return;

    const activate = (id) => {
        for (const link of links) {
            const active = decodeURIComponent(link.hash.slice(1)) === id;
            link.classList.toggle("active", active);
            if (active) link.setAttribute("aria-current", "location");
            else link.removeAttribute("aria-current");
        }
    };

    const updateFromScroll = () => {
        const threshold = 140;
        let current = headings[0];
        for (const heading of headings) {
            if (heading.getBoundingClientRect().top <= threshold) current = heading;
            else break;
        }
        activate(current.id);
    };

    for (const link of links) {
        link.addEventListener("click", (event) => {
            const id = decodeURIComponent(link.hash.slice(1));
            const target = document.getElementById(id);
            if (!target) return;
            event.preventDefault();
            const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
            target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
            history.pushState(null, "", `#${encodeURIComponent(id)}`);
            target.setAttribute("tabindex", "-1");
            target.focus({ preventScroll: true });
            activate(id);
        });
    }

    if ("IntersectionObserver" in window) {
        const visible = new Map();
        const observer = new IntersectionObserver((entries) => {
            for (const entry of entries) visible.set(entry.target.id, entry.isIntersecting);
            const firstVisible = headings.find((heading) => visible.get(heading.id));
            if (firstVisible) activate(firstVisible.id);
            else updateFromScroll();
        }, { rootMargin: "-15% 0px -70% 0px", threshold: [0, 1] });
        headings.forEach((heading) => observer.observe(heading));
    }

    window.addEventListener("scroll", updateFromScroll, { passive: true });
    updateFromScroll();
})();
