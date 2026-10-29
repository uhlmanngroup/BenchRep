(() => {
    let drag = null;

    function finishDrag(event) {
        if (!drag || event.pointerId !== drag.pointerId) return;

        const { viewport, pointerId } = drag;
        drag = null;
        viewport.classList.remove("br-graph-dragging");

        if (viewport.hasPointerCapture(pointerId)) {
            viewport.releasePointerCapture(pointerId);
        }
    }

    document.addEventListener("pointerdown", (event) => {
        // Touchscreens retain their native scrolling.
        if (event.pointerType !== "mouse" || event.button !== 0 || drag) {
            return;
        }

        const viewport = event.target.closest(".br-graph-viewport");
        if (!viewport) return;

        // Leave native scrollbar interactions alone.
        const bounds = viewport.getBoundingClientRect();
        const x = event.clientX - bounds.left - viewport.clientLeft;
        const y = event.clientY - bounds.top - viewport.clientTop;
        if (
            x < 0 || y < 0 ||
            x >= viewport.clientWidth ||
            y >= viewport.clientHeight
        ) {
            return;
        }

        if (
            viewport.scrollWidth <= viewport.clientWidth &&
            viewport.scrollHeight <= viewport.clientHeight
        ) {
            return;
        }

        drag = {
            viewport,
            pointerId: event.pointerId,
            x: event.clientX,
            y: event.clientY,
            left: viewport.scrollLeft,
            top: viewport.scrollTop,
        };

        viewport.setPointerCapture(event.pointerId);
        viewport.classList.add("br-graph-dragging");
        viewport.focus({ preventScroll: true });
        event.preventDefault();
    });

    document.addEventListener("pointermove", (event) => {
        if (!drag || event.pointerId !== drag.pointerId) return;

        drag.viewport.scrollLeft = drag.left - (event.clientX - drag.x);
        drag.viewport.scrollTop = drag.top - (event.clientY - drag.y);
    });

    document.addEventListener("pointerup", finishDrag);
    document.addEventListener("pointercancel", finishDrag);
    document.addEventListener("lostpointercapture", finishDrag, true);
})();