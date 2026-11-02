/* 3D leaf and card motion for the Streamlit page (presentation only).
 *
 * Mount points are plain HTML written by src/ui_components.py: a div.leaf3d with
 * data-leaf3d="hero" or "scan" that contains an img.leaf3d-poster. The poster image is the fallback.
 * (No tag-like text in this file: Streamlit's sanitiser drops any script that contains some.) When the browser can animate, three.js and
 * static/3d/leaf.glb are loaded after the page has loaded and a WebGL leaf replaces the
 * poster. Nothing loads, and the poster stays, when the user prefers reduced motion,
 * WebGL is unavailable, Save-Data is on or the device looks low-powered.
 *
 * Streamlit re-runs this script on reruns, so everything is set up once on window.
 */
(function () {
    if (window.__leafMotion) {
        window.__leafMotion.scanDom();
        return;
    }

    var THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/+esm";
    var GLTF_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js/+esm";
    var asset = function (name) { return new URL("app/static/3d/" + name, document.baseURI).href; };

    var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    var finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
    var nav = navigator;
    var lowPower = Boolean(nav.connection && nav.connection.saveData) ||
        (nav.deviceMemory !== undefined && nav.deviceMemory <= 2) ||
        (nav.hardwareConcurrency !== undefined && nav.hardwareConcurrency <= 2);
    var webgl = (function () {
        try {
            var canvas = document.createElement("canvas");
            return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
        } catch (error) {
            return false;
        }
    })();

    /* ---------- Dark mode by default ----------
     * Streamlit keeps the visitor's Light / Dark / System choice in localStorage under
     * "stActiveTheme-(page path)-v2" (Streamlit 1.66) and otherwise follows the device. On a first visit
     * (Streamlit itself saves "System" as soon as the page loads), save "Dark" and reload once so
     * Streamlit starts in dark. This happens only once per browser: a later choice in Settings,
     * including System, is never overridden. */
    try {
        var themeKey = "stActiveTheme-" + window.location.pathname + "-v2";
        var saved = localStorage.getItem(themeKey);
        if (!localStorage.getItem("leafAdvisorDarkDefault") && saved !== '"Light"' && saved !== '"Dark"') {
            localStorage.setItem("leafAdvisorDarkDefault", "1");
            localStorage.setItem(themeKey, JSON.stringify("Dark"));
            if (!window.matchMedia("(prefers-color-scheme: dark)").matches) window.location.reload();
        }
    } catch (error) { /* storage blocked: Streamlit follows the device setting */ }

    /* ---------- Follow the active Streamlit theme ----------
     * Custom cards switch colours with html.theme-dark / html.theme-light, read from the background
     * Streamlit paints, so they change the moment the visitor switches theme. */
    function syncTheme() {
        var app = document.querySelector('[data-testid="stApp"]');
        if (!app) return;
        var rgb = getComputedStyle(app).backgroundColor.match(/\d+/g);
        if (!rgb) return;
        var dark = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255 < 0.5;
        document.documentElement.classList.toggle("theme-dark", dark);
        document.documentElement.classList.toggle("theme-light", !dark);
    }
    syncTheme();
    setInterval(syncTheme, 700);

    /* Phone app bar: the light / dark button saves the opposite of the current theme the way
     * Streamlit's Settings menu does, then reloads so Streamlit's own widgets switch too. */
    document.addEventListener("click", function (event) {
        var button = event.target.closest ? event.target.closest(".m-theme") : null;
        if (!button) return;
        var dark = document.documentElement.classList.contains("theme-dark");
        try {
            localStorage.setItem("leafAdvisorDarkDefault", "1");
            localStorage.setItem("stActiveTheme-" + window.location.pathname + "-v2", JSON.stringify(dark ? "Light" : "Dark"));
        } catch (error) { return; }
        window.location.reload();
    });

    /* ---------- Width of the main area, for the background layer ---------- */
    function syncMainWidth() {
        var main = document.querySelector('[data-testid="stMain"]');
        if (main) document.documentElement.style.setProperty("--hud-main-w", main.clientWidth + "px");
    }
    syncMainWidth();
    window.addEventListener("resize", syncMainWidth);
    setInterval(syncMainWidth, 1000);   // the sidebar opening or closing also changes it

    function canAnimate3d() { return webgl && !lowPower && !reducedMotion.matches; }
    function syncRootClass() {
        document.documentElement.classList.toggle("motion-reduced", reducedMotion.matches);
        document.documentElement.classList.toggle("motion-static", !canAnimate3d());
    }
    syncRootClass();

    /* ---------- Lazy loading of three.js and the model (once) ---------- */
    var library = null;
    function loadLibrary() {
        if (!library) {
            library = Promise.all([import(THREE_URL), import(GLTF_URL)]).then(function (mods) {
                var THREE = mods[0];
                return new mods[1].GLTFLoader().loadAsync(asset("leaf.glb")).then(function (gltf) {
                    return { THREE: THREE, gltf: gltf };
                });
            });
            library.catch(function () { library = null; });   // e.g. offline: keep the poster
        }
        return library;
    }

    /* ---------- Pointer position, shared by every scene ---------- */
    var pointer = { x: 0, y: 0 };
    window.addEventListener("pointermove", function (event) {
        pointer.x = event.clientX / window.innerWidth * 2 - 1;
        pointer.y = event.clientY / window.innerHeight * 2 - 1;
    }, { passive: true });

    /* ---------- One WebGL scene per mount point ---------- */
    function buildScene(el, kind, lib) {
        var THREE = lib.THREE;
        var isScan = kind === "scan";
        var canvas = document.createElement("canvas");
        canvas.setAttribute("aria-hidden", "true");
        el.appendChild(canvas);

        var renderer = new THREE.WebGLRenderer({
            canvas: canvas, alpha: true, antialias: true, powerPreference: "low-power"
        });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));

        var scene = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera(32, 1, 0.1, 50);
        scene.add(new THREE.HemisphereLight(0xeaf6ec, 0x2b5a36, 1.9));
        var key = new THREE.DirectionalLight(0xffffff, 2.8);
        key.position.set(2, 3, 4);
        scene.add(key);
        var rim = new THREE.DirectionalLight(0x9be7a8, 1.4);
        rim.position.set(-3, 1, -2);
        scene.add(rim);

        // Leaf: glTF is Y-up with the blade along -Z; stand it up, facing the camera
        var leaf = lib.gltf.scene.clone(true);
        var frame = new THREE.Group();          // leaf orientation; the scan line lives in it too
        frame.add(leaf);
        frame.rotation.set(Math.PI / 2, 0, 0);
        var box = new THREE.Box3().setFromObject(frame);
        var center = box.getCenter(new THREE.Vector3());
        leaf.position.sub(center.clone().applyQuaternion(frame.quaternion.clone().invert()));
        var size = box.getSize(new THREE.Vector3());

        var tilt = new THREE.Group();           // diagonal pose + pointer tilt
        tilt.rotation.z = isScan ? -0.25 : -0.62;
        tilt.add(frame);
        var spin = new THREE.Group();           // idle motion
        spin.add(tilt);
        scene.add(spin);

        var radius = size.length() / 2;
        camera.position.set(0, 0, radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * (isScan ? 1.0 : 0.82));
        camera.lookAt(0, 0, 0);

        var mixer = new THREE.AnimationMixer(leaf);
        lib.gltf.animations.forEach(function (clip) {
            mixer.clipAction(clip).setEffectiveTimeScale(isScan ? 1.6 : 1).play();
        });

        // Scan line: a soft glowing band that sweeps along the blade (a nod to the image model)
        var scanMaterial = new THREE.ShaderMaterial({
            transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
            uniforms: { uStrength: { value: 0 } },
            vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
            fragmentShader: "varying vec2 vUv; uniform float uStrength;" +
                "void main(){ float across = 1.0 - abs(vUv.y * 2.0 - 1.0); float along = smoothstep(0.0, 0.25, vUv.x) * smoothstep(1.0, 0.75, vUv.x);" +
                " float a = pow(across, 3.0) * along * uStrength; gl_FragColor = vec4(vec3(0.62, 1.0, 0.70) * a, a); }"
        });
        var bladeLength = Math.max(size.y, size.x);
        var scanLine = new THREE.Mesh(new THREE.PlaneGeometry(size.x * 1.5, bladeLength * 0.07), scanMaterial);
        scanLine.rotation.x = -Math.PI / 2;     // lies across the blade inside the leaf frame
        frame.add(scanLine);

        // Hero only: a few drifting spores for depth
        var spores = null;
        if (!isScan) {
            var count = 36, positions = new Float32Array(count * 3);
            for (var i = 0; i < count; i++) {
                positions[i * 3] = (Math.random() - 0.5) * radius * 2.4;
                positions[i * 3 + 1] = (Math.random() - 0.5) * radius * 2.4;
                positions[i * 3 + 2] = (Math.random() - 0.5) * radius;
            }
            var geometry = new THREE.BufferGeometry();
            geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
            spores = new THREE.Points(geometry, new THREE.PointsMaterial({
                color: 0xc8f2cf, size: radius * 0.035, transparent: true, opacity: 0.55, depthWrite: false
            }));
            scene.add(spores);
        }

        // Scan only: a turning ring under the leaf
        var ring = null;
        if (isScan) {
            ring = new THREE.Mesh(
                new THREE.RingGeometry(radius * 0.78, radius * 0.8, 96),
                new THREE.MeshBasicMaterial({ color: 0x7ee08f, transparent: true, opacity: 0.55, side: THREE.DoubleSide })
            );
            ring.rotation.x = -1.2;
            scene.add(ring);
        }

        var visible = true, running = true, started = false;
        var clock = new THREE.Clock();
        var smooth = { x: 0, y: 0 };
        var halfLength = bladeLength / 2;

        var io = new IntersectionObserver(function (entries) { visible = entries[0].isIntersecting; });
        io.observe(el);
        var ro = new ResizeObserver(function () {
            var w = el.clientWidth, h = el.clientHeight;
            if (!w || !h) return;
            renderer.setSize(w, h, false);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        });
        ro.observe(el);

        function stop() {
            running = false;
            io.disconnect();
            ro.disconnect();
            mixer.stopAllAction();
            renderer.dispose();
            el.classList.remove("is-live");
            if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
            el.__leaf3d = null;
        }

        function tick() {
            if (!running) return;
            if (!el.isConnected || !canAnimate3d()) { stop(); return; }
            requestAnimationFrame(tick);
            var dt = Math.min(clock.getDelta(), 0.1);
            if (!visible || document.hidden) return;
            var t = clock.elapsedTime;

            mixer.update(dt);
            smooth.x += (pointer.x - smooth.x) * 0.05;
            smooth.y += (pointer.y - smooth.y) * 0.05;

            if (isScan) {
                spin.rotation.y = t * 0.9;
                ring.rotation.z = -t * 0.8;
                ring.material.opacity = 0.35 + 0.2 * Math.sin(t * 3);
            } else {
                spin.rotation.y = Math.sin(t * 0.35) * 0.45 + smooth.x * 0.35;
                spin.rotation.x = smooth.y * 0.2;
                spin.position.y = Math.sin(t * 0.8) * radius * 0.03;
                spores.rotation.y = t * 0.05;
                spores.position.y = (t * radius * 0.02) % (radius * 0.4) - radius * 0.2;
            }

            // Sweep base -> tip; the hero sweeps every 7 s, the scan state continuously
            var period = isScan ? 1.6 : 7.0, sweep = isScan ? 1.6 : 2.2;
            var phase = (t % period) / sweep;
            scanLine.visible = phase <= 1;
            if (phase <= 1) {
                scanLine.position.z = halfLength - phase * bladeLength;   // frame -Z runs base -> tip
                scanLine.position.y = 0.04;
                scanMaterial.uniforms.uStrength.value = Math.sin(phase * Math.PI) * (isScan ? 1.0 : 0.8);
            }

            renderer.render(scene, camera);
            if (!started) {
                started = true;
                el.classList.add("is-live");
            }
        }
        requestAnimationFrame(tick);
        return { stop: stop };
    }

    function mount(el) {
        if (el.__leaf3d || !canAnimate3d()) return;
        el.__leaf3d = "loading";
        loadLibrary().then(function (lib) {
            if (!el.isConnected || !canAnimate3d()) { el.__leaf3d = null; return; }
            try {
                el.__leaf3d = buildScene(el, el.getAttribute("data-leaf3d"), lib);
            } catch (error) {
                el.__leaf3d = null;    // poster stays
            }
        }, function () { el.__leaf3d = null; });
    }

    function scanDom() {
        document.querySelectorAll("[data-leaf3d]").forEach(mount);
    }

    /* ---------- Gentle 3D tilt on cards (mouse / trackpad only) ---------- */
    var TILT_SELECTOR = ".card, .hint-card, .step";
    var tilted = null;
    function resetTilt(card) {
        card.style.removeProperty("--tilt-x");
        card.style.removeProperty("--tilt-y");
        card.classList.remove("is-tilting");
    }
    document.addEventListener("pointermove", function (event) {
        if (!finePointer.matches || reducedMotion.matches) return;
        var card = event.target.closest ? event.target.closest(TILT_SELECTOR) : null;
        if (tilted && tilted !== card) resetTilt(tilted);
        tilted = card;
        if (!card) return;
        var r = card.getBoundingClientRect();
        var x = (event.clientX - r.left) / r.width - 0.5;
        var y = (event.clientY - r.top) / r.height - 0.5;
        var strength = r.height > 260 ? 1.5 : 3;   // tall cards tilt less, so text stays steady
        card.style.setProperty("--tilt-x", (-y * strength).toFixed(2) + "deg");
        card.style.setProperty("--tilt-y", (x * strength).toFixed(2) + "deg");
        card.classList.add("is-tilting");
    }, { passive: true });
    document.addEventListener("pointerleave", function () { if (tilted) resetTilt(tilted); });

    /* ---------- Start after the page has loaded, and watch for new mount points ---------- */
    function start() {
        scanDom();
        new MutationObserver(function (records) {
            for (var i = 0; i < records.length; i++) {
                if (records[i].addedNodes.length) { scanDom(); return; }
            }
        }).observe(document.body, { childList: true, subtree: true });
    }
    reducedMotion.addEventListener("change", function () { syncRootClass(); scanDom(); });

    window.__leafMotion = { scanDom: scanDom };
    var idle = window.requestIdleCallback || function (fn) { return setTimeout(fn, 200); };
    if (document.readyState === "complete") idle(start);
    else window.addEventListener("load", function () { idle(start); });
})();
