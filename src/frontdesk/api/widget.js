/* The site's chat (frontdesk, design of 2026-11-07). One file for every business: the name, the
   colour and the words come from GET /web/config. Plain text only: no reply is ever parsed as HTML.
   <script src="https://<service>/web/widget.js" data-lang="sq" defer></script> */
(function () {
  "use strict";
  var script = document.currentScript;
  if (!script || window.frontdeskWidget) return;
  window.frontdeskWidget = true;
  var base = script.src.replace(/\/web\/widget\.js(\?.*)?$/, "");
  var POLL_MS = 1500, POLL_FOR_MS = 60000, MAX_CHARS = 500, RETRIES = 2;
  var SESSION_KEY = "frontdesk-web-session", CURSOR_KEY = "frontdesk-web-cursor";
  var PHONE = /(\+|\b0)\d[\d ]{5,}\d/g;  // +355 69 ... or 069 ...: not a price
  var memory = {};

  function stored(key, make) {
    try {
      var value = sessionStorage.getItem(key);
      if (!value && make) { value = make(); sessionStorage.setItem(key, value); }
      return value;
    } catch (e) {
      if (!memory[key] && make) memory[key] = make();
      return memory[key];
    }
  }
  function store(key, value) {
    try { sessionStorage.setItem(key, value); } catch (e) { memory[key] = value; }
  }
  function newId() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    var bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return Array.prototype.map.call(bytes, function (b) { return ("0" + b.toString(16)).slice(-2); }).join("");
  }
  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (name) { node.setAttribute(name, attrs[name]); });
    if (text) node.textContent = text;
    return node;
  }
  function pickLanguage(config) {
    var wanted = [script.getAttribute("data-lang"), (document.documentElement.lang || "").slice(0, 2)];
    for (var i = 0; i < wanted.length; i++) {
      var lang = (wanted[i] || "").toLowerCase();
      if (config.languages.indexOf(lang) >= 0) return lang;
    }
    return config.default_language;
  }

  var CSS = [
    ":host{all:initial}",
    "*{box-sizing:border-box;font-family:system-ui,-apple-system,'Segoe UI',sans-serif}",
    ".launch{position:fixed;right:20px;bottom:20px;z-index:2147483000;border:0;border-radius:28px;padding:14px 20px;background:var(--fd);color:#fff;font-size:16px;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.25)}",
    ".panel{position:fixed;right:20px;bottom:20px;z-index:2147483001;width:min(380px,calc(100vw - 32px));height:min(560px,calc(100vh - 40px));display:flex;flex-direction:column;background:#fff;color:#1b1b1b;border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,.3);overflow:hidden}",
    ".panel[hidden],.launch[hidden]{display:none}",
    "header{display:flex;align-items:center;justify-content:space-between;padding:12px 14px;background:var(--fd);color:#fff}",
    "h2{margin:0;font-size:16px;font-weight:600}",
    "header button{border:0;background:transparent;color:#fff;font-size:22px;line-height:1;cursor:pointer;padding:4px 8px}",
    ".person{display:block;margin:8px 12px 0;padding:8px;text-align:center;border:1px solid var(--fd);border-radius:8px;color:var(--fd);text-decoration:none;font-size:14px}",
    ".log{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px;font-size:15px;line-height:1.4}",
    ".msg{max-width:85%;padding:8px 11px;border-radius:12px;white-space:pre-wrap;overflow-wrap:anywhere}",
    ".me{align-self:flex-end;background:var(--fd);color:#fff}",
    ".bot{align-self:flex-start;background:#f0f1f3}",
    ".bot a{color:inherit}",
    ".wait{align-self:flex-start;color:#666;font-size:14px}",
    ".notice{margin:0;padding:8px 12px;font-size:12px;color:#444;background:#fafafa;border-top:1px solid #e3e3e3}",
    ".notice a{color:#1b1b1b}",
    "form{display:flex;gap:8px;padding:10px 12px;border-top:1px solid #e3e3e3}",
    "textarea{flex:1;resize:none;height:44px;padding:10px;border:1px solid #bbb;border-radius:8px;font-size:15px;color:#1b1b1b;background:#fff}",
    "form button{border:0;border-radius:8px;padding:0 14px;background:var(--fd);color:#fff;font-size:15px;cursor:pointer}",
    "button:focus-visible,a:focus-visible,textarea:focus-visible{outline:3px solid #ffbf47;outline-offset:2px}",
    ".hidden{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}"
  ].join("");

  function build(config) {
    var lang = pickLanguage(config), words = config.texts[lang], handoff = config.handoff[lang];
    var host = el("div", { "data-frontdesk": "" });
    var root = host.attachShadow({ mode: "open" });
    host.style.setProperty("--fd", config.colour);
    var style = el("style");
    style.textContent = CSS;
    var launch = el("button", { "class": "launch", type: "button", "aria-expanded": "false", "aria-controls": "fd-panel" }, words.open);
    var panel = el("section", { "class": "panel", id: "fd-panel", role: "dialog", "aria-label": words.title, hidden: "" });
    var header = el("header");
    var close = el("button", { type: "button", "aria-label": words.close }, "×");
    header.append(el("h2", {}, words.title), close);
    var person = el("a", { "class": "person", href: config.whatsapp ? "https://wa.me/" + config.whatsapp : "tel:" + config.phone, target: "_blank", rel: "noopener" }, words.person);
    var log = el("div", { "class": "log", role: "log", "aria-live": "polite", tabindex: "0" });
    var notice = el("p", { "class": "notice" }, words.notice + " ");
    if (config.privacy_url) notice.append(el("a", { href: config.privacy_url, target: "_blank", rel: "noopener" }, words.privacy));
    var form = el("form");
    var label = el("label", { "for": "fd-input", "class": "hidden" }, words.placeholder);
    var input = el("textarea", { id: "fd-input", maxlength: String(MAX_CHARS), placeholder: words.placeholder, rows: "1" });
    form.append(label, input, el("button", { type: "submit" }, words.send));
    panel.append(header, person, log, notice, form);
    root.append(style, launch, panel);
    document.body.appendChild(host);

    var session = stored(SESSION_KEY, newId), polling = null, deadline = 0, heard = false, waitNode = null;

    function show(open) {
      panel.hidden = !open;
      launch.hidden = open;
      launch.setAttribute("aria-expanded", String(open));
      (open ? input : launch).focus();
    }
    function add(who, text) {
      var bubble = el("div", { "class": "msg " + who });
      var last = 0;
      text.replace(PHONE, function (number, at) {
        bubble.append(document.createTextNode(text.slice(last, at)));
        bubble.append(el("a", { href: "tel:" + number.replace(/ /g, "") }, number));
        last = at + number.length;
        return number;
      });
      bubble.append(document.createTextNode(text.slice(last)));
      log.insertBefore(bubble, waitNode);
      log.scrollTop = log.scrollHeight;
    }
    function waiting(on) {
      if (on && !waitNode) { waitNode = el("div", { "class": "wait", "aria-hidden": "true" }, "…"); log.append(waitNode); }
      if (!on && waitNode) { waitNode.remove(); waitNode = null; }
    }
    function stop(fallback) {
      clearTimeout(polling);
      polling = null;
      waiting(false);
      if (fallback) add("bot", handoff);
    }
    function poll() {
      var url = base + "/web/replies?session=" + encodeURIComponent(session) + "&after=" + (stored(CURSOR_KEY) || "0");
      fetch(url, { cache: "no-store" }).then(function (response) {
        if (!response.ok) throw new Error(String(response.status));
        return response.json();
      }).then(function (data) {
        data.replies.forEach(function (reply) { waiting(false); add("bot", reply.text); store(CURSOR_KEY, reply.cursor); heard = true; });
        if (!data.waiting && heard) return stop(false);
        if (!data.waiting || Date.now() > deadline) return stop(!heard);
        polling = setTimeout(poll, POLL_MS);
      }).catch(function () {
        if (Date.now() > deadline) return stop(!heard);
        polling = setTimeout(poll, POLL_MS);
      });
    }
    function post(text, id, attempt) {
      fetch(base + "/web/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: session, id: id, text: text })
      }).then(function (response) {
        if (response.status === 202) {
          heard = false;
          deadline = Date.now() + POLL_FOR_MS;
          if (!polling) polling = setTimeout(poll, POLL_MS);
          return;
        }
        if (response.status >= 500 && attempt < RETRIES) return setTimeout(function () { post(text, id, attempt + 1); }, 2000);
        stop(true);
      }).catch(function () {
        if (attempt < RETRIES) return setTimeout(function () { post(text, id, attempt + 1); }, 2000);
        stop(true);
      });
    }
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var text = input.value.trim().slice(0, MAX_CHARS);
      if (!text) return;
      input.value = "";
      add("me", text);
      waiting(true);
      post(text, newId(), 0);
    });
    input.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
    });
    panel.addEventListener("keydown", function (event) { if (event.key === "Escape") show(false); });
    launch.addEventListener("click", function () { show(true); });
    close.addEventListener("click", function () { show(false); });
  }

  fetch(base + "/web/config").then(function (response) {
    if (!response.ok) throw new Error(String(response.status));
    return response.json();
  }).then(function (config) {
    if (document.body) build(config);
    else document.addEventListener("DOMContentLoaded", function () { build(config); });
  }).catch(function () { /* no chat on this page: the site works as before */ });
})();
