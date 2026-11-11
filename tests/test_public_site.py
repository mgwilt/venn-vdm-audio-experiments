from html.parser import HTMLParser
from pathlib import Path
import json
import posixpath
import re
import subprocess
from urllib.parse import unquote, urlsplit

import pytest

from scripts import build_site


def test_repository_boundary_excludes_experiment_material():
    root = Path(__file__).resolve().parents[1]
    for name in ("design", "results", "logs", "venn/runs", "venn/assets"):
        assert not (root / name).exists(), name
    assert not (root / "examples/precedent_desk/data/tickets.json").exists()
    for directory in ("venn", "scripts", "tests"):
        for path in (root / directory).rglob("*.py"):
            assert not path.stem.startswith("phase"), path
    if (root / ".git").exists():
        tracked = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True).stdout.split()
        forbidden = re.compile(r"\.(safetensors|pt|npz|npy|sqlite\d*|parquet)$|^(design|results|logs)/|/runs/|^examples/precedent_desk/data/")
        assert not [name for name in tracked if forbidden.search(name) and (root / name).exists()]


def test_precedent_desk_renders_ticket_and_model_strings_as_text():
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for browser rendering checks")
    path = Path(__file__).parents[1] / "examples/precedent_desk/static/index.html"
    code = r"""
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";
const source = readFileSync(process.argv[1], "utf8");
const escape = source.slice(source.indexOf("function escapeHTML"), source.indexOf("async function api"));
const render = source.slice(source.indexOf("function showTicket"), source.indexOf("async function routeCurrent"));
const elements = {};
const payload = '<img src=x onerror="alert(1)">';
const context = {
  assert, payload, current: null, $: id => elements[id] ??= {},
  pct: value => String(value), ci: () => "",
  state: {results: {jev: {accuracy: .5, source: payload}}, tau_margin: .1},
};
vm.runInNewContext(escape + render + `
  const showName = name => escapeHTML(name.replace(/_/g, " "));
  showTicket({text: payload, id: payload, label: payload, row: 0});
  const outputs = [
    $("ticket").innerHTML,
    renderJev({available: false, note: payload}),
    renderJev({available: true, choice: payload, model: payload}),
    renderVenn({mode: "queue names only", queue: payload, label_set: payload,
      option_text: payload, confidence: .5, latency_ms: 1, embeddings_this_call: 0,
      request: {text: payload}}),
    renderVenn({queue: payload, confidence: .5, latency_ms: 1, store_rows: 1,
      margin: .2, embeddings_this_call: 0, request: {text: payload},
      precedents: [{text: payload, queue: payload, note: payload}]}),
  ];
  for (const output of outputs) {
    assert(!output.includes("<img"), output);
    assert(output.includes("&lt;img"), output);
  }
`, context);
"""
    subprocess.run([node, "--input-type=module", "-e", code, str(path)], check=True,
                   capture_output=True, text=True)


@pytest.mark.parametrize("local_video_files", [False, True])
@pytest.mark.parametrize("attachment_embeds", [False, True])
def test_site_rewrites_rendered_pages_not_markdown_source_downloads(tmp_path, monkeypatch, local_video_files, attachment_embeds):
    source = tmp_path / "source"

    def embed(stem):
        poster = stem.replace("venn-precedent-desk", "precedent-desk")
        url = (f"https://github.com/user-attachments/assets/{stem}-fixture" if attachment_embeds
               else f"[![Watch](media/{poster}-poster.jpg)]({build_site.MEDIA_RELEASE}/{stem}.mp4)")
        return f"<!-- video: {stem} -->\n\n{url}"

    files = {
        "README.md": "# Venn\n[API](docs/api.md) [Skill](skills/venn/SKILL.md)\n\n"
                                 "[Wine](examples/wine/README.md)\n\n"
                                 "<!-- video: venn-arena -->\n\n"
                                 "https://github.com/user-attachments/assets/arena-fixture\n\n"
                                 "<!-- video: venn-same-memory-review -->\n\n"
                                 "https://github.com/user-attachments/assets/comparison-fixture",
        "docs/api.md": "# API",
        "docs/licenses.md": "# Licenses\n[License](../LICENSE) [Notice](../NOTICE)",
        "LICENSE": "license fixture",
        "NOTICE": "notice fixture",
        "docs/new-guide.md": "# New guide\n\n## New section\n\nNew guidance.",
        "docs/venn-and-jev.md": "# Venn and Jev\n\n" + embed("venn-same-memory-review"),
        "docs/demos.md": "# Demos\n\n" + "\n\n".join(
            embed(stem) for stem in ("venn-arena", "venn-same-memory-review", "venn-precedent-desk")),
        "docs/site.css": "body { margin: 0; }",
        "clients/venn.mjs": "// public client",
        "skills/venn/SKILL.md": "# Skill",
        "docs/media/million-sweep.gif": "generated recording fixture",
        "docs/media/venn-logo.svg": "<svg xmlns=\"http://www.w3.org/2000/svg\"/>",
        "docs/media/venn-arena-poster.jpg": "poster fixture",
        "docs/media/venn-arena.mp4": "gameplay fixture",
        "docs/media/venn-arena-captions.vtt": "WEBVTT\n",
        "docs/same-memory-video.md": "# Same questions. Different memory.\n\n"
            + embed("venn-same-memory-review") + "\n\n" + embed("venn-same-memory-vertical-review"),
        "docs/media/venn-precedent-desk.mp4": "desk fixture",
        "docs/media/precedent-desk-poster.jpg": "desk poster fixture",
        "docs/media/precedent-desk-captions.vtt": "WEBVTT\n",
        "docs/media/venn-same-memory-review.mp4": "landscape fixture",
        "docs/media/venn-same-memory-review-poster.jpg": "poster fixture",
        "docs/media/venn-same-memory-review-captions.vtt": "WEBVTT\n",
        "docs/media/venn-same-memory-vertical-review.mp4": "vertical fixture",
        "docs/media/venn-same-memory-vertical-review-poster.jpg": "vertical poster fixture",
        "docs/media/venn-same-memory-vertical-review-captions.vtt": "WEBVTT\n",
        "docs/media/same-memory/history-eval.json": "{}",
        "examples/same_memory_video/history.ipynb": "{}",
        "examples/same_memory_video/__pycache__/render.pyc": "excluded",
        "examples/wine/README.md": "# Wine",
        "examples/hotdog/README.md": "# Hot dog example\n[UI source](index.html)",
        "examples/hotdog/index.html": "<!doctype html><title>Local app fixture</title>",
        "examples/hotdog/photos/private/README.md": "private photo notes",
        "examples/hotdog/local-state/README.md": "private model notes",
    }
    for name, value in files.items():
        if name.endswith(".mp4") and not local_video_files:
            continue
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    monkeypatch.setattr(build_site, "ROOT", source)
    output = tmp_path / "site"
    build_site.build(output)
    index = (output / "index.html").read_text(encoding="utf-8")
    assert 'href="api.html"' in index
    assert 'href="examples/wine/index.html"' in index
    wine_page = (output / "examples/wine/index.html").read_text(encoding="utf-8")
    assert 'aria-label="Documentation"' in wine_page
    assert 'href="../../api.html"' in wine_page
    assert 'href="skills/venn/SKILL.md"' in index
    assert "SKILL.html" not in index
    assert "More documentation" in index
    assert '<img src="media/venn-logo.svg" alt="" width="34" height="34">Venn</a>' in index
    assert '<link rel="icon" type="image/svg+xml" href="media/venn-logo.svg">' in index
    assert (output / "media/venn-logo.svg").is_file()
    assert "docs-scope" not in index and "User documentation" not in index
    assert 'href="new-guide.html"' in index
    assert (output / "skills/venn/SKILL.md").read_text(encoding="utf-8") == "# Skill"
    assert '<video controls playsinline preload="none"' in index
    assert '<track kind="captions" src="media/venn-arena-captions.vtt"' in index
    assert not re.search(r"<track[^>]*venn-arena-captions[^>]*\bdefault\b", index)  # captions are burned in
    assert "autoplay" not in index
    arena_url = "media/venn-arena.mp4" if local_video_files else build_site.VIDEO_URL
    assert f'<source src="{arena_url}"' in index
    assert (output / "media/venn-arena.mp4").exists() == local_video_files
    companion = (output / "same-memory-video.html").read_text(encoding="utf-8")
    assert companion.count("<video controls playsinline") == 2
    assert 'width="1920" height="1080"' in companion
    assert 'width="1080" height="1920"' in companion
    for stem in ("venn-same-memory-review", "venn-same-memory-vertical-review"):
        url = f"media/{stem}.mp4" if local_video_files else f"{build_site.MEDIA_RELEASE}/{stem}.mp4"
        assert f'<source src="{url}"' in companion
        assert f'<a href="{url}">Download the video</a>' in companion
        assert f'<track kind="captions" src="media/{stem}-captions.vtt"' in companion
        track = re.search(rf'<track\b[^>]*src="media/{stem}-captions.vtt"[^>]*>', companion)[0]
        assert not re.search(r"\bdefault\b", track)
        assert (output / "media" / f"{stem}.mp4").is_file() == local_video_files
    assert "autoplay" not in companion
    demos = (output / "demos.html").read_text(encoding="utf-8")
    assert demos.count("<video controls playsinline") == 3
    desk_url = ("media/venn-precedent-desk.mp4" if local_video_files
                else f"{build_site.MEDIA_RELEASE}/venn-precedent-desk.mp4")
    assert f'<source src="{desk_url}" type="video/mp4">' in demos
    assert 'poster="media/precedent-desk-poster.jpg"' in demos
    assert '<track kind="captions" src="media/precedent-desk-captions.vtt"' in demos
    assert (output / "media/venn-precedent-desk.mp4").is_file() == local_video_files
    assert "user-attachments/assets/" not in demos + companion
    landscape_url = ("media/venn-same-memory-review.mp4" if local_video_files
                     else f"{build_site.MEDIA_RELEASE}/venn-same-memory-review.mp4")
    for page in (index, (output / "venn-and-jev.html").read_text(encoding="utf-8")):
        assert f'<source src="{landscape_url}" type="video/mp4">' in page
        assert 'poster="media/venn-same-memory-review-poster.jpg"' in page
        assert '<track kind="captions" src="media/venn-same-memory-review-captions.vtt"' in page
        assert "autoplay" not in page
    assert (output / "media/same-memory/history-eval.json").is_file()
    assert (output / "examples/same_memory_video/history.ipynb").is_file()
    assert (output / "examples/wine/README.md").is_file()
    assert 'aria-label="Documentation"' in (output / "examples/hotdog/README.html").read_text(encoding="utf-8")
    assert (output / "examples/hotdog/index.html").read_text(encoding="utf-8") == files["examples/hotdog/index.html"]
    assert not (output / "examples/hotdog/photos").exists()
    assert not (output / "examples/hotdog/local-state").exists()
    assert not list(output.rglob("__pycache__"))
    assert (output / "LICENSE").read_text(encoding="utf-8") == files["LICENSE"]
    assert (output / "NOTICE").read_text(encoding="utf-8") == files["NOTICE"]
    licensing = (output / "licenses.html").read_text(encoding="utf-8")
    assert 'href="LICENSE"' in licensing and 'href="NOTICE"' in licensing


@pytest.mark.parametrize("default_encoding", ["utf-8", "cp1252"])
def test_project_site_works_under_pages_subpath_without_a_backend(tmp_path, monkeypatch, default_encoding):
    # Generated UTF-8 pages must not depend on the host's default text encoding.
    original_read = Path.read_text

    def platform_read(path, encoding=None, errors=None):
        return original_read(path, encoding=encoding or default_encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", platform_read)
    source = build_site.ROOT
    monkeypatch.setattr(build_site, "ROOT", source)
    output = tmp_path / "site"
    build_site.build(output)

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
            self.ids = set()
            self.relations = {}

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag == "a" and attributes.get("rel") in {"prev", "next"}:
                self.relations[attributes["rel"]] = attributes["href"]
            for key, value in attrs:
                if key in {"href", "src", "poster"}:
                    self.links.append(value)
                elif key == "id":
                    self.ids.add(value)

    parsed = {}
    for path in output.rglob("*.html"):
        # Runnable HTML applications are assets, not documentation pages.
        document = path.read_text(encoding="utf-8")
        if '<nav aria-label="Documentation">' not in document:
            continue
        parser = Links()
        parser.feed(document)
        parsed[path.relative_to(output).as_posix()] = parser
    for name, parser in parsed.items():
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            assert not url.path.startswith("/"), (name, link)
            relative = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(url.path))) if url.path else name
            target = output / relative
            assert target.is_file(), (name, link)
            if url.fragment and relative in parsed:
                assert unquote(url.fragment) in parsed[relative].ids, (name, link)

    # Every rendered page is reachable from every documentation sidebar.
    for name in parsed:
        document = (output / name).read_text(encoding="utf-8")
        sidebar = re.search(r'<nav aria-label="Documentation">(.*?)</nav>', document, re.DOTALL)[1]
        targets = {posixpath.normpath(posixpath.join(posixpath.dirname(name), link))
                   for link in re.findall(r'href="([^"]+)"', sidebar)}
        assert targets == set(parsed)
        active = re.findall(r'href="([^"]+)" aria-current="page"', sidebar)
        assert len(active) == 1
        assert posixpath.normpath(posixpath.join(posixpath.dirname(name), active[0])) == name
        assert 'aria-label="Previous and next pages"' in document
        if name != "index.html":
            assert 'aria-label="Breadcrumb"' in document
        assert 'id="docs-query"' in document

    # Reading trails cover both audiences without sending a user into release
    # procedures. Previous links reverse the appropriate trail, including nested examples.
    for start, expected in (
        ("index.html", {name for name in parsed if not name.startswith("maintainers/")}),
        ("maintainers/maintaining.html", {name for name in parsed if name.startswith("maintainers/")}),
    ):
        visited = []
        current = start
        while current:
            assert current not in visited
            previous = visited[-1] if visited else None
            visited.append(current)
            relations = parsed[current].relations
            if previous:
                assert posixpath.normpath(posixpath.join(posixpath.dirname(current), relations["prev"])) == previous
            else:
                assert "prev" not in relations
            following = relations.get("next")
            current = posixpath.normpath(posixpath.join(posixpath.dirname(current), following)) if following else None
        assert set(visited) == expected

    assert 'class="maintainer-navigation" open' not in (output / "getting-started.html").read_text(encoding="utf-8")
    assert 'class="maintainer-navigation" open' in (output / "maintainers/maintaining.html").read_text(encoding="utf-8")

    # Search includes every page and section destinations actually exist.
    entries = json.loads((output / "search-index.js").read_text(encoding="utf-8").removeprefix("window.VENN_SEARCH = ").strip().removesuffix(";"))
    assert {urlsplit(entry["url"]).path for entry in entries} == set(parsed)
    assert any(entry["section"] == "Troubleshooting" for entry in entries)
    for entry in entries:
        url = urlsplit(entry["url"])
        assert entry["audience"] == ("maintainer" if url.path.startswith("maintainers/") else "user")
        if url.fragment:
            assert url.fragment in parsed[url.path].ids

    assert "start" in parsed["index.html"].ids
    index_html = (output / "index.html").read_text(encoding="utf-8")
    assert 'href="venn-and-jev.html"' in index_html
    assert 'href="examples/index.html"' in index_html
    arena = (output / "arena-example.html").read_text(encoding="utf-8")
    arena_url = ("media/venn-arena.mp4" if (source / "docs/media/venn-arena.mp4").is_file()
                 else build_site.VIDEO_URL)
    assert '<video controls playsinline' in arena
    assert f'<source src="{arena_url}" type="video/mp4">' in arena
    assert 'poster="media/venn-arena-poster.jpg"' in arena
    assert '<track kind="captions" src="media/venn-arena-captions.vtt"' in arena
    comparison = (output / "venn-and-jev.html").read_text(encoding="utf-8")
    comparison_url = ("media/venn-same-memory-review.mp4"
                      if (source / "docs/media/venn-same-memory-review.mp4").is_file()
                      else f"{build_site.MEDIA_RELEASE}/venn-same-memory-review.mp4")
    assert f'<source src="{comparison_url}" type="video/mp4">' in comparison
    assert 'poster="media/venn-same-memory-review-poster.jpg"' in comparison
    for name, stems in {
        "docs/arena-example.md": ("venn-arena",),
        "docs/venn-and-jev.md": ("venn-same-memory-review",),
        "docs/demos.md": ("venn-arena", "venn-same-memory-review", "venn-precedent-desk"),
        "docs/same-memory-video.md": ("venn-same-memory-review", "venn-same-memory-vertical-review"),
    }.items():
        text = (source / name).read_text(encoding="utf-8")
        for stem in stems:
            assert re.search(
                rf'<!-- video: {stem} -->\n\nhttps://github.com/user-attachments/assets/[\w-]+\n\n', text
            ), (name, stem)
        page = (output / build_site.source_pages(source)[1][(source / name).resolve()]).read_text(encoding="utf-8")
        assert page.count("<video controls playsinline") == len(stems)
        assert "user-attachments/assets/" not in page
    assert 'href="examples/wine/index.html"' in comparison
    assert 'href="examples/audio/index.html"' in comparison
    example_index = (output / "examples/index.html").read_text(encoding="utf-8")
    assert 'href="wine/index.html"' in example_index
    assert 'href="../api.html"' in example_index
    assert (output / "examples/audio/emotion.py").is_file()
    assert (output / "site.css").read_bytes() == (source / "docs/site.css").read_bytes()
    assert not list(output.rglob("*.safetensors"))
    assert not (output / "results").exists()
    assert not (output / "design").exists()
    assert "Vector Decision Models</span></h1>" in (output / "index.html").read_text(encoding="utf-8")
