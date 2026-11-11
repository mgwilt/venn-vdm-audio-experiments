from pathlib import Path
import re
from urllib.parse import urlsplit, unquote

from scripts import build_site, update_doc_navigation


def test_markdown_navigation_is_visible_current_and_has_valid_destinations():
    root = Path(__file__).resolve().parents[1]
    expected = update_doc_navigation.render_navigation(root)
    assert (root / 'README.md') not in expected
    assert (root / 'docs/README.md') in expected
    assert (root / 'docs/overview.md') in expected
    for path, generated in expected.items():
        text = path.read_text(encoding='utf-8')
        assert text == generated, f'Run python scripts/update_doc_navigation.py: {path.relative_to(root)}'
        header = re.search(r'<!-- docs-navigation:start -->(.*?)<!-- docs-navigation:end -->', text, re.DOTALL)[1]
        assert 'Documentation' in header and 'Examples' in header and 'API' in header
        primary = header.split('<details>', 1)[0]
        assert '[Adapters](' in primary or '**Adapters**' in primary
        assert text.index('<!-- docs-navigation:start -->') < text.index('<!-- docs-pager:start -->')
        assert '[Venn vs Jev]' in header or '**Venn vs Jev**' in header
        assert '<summary>Browse all documentation</summary>' in header
        destinations = {(path.parent / target).resolve()
                        for target in re.findall(r'\]\(([^)]+)\)', header)}
        for destination in destinations:
            assert destination.is_file(), (path, destination)
        # Every source page is reachable, including future top-level docs.
        # The current page is highlighted rather than linked; overview is an alias.
        required = {page.resolve() for page in expected if page.name != 'overview.md'}
        assert required <= destinations | {path.resolve()}, (path, required - destinations - {path.resolve()})
        pager = text.split('<!-- docs-pager:start -->')[1]
        for target in re.findall(r'\]\(([^)]+)\)', pager):
            assert (path.parent / target).is_file(), (path, target)
            if 'maintainers' not in path.relative_to(root).parts:
                assert 'maintainers' not in (path.parent / target).resolve().relative_to(root).parts
        # Source section links match actual rendered section IDs, not comments
        # inside code blocks. The site never needs to render the source controls.
        clean = build_site.without_source_navigation(text)
        rendered = build_site.markdown.Markdown(extensions=['fenced_code', 'tables', 'toc']).convert(clean)
        ids = set(re.findall(r'\bid="([^"]+)"', rendered))
        contents = re.search(r'<!-- docs-contents:start -->(.*?)<!-- docs-contents:end -->', text, re.DOTALL)
        if contents:
            for target in re.findall(r'\]\(([^)]+)\)', contents[1]):
                url = urlsplit(target)
                assert not url.path and unquote(url.fragment) in ids
        assert '<!-- docs-navigation:' not in clean and '<!-- docs-pager:' not in clean


def test_docs_readme_keeps_the_html_address_and_compatibility_links(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(build_site, 'ROOT', root)
    output = tmp_path / 'site'
    build_site.build(output)
    page = (output / 'overview.html').read_text(encoding='utf-8')
    assert '<h1 id="documentation-overview">' in page
    assert 'docs-navigation:start' not in page and 'docs-contents:start' not in page
    assert 'docs-pager:start' not in page
    assert 'View this page on GitHub' in page
    assert '/blob/main/docs/README.md' in page
    assert not (output / 'README.html').exists()
    entries = (output / 'search-index.js').read_text(encoding='utf-8')
    assert 'Documentation overview moved' not in entries
