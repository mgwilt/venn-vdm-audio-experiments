"""Keep GitHub-visible Markdown navigation aligned with the static site."""
import argparse
import html
import posixpath
import re

import markdown

try:
    from scripts.build_site import ROOT, PAGE_LABELS, navigation_groups, source_pages, plain_text, without_source_navigation
except ModuleNotFoundError:
    from build_site import ROOT, PAGE_LABELS, navigation_groups, source_pages, plain_text, without_source_navigation


def render_navigation(root=ROOT):
    pages, targets = source_pages(root)
    sources = {targets[page.resolve()].removesuffix('.html'): page for page in pages}
    groups = navigation_groups(targets)
    labels = {}
    for name, page in sources.items():
        heading = re.search(r'^# (.+)$', page.read_text(encoding='utf-8'), re.MULTILINE)
        labels[name] = PAGE_LABELS.get(name) or (plain_text(markdown.markdown(heading[1])) if heading else page.stem)
    labels['index'] = 'Home'
    trails = [[name for _, names in groups for name in names if name.startswith('maintainers/')],
              ['index', *[name for _, names in groups for name in names if not name.startswith('maintainers/')]]]
    primary = [('overview', 'Documentation'), ('getting-started', 'Get started'), ('concepts', 'Concepts'), ('venn-and-jev', 'Venn vs Jev'),
               ('examples/index', 'Examples'), ('cookbooks', 'Task recipes'), ('adapters', 'Adapters'),
               ('api', 'API'), ('evidence', 'Results')]
    result = {}
    entries = list(sources.items())
    compatibility = root / 'docs/overview.md'
    if (root / 'docs/README.md').is_file() and compatibility.is_file():
        entries.append(('overview', compatibility))
    for name, page in entries:
        # The fork's developer README is hand-authored; keep it linked from the
        # documentation without inserting the upstream directory and pager.
        if name == 'index':
            continue
        original = without_source_navigation(page.read_text(encoding='utf-8')).strip()+'\n'
        def link(target, label):
            destination = posixpath.relpath(sources[target].as_posix(), page.parent.as_posix())
            return f'[{label}]({destination})'
        header = ' · '.join(f'**{label}**' if sources[target] == page else link(target, label)
                            for target, label in primary if target in sources)
        group = next((label for label, names in groups if name in names), 'Project')
        directory = ['**Project home:** '+('**Home**' if name == 'index' else link('index', 'Home'))]
        for group_label, names in groups:
            items = [f'**{labels[target]}**' if sources[target] == page else link(target, labels[target])
                     for target in names if target in sources]
            if items:
                directory.append('**'+group_label+':** '+' · '.join(items))
        directory = '<details>\n<summary>Browse all documentation</summary>\n\n'+'\n\n'.join(directory)+'\n\n</details>'
        header = '<!-- docs-navigation:start -->\n'+header+'\n\n'+directory+'\n\n**'+group+' / '+labels[name]+'**\n<!-- docs-navigation:end -->\n\n'
        renderer = markdown.Markdown(extensions=['fenced_code', 'tables', 'toc'],
                                     extension_configs={'toc': {'toc_depth': '2'}})
        renderer.convert(original)
        def sections(tokens):
            for token in tokens:
                if token['level']==2:yield token
                yield from sections(token.get('children', []))
        sections = list(sections(renderer.toc_tokens))
        contents = ''
        if len(sections)>=3:
            entries = ['- ['+plain_text(html.unescape(token['name'])).replace('[',r'\[').replace(']',r'\]')+'](#'+token['id']+')' for token in sections]
            contents = '<!-- docs-contents:start -->\n<details>\n<summary>On this page</summary>\n\n'+'\n'.join(entries)+'\n\n</details>\n<!-- docs-contents:end -->\n\n'
        heading = re.search(r'^# .+\n', original, re.MULTILINE)
        offset = heading.end() if heading else 0
        body = original[:offset]+'\n'+header+contents+original[offset:].lstrip('\n')
        trail = trails[0 if name.startswith('maintainers/') else 1]
        position = trail.index(name)
        pager = []
        if position:pager.append(link(trail[position-1], '← Previous: '+labels[trail[position-1]]))
        if position+1<len(trail):pager.append(link(trail[position+1], 'Next: '+labels[trail[position+1]]+' →'))
        pager.append(link('overview', 'Documentation home'))
        result[page] = body.rstrip()+'\n\n<!-- docs-pager:start -->\n---\n\n'+' · '.join(pager)+'\n<!-- docs-pager:end -->\n'
    return result


def update(root=ROOT, check=False):
    stale=[]
    for page, text in render_navigation(root).items():
        if page.read_text(encoding='utf-8')!=text:
            stale.append(page.relative_to(root).as_posix())
            if not check:page.write_text(text,encoding='utf-8')
    return stale


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();stale=update(check=args.check)
    if args.check and stale:
        print('Run python scripts/update_doc_navigation.py to update: '+', '.join(stale))
        raise SystemExit(1)
    print('Markdown navigation current' if args.check else f'Updated navigation in {len(stale)} Markdown pages')
