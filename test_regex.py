import re

_EDGE_RE = re.compile(
    r"""(?P<sq>['"])(?P<src>.+?)(?P=sq)"""
    r"""\s*--\[(?P<rel>[^\]]+)\]-->\s*"""
    r"""(?P<tq>['"])(?P<tgt>.+?)(?P=tq)"""
    r"""(?P<flags>.*)"""
)

line = "'foo' --[rel]--> 'bar'"
m = _EDGE_RE.search(line)
print(f"Match: {m}")
if m:
    print(m.groupdict())
