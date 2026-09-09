"""Verify a transferred corpus against the sender's per-file manifest.

A matching TOTAL is not verification: a truncated file compensated by a longer one
elsewhere passes it, and that is the exact shape rsync failure takes on a flaky link.
So this compares per-file sizes and reports the three failure classes separately -
missing, extra, and size-mismatched - rather than a single boolean.
"""
import sys, pathlib, collections

manifest_path, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
expected = {}
for line in manifest_path.read_text().splitlines():
    if not line.strip():
        continue
    size, path = line.split(None, 1)
    expected[path.lstrip('./')] = int(size)

# Manifest paths are relative to the corpus's PARENT (they lead with the corpus dir),
# so scanning from the corpus dir itself makes every path look both missing and extra.
# Derive the prefix from the manifest rather than assuming either convention: an earlier
# version of this script assumed the wrong one and reported 146 missing and 146 extra
# files on a corpus whose count and byte total both matched exactly - a checker failing,
# not a transfer.
prefixes = {k.split('/', 1)[0] for k in expected}
scan_root = root if len(prefixes) != 1 else (root if root.name in prefixes else root)
strip = ''
if len(prefixes) == 1:
    only = prefixes.pop()
    if root.name == only:
        strip = only + '/'          # rooted AT the corpus: add the prefix back on
actual = {}
for p in scan_root.rglob('*'):
    if p.is_file():
        rel = str(p.relative_to(scan_root))
        actual[strip + rel] = p.stat().st_size

missing   = {k: v for k, v in expected.items() if k not in actual}
extra     = {k: v for k, v in actual.items() if k not in expected}
mismatch  = {k: (expected[k], actual[k]) for k in expected if k in actual and expected[k] != actual[k]}

print(f"manifest : {len(expected):,} files, {sum(expected.values()):,} bytes")
print(f"on disk  : {len(actual):,} files, {sum(actual.values()):,} bytes")
# rsync writes an in-flight file as `.<name>.<random>` and renames on completion, so a
# leftover temp is not an unexpected FILE, it is an INCOMPLETE TRANSFER. Reporting it as
# "extra: 1" at the end would be ambiguous at exactly the moment the answer matters.
partials = {k: v for k, v in extra.items() if pathlib.Path(k).name.startswith('.')}
unexpected = {k: v for k, v in extra.items() if k not in partials}
print(f"missing  : {len(missing)}")
print(f"extra    : {len(extra)}"
      + (f"  ({len(partials)} rsync in-flight temp file(s) -> TRANSFER INCOMPLETE)" if partials else ""))
print(f"size mismatch: {len(mismatch)}")
for k in list(missing)[:5]:  print("   MISSING", k)
for k in list(partials)[:5]: print("   PARTIAL", k, "(rsync still writing, or stopped mid-file)")
for k in list(unexpected)[:5]: print("   EXTRA  ", k, "(not in the manifest and not an rsync temp)")
for k, (e, a) in list(mismatch.items())[:5]: print(f"   SIZE   {k}: expected {e}, got {a}")
# A lingering partial means the transfer stopped early, so it must block the verdict even
# though every file that DID arrive is intact.
ok = not missing and not mismatch and not partials
print("\nVERIFIED: every file present at its recorded size." if ok else "\nFAILED - do not use this corpus.")
sys.exit(0 if ok else 1)
