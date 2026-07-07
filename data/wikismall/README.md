# WikiSmall

This directory contains the de-anonymized/original WikiSmall parallel
simplification files used by `src/data/wikismall_loader.py`.

The files are line-aligned:

- `*.src`: complex/original sentence
- `*.dst`: simplified sentence

Splits:

| Split | Pairs |
| --- | ---: |
| train | 88,837 |
| valid | 205 |
| test | 100 |

These files correspond to the WikiSmall simplification corpus derived from
English Wikipedia and Simple English Wikipedia. The `.ori` variant maps the
named-entity placeholders in the anonymized `PWKP_108016.tag.80.aner.*` files
back to readable surface forms.
