import gzip
from pathlib import Path

from preprocessing.simpleppdb import collect_simpleppdb_replacements


def write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def test_collect_simpleppdb_replacements_uses_score_direction(tmp_path: Path) -> None:
    simpleppdb_path = tmp_path / "simpleppdb.tsv.gz"
    write_gzip(
        simpleppdb_path,
        "\n".join(
            [
                "utilize\tuse\t2.5\t1.0",
                "easy\tarduous\t-3.0\t1.0",
                "multi word\tphrase\t4.0\t1.0",
                "outside\tknown\t5.0\t1.0",
            ]
        ),
    )

    dictionary, metadata = collect_simpleppdb_replacements(
        simpleppdb_path,
        source_vocabulary={"utilize", "arduous"},
    )

    assert dictionary.best_replacement("utilize") == "use"
    assert dictionary.best_replacement("arduous") == "easy"
    assert dictionary.best_replacement("outside") is None
    assert metadata.rules_seen == 4
    assert metadata.usable_rules == 2
    assert metadata.replacement_sources == 2
