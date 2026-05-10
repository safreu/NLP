from data.simple_ppdb import SimplePPDB
from data.simple_ppdb import SimplePPDBEntry
import pytest
from pathlib import Path

def write_test_data(tmp_path) -> Path:
    file = tmp_path / "valid.txt"

    file.write_text(
        "4.22848	1.00000	[X]	bring benefit	to benefit from\n"
        "4.00479	1.00000	[ADJP]	uncomplicated	simple enough\n"
        "3.88357	1.00000	[NP]	unaccompanied minors	minors",
        encoding="utf-8"
    )
    return file

def test_regex_valid(tmp_path):

    file = write_test_data(tmp_path)

    data = SimplePPDB.load_from_disk(file)

def test_regex_invalid(tmp_path):
    with pytest.raises(RuntimeError):
        file = tmp_path / "valid.txt"

        file.write_text(
            "4.2284	[X]	bring benefit	to benefit from\n",
            encoding="utf-8"
        )

        data = SimplePPDB.load_from_disk(file)

def test_caching(tmp_path):
    file = write_test_data(tmp_path)
    data_cache_file = file.with_suffix(".pkl")

    data = SimplePPDB.load_from_disk(file)
    data_cache = SimplePPDB.load_from_disk(file)


    assert data_cache_file.exists()
    assert data == data_cache

def test_load_data_correctly(tmp_path):
    file = write_test_data(tmp_path)

    data = SimplePPDB.load_from_disk(file)

    expected =  [
        SimplePPDBEntry(4.22848, 1.00000, "X", "bring benefit", "to benefit from"),
        SimplePPDBEntry(4.00479, 1.00000, "ADJP", "uncomplicated", "simple enough"),
        SimplePPDBEntry(3.88357, 1.00000, "NP", "unaccompanied minors", "minors")
    ]
    
    assert data.entries == expected
    



