
from data.simple_ppdb import SimplePPDB

def main() -> None:
    print("setting up")
    ppdb: SimplePPDB = SimplePPDB.load_from_disk('data/simple-ppdb/SimplePPDB');    
    print(ppdb.entries[0])
    print(f"dimension: {len(ppdb.entries)}")


if __name__ == "__main__":
    main()
