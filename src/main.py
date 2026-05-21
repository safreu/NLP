
from data.simple_ppdb import SimplePPDB
from data.one_stop_english_corpus import OneStopEnglish

def main() -> None:
    print("setting up")
    ppdb: SimplePPDB = SimplePPDB.load_from_disk('data/simple-ppdb/SimplePPDB');    
    print(ppdb.entries[0])
    print(f"dimension: {len(ppdb.entries)}")
    
    print("setting up")
    onestop: OneStopEnglish = OneStopEnglish.load_from_disk();    
    print(onestop.entries[0])
    print(f"dimension: {len(onestop.entries)}")



if __name__ == "__main__":
    main()
