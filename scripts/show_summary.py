import csv, sys
for f in sys.argv[1:]:
    print("###", f)
    rows = sorted(csv.DictReader(open(f)), key=lambda r: float(r["wer"]))
    print("  %-17s%7s%7s%7s%7s%8s%7s" % ("system","wer","S","D","I","f1rare","cnt"))
    for r in rows:
        print("  %-17s%7.3f%7.3f%7.3f%7.3f%8.3f%7.3f" % (
            r["system"], float(r["wer"]), float(r["sub"]), float(r["del"]),
            float(r["ins"]), float(r["f1_rare"]), float(r["count_relerr"])))
