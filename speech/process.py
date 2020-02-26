import os
import sys

# fldr = "rmnist_logs_3"
fldr = sys.argv[1]
print fldr
for fname in os.listdir(fldr):
    if fname.find('ucgv8_1000') < 0:
        continue
    #if fname.find('ucg_wogating_100_1') < 0:
    #    continue

    # if not fname.endswith('dfunc=l1'):
    #    continue
    flds = fname.split("_")
    uid = "|".join(flds[-4:])
    test_acc, val_acc = None, None
    with open(os.path.join(fldr, fname)) as f:
        num = 0
        val_patt = 'Validation accuracy = '
        test_patt = 'test accuracy = '
        for l in f:
            idx =  l.find(val_patt)
            if idx >= 0:
                for idx2 in range(idx, len(l)):
                    if l[idx2]=='%':
                        break
                val_acc = float(l[idx+len(val_patt):idx2])
                
            idx =  l.find(test_patt)
            if idx >= 0:
                for idx2 in range(idx, len(l)):
                    if l[idx2]=='%':
                        break
                test_acc = float(l[idx+len(test_patt):idx2])
    print "|" + uid + "|" + "|".join(map(str, [val_acc, test_acc])) + "|"
