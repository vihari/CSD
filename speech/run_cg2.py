import numpy as np
import os
import re

_dir = "logs_sanity_check6"
fnames = os.listdir(_dir)
patt = "= (.*)%"
dev_accs, test_accs = {}, {}

for fname in fnames:
  flds = fname.split("_")
  name = "|".join(flds[2:4]) + "|" + flds[-2] 

  f = open(_dir + "/" + fname)
  lines = f.readlines()
  lines = lines[-10:]
  try:
    dev_line_idx = [lidx for lidx, line in enumerate(lines) if line.find('Validation') >= 0][0]
  except IndexError:
    continue
  f.close()
  test_line = lines[-1]
  dev_line = lines[dev_line_idx]

  grp = re.search(patt, test_line)
  if grp is None:
    continue
  test_acc = grp.group(1)

  grp = re.search(patt, dev_line)
  dev_acc = grp.group(1)

  dev_accs[name] = dev_accs.get(name, []) + [float(dev_acc)]
  test_accs[name] = test_accs.get(name, []) + [float(test_acc)]

for name in dev_accs:
  dacc = dev_accs[name]
  tacc = test_accs[name]
  print ("|%s| %0.3f (%0.3f, %d)| %0.3f (%0.3f, %d)|" % (name, np.mean(dacc), np.std(dacc), len(dacc), np.mean(tacc), np.std(tacc), len(tacc)))
