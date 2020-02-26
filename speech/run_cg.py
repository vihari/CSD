import numpy as np
import os

split=1000
model="mos"

# ckpt="ckpts/speech_commands_ucgv13_100_1_0.25_5_0/conv.ckpt-16200"
#1000
simple_ckpts = {
  50: ["/tmp/speech_commands_simple_50_0.005_0_0_5_0/conv.ckpt-2463"],
  # 50: ["/tmp/speech_commands_simple_50_0.01,0.001_0_0_5_0/conv.ckpt-24000"],
  100: ["/tmp/speech_commands_simple_100_0.005_0_0_5_0/conv.ckpt-4680"],
  200: ["/tmp/speech_commands_simple_200_0.01,0.001_0_0_5_0/conv.ckpt-23200"],
  1000: ["/tmp/speech_commands_simple_1000_0.007,0.001_0_0_15_0/conv.ckpt-21200"],
}
mos_ckpts = {
  #  50: ["/tmp/speech_commands_mos_50_0.01,0.001_0_0_5_0/conv.ckpt-17200", "/tmp/speech_commands_mos_50_0.01,0.001_0_0_5_1/conv.ckpt-16400", "/tmp/speech_commands_mos_50_0.01,0.001_0_0_5_2/conv.ckpt-22400"],
  50: ["/tmp/speech_commands_mos_50_0.005_0_0_5_0/conv.ckpt-2400", "/tmp/speech_commands_mos_50_0.005_0_0_5_1/conv.ckpt-2463", "/tmp/speech_commands_mos_50_0.005_0_0_5_2/conv.ckpt-2463"],
  100: ["/tmp/speech_commands_mos_100_0.005_0_0_5_0/conv.ckpt-4400", "/tmp/speech_commands_mos_100_0.005_0_0_5_1/conv.ckpt-3200", "/tmp/speech_commands_mos_100_0.005_0_0_5_2/conv.ckpt-4000"],
  200: ["/tmp/speech_commands_mos_200_0.01,0.001_0_0_5_0/conv.ckpt-24000", "/tmp/speech_commands_mos_200_0.01,0.001_0_0_5_1/conv.ckpt-23600", "/tmp/speech_commands_mos_200_0.01,0.001_0_0_5_2/conv.ckpt-23600"],
  1000: ["/tmp/speech_commands_mos_1000_0.007,0.001_0_0_15_0/conv.ckpt-23600", "/tmp/speech_commands_mos_1000_0.007,0.001_0_0_15_1/conv.ckpt-23600"],
}
nus = {50: 5, 100: 5, 200: 5, 1000: 15}

lrs = [0.01]
moms = [0.]
nseeds = [20]
inners = [5]
nsteps = [1]

def pick(*objs):
  _p = []
  for obj in objs:
    _p.append(np.random.choice(obj))
  return _p

for split in [100]:
  for ci, ckpts in enumerate([simple_ckpts, mos_ckpts]):
    if ci == 0:
      sfname = "eval_mos.py"
    else:
      sfname = "eval_mos2.py"

    nu = nus[split]
    ckpts_for_split_and_type = ckpts[split]
    seed = 1
    dev_accs, test_accs = [], []
    for ckpt in ckpts_for_split_and_type:
      run_str = "python3.5 %s --epsilon=0 --alpha=0 --num_uids=%d --training_percentage %d --model_dir %s --model mos --seed %d" % (sfname, nu, split, ckpt, seed)
      run_str += " >run.txt 2>/dev/null"
      print (run_str)
      os.system(run_str)
      
      with open("run.txt") as f:
        lines = f.readlines()[-2:]
        _ln = len('Accuracy: ')
        dev_acc = float(lines[0][_ln:])
        test_acc = float(lines[1][_ln:])
        print (dev_acc, test_acc)
        dev_accs.append(dev_acc)
        test_accs.append(test_acc)
        
    print ("split: %d type: %d %f (%f) %f (%f)" % (split, ci, np.mean(dev_accs), np.std(dev_accs), np.mean(test_accs), np.std(test_accs)))
