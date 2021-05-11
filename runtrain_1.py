import os
import shlex
import subprocess
import threading
import signal
from itertools import product

f = open("SSM_training_data", 'w')
f.close()

def run_command(env):
    print('=> run commands on GPU:{}', env['CUDA_VISIBLE_DEVICES'])

    while True:
        try:
            comm = command_list.pop(0)
        except IndexError:
            break

        proc = subprocess.Popen(comm, env=env)
        proc.wait()


test_param_groups = {}
test_param_groups["channel_group"] = [64,128,256]
test_param_groups["iteration_group"] = ["2111","2222"]
test_param_groups["lr_group"] = [1.0,0.1]
test_param_groups["wd_group"] = [0.0001,0.0005,0.001]
test_param_groups["trail_group"] = [1,2]
test_param_groups["batch_size_group"] = [64,128,256]
test_param_groups["epochs_group"] = [120,240,360]
test_param_groups["net_group"] = ["resnet18","mgnet"]
test_param_groups["leaky_group"] = [4,8,10]
test_param_groups["momentum_group"] = [0.6,0.8]
test_param_groups["significance_group"] = [0.01,0.05,0.1]
test_param_groups["samplefreq_group"] = [5,10,15]
test_param_groups["truncate_group"] = [0.01,0.02,0.03]
test_param_groups["ministate_group"] = [50,100,150]
test_param_groups["keymode_group"] = ["loss_plus_smooth","loss"]
test_param_groups["varmode_group"] = ["bm","olbm"]
test_param_groups["data_group"] = ["cifar10","cifar100"]




# flexible
chosen_param_groups = ["channel_group","iteration_group","batch_size_group","momentum_group","wd_group","keymode_group","varmode_group","leaky_group","ministate_group","samplefreq_group","truncate_group","trail_group"]
value_list = [test_param_groups[group] for group in chosen_param_groups]
test_list = list(product(*value_list))
command = 'python trainmain_1.py --cuda --net=mgnet --ch={} --iter={} --data=cifar10 --epochs=360 --lr=1 -b={} -m={} --wd={} --km={} --vm={} --lk={} --minstat={} --sf={} --trun={} --sig=0.05 --trail={}'
# flexible


command_list = []
for file in test_list:
    command_list += [command.format(*file)]
command_list = [shlex.split(comm) for comm in command_list]


# List all the GPUs you have

ids_cuda = [0,1,2,3]
for c in ids_cuda:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(c)
    thread = threading.Thread(target = run_command, args = (env, ))
    thread.start()
