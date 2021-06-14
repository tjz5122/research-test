import os
import shlex
import subprocess
import threading
import signal
from itertools import product

f = open("SSM_Mgnet_data", 'w')
f.write('count,channel,iteration,dataset,epochs,lr,wd,momentum,batchsize,truncate,dropfactor,significance,leakratio,ministat,samplefreq,varmode,keymode,trail,decrease_time,e1,a1,l1,e2,a2,l2,e3,a3,l3,e4,a4,l4,final_train_accuracy,final_test_accuracy,final_avg_loss,total_time\n')
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
test_param_groups["channel_group"] = [128,256]
test_param_groups["iteration_group"] = ["2222"]
test_param_groups["lr_group"] = [1.0]
test_param_groups["wd_group"] = [0.0001,0.0005,0.001]
test_param_groups["trail_group"] = [1,2]
test_param_groups["drop_factor_group"] = [2,5,10] #SASA+
test_param_groups["batch_size_group"] = [128,256]
test_param_groups["epochs_group"] = [240]
test_param_groups["leaky_group"] = [2,4,8,16]     #SASA+
test_param_groups["momentum_group"] = [0.6,0.8]
test_param_groups["significance_group"] = [0.001,0.01,0.05,0.1] #SASA+
test_param_groups["samplefreq_group"] = [5,10,15]
test_param_groups["truncate_group"] = [0.02,0.03]
test_param_groups["ministate_group"] = [50,100]
test_param_groups["keymode_group"] = ["loss_plus_smooth"]
test_param_groups["varmode_group"] = ["bm"]
test_param_groups["data_group"] = ["cifar10"]


# flexible
chosen_param_groups = ["channel_group","iteration_group","lr_group","wd_group","trail_group","drop_factor_group","batch_size_group","epochs_group","leaky_group","momentum_group","significance_group","samplefreq_group","truncate_group","ministate_group","keymode_group","varmode_group","data_group"]
value_list = [test_param_groups[group] for group in chosen_param_groups]
test_list = list(product(*value_list))
command = 'python SSM_Mgnet_train.py --cuda --ch={} --iter={} --lr={} --wd={} --trail={} --drop={} -b={} --epochs={} --lk={} -m={} --sig={} --sf={} --trun={} --minstat={} --km={} --vm={} --data={}'
# flexible

print("the total combinations of hyperparamater is", len(test_list))
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
