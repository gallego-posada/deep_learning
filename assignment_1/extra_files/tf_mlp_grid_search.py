import os

lr_list = [0.00007, 0.0003, 0.0005]
init_list = ["normal", "uniform"]
optim_list = ["adam", "rmsprop"]
act_list = ["relu", "elu"]
arch_list = ["100", "250,250", "500,500"]
drop_list =[0.1, 0.2]

for lr in lr_list:
    for init in init_list:
        for optim in optim_list:
            for act in act_list:
                for arch in arch_list:
                    for drop in drop_list:
                        name = "lr_" + str(lr) + "_init_" + str(init) + \
                               "_opt_" + str(optim) + "_act_" + str(act) + \
                               "_arch_" + str(arch) + "_drop_" + str(drop)

                        cmd = "python train_mlp_tf.py --name " + str(name) + \
                              " --learning_rate " + str(lr) + " --weight_init " + str(init) +\
                              " --optimizer " + str(optim) + " --dnn_hidden_units " + str(arch) +\
                              " --dropout_rate " + str(drop) + " --max_steps 2000"

                        print("RUNNING: " + name)
                        os.system(cmd)

    #                     break
    #                 break
    #             break
    #         break
    #     break
    # break
