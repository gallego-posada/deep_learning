import os

length_list = [5, 10, 20, 30, 50, 100]
model_list = ["RNN", "LSTM"]
optim_list = ["adam", "rmsprop"]
lr_list = [0.025, 0.0025, 0.00025]

for length in length_list:
    for model in model_list:
        for lr in lr_list:
            for optim in optim_list:

                name = "model_" + str(model) + "_len_" + str(length) + \
                       "_lr_" + str(lr) + "_opt_" + str(optim)

                cmd = "python train.py --name " + str(name) + \
                      " --learning_rate " + str(lr) + " --model_type " + str(model) +\
                      " --optimizer " + str(optim) + " --input_length " + str(length) +\
                      " --train_steps 2000"

                print("RUNNING: " + name)
                os.system(cmd)
