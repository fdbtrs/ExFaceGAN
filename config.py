architecture = "resnet50"

dataset = "ExFace_SG3"  # training dataset

batch_size = 128    # 256
workers = 8  # 32
embedding_size = 512
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4

output_dir = "output/ExFace_SG3_CosFace_RA"
global_step = 0  # to resume
start_epoch = 0

s = 64.0
m = 0.35     # 0.35  0.5
loss = "CosFace"     # CosFace, ArcFace
dropout_ratio = 0.4

augmentation = "ra_4_16"  # hf, ra_4_16, digiwarp

print_freq = 50
val_path = "/data/Biometrics/database/faces_emore"
val_targets = ["lfw", "agedb_30", "cfp_fp", "calfw", "cplfw"]

rec2 = None
if dataset == "ExFace_SG3":
    rec = "/data/synthetic_imgs/ExFaceGAN_SG3"
    num_classes = 10000
elif dataset == "ExFace_SG2":
    rec = "/data/synthetic_imgs/ExFaceGAN_SG2"
    num_classes = 10000
elif dataset == "ExFace_GANCon":
    rec = "/data/synthetic_imgs/ExFace_GANCon"
    num_classes = 10000

auto_schedule = False
num_epoch = 200 if auto_schedule else 40
schedule = [22, 30, 35]


def lr_step_func(epoch):
    return (
        ((epoch + 1) / (4 + 1)) ** 2
        if epoch < -1
        else 0.1 ** len([m for m in schedule if m - 1 <= epoch])
    )


lr_func = lr_step_func
