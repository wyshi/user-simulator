mkdir -p model/save/template/oneHot_newReward_bitMore/
python run_mydata_new.py --one_hot=True --new_reward=True --nlg_sample=False --with_bit=2 --save_dir=model/save/template/oneHot_newReward_bitMore/ > model/save/template/oneHot_newReward_bitMore/1

mkdir -p model/save/template/oneHot_oldReward_bitMore/
python run_mydata_new.py --one_hot=True --new_reward=False --nlg_sample=False --with_bit=2 --save_dir=model/save/template/oneHot_oldReward_bitMore/ > model/save/template/oneHot_oldReward_bitMore/2 

mkdir -p model/save/template/oneHot_newReward_bitRep/
python run_mydata_new.py --one_hot=True --new_reward=True --nlg_sample=False --with_bit=1 --save_dir=model/save/template/oneHot_newReward_bitRep/ > model/save/template/oneHot_newReward_bitRep/3

mkdir -p model/save/template/oneHot_oldReward_bitRep/
python run_mydata_new.py --one_hot=True --new_reward=False --nlg_sample=False --with_bit=1 --save_dir=model/save/template/oneHot_oldReward_bitRep/ > model/save/template/oneHot_oldReward_bitRep/4 

mkdir -p model/save/template/oneHot_newReward_bitNo/
python run_mydata_new.py --one_hot=True --new_reward=True --nlg_sample=False --with_bit=0 --save_dir=model/save/template/oneHot_newReward_bitNo/ > model/save/template/oneHot_newReward_bitNo/5

mkdir -p model/save/template/oneHot_oldReward_bitNo/
python run_mydata_new.py --one_hot=True --new_reward=False --nlg_sample=False --with_bit=0 --save_dir=model/save/template/oneHot_oldReward_bitNo/ > model/save/template/oneHot_oldReward_bitNo/6 



mkdir -p model/save/nlg_sample/oneHot_newReward_bitMore/
python run_mydata_new.py --one_hot=True --new_reward=True --nlg_sample=True --with_bit=2 --save_dir=model/save/nlg_sample/oneHot_newReward_bitMore/ > model/save/nlg_sample/oneHot_newReward_bitMore/7

mkdir -p model/save/nlg_sample/oneHot_oldReward_bitMore/
python run_mydata_new.py --one_hot=True --new_reward=False --nlg_sample=True --with_bit=2 --save_dir=model/save/nlg_sample/oneHot_oldReward_bitMore/ > model/save/nlg_sample/oneHot_oldReward_bitMore/8 

mkdir -p model/save/nlg_sample/oneHot_newReward_bitRep/
python run_mydata_new.py --one_hot=True --new_reward=True --nlg_sample=True --with_bit=1 --save_dir=model/save/nlg_sample/oneHot_newReward_bitRep/ > model/save/nlg_sample/oneHot_newReward_bitRep/9

mkdir -p model/save/nlg_sample/oneHot_oldReward_bitRep/
python run_mydata_new.py --one_hot=True --new_reward=False --nlg_sample=True --with_bit=1 --save_dir=model/save/nlg_sample/oneHot_oldReward_bitRep/ > model/save/nlg_sample/oneHot_oldReward_bitRep/10 

mkdir -p model/save/nlg_sample/oneHot_newReward_bitNo/
python run_mydata_new.py --one_hot=True --new_reward=True --nlg_sample=True --with_bit=0 --save_dir=model/save/nlg_sample/oneHot_newReward_bitNo/ > model/save/nlg_sample/oneHot_newReward_bitNo/11

mkdir -p model/save/nlg_sample/oneHot_oldReward_bitNo/
python run_mydata_new.py --one_hot=True --new_reward=False --nlg_sample=True --with_bit=0 --save_dir=model/save/nlg_sample/oneHot_oldReward_bitNo/ > model/save/nlg_sample/oneHot_oldReward_bitNo/12 



# CUDA_VISIBLE_DEVICES=7

# #baseline
# for i in `seq 4 10`
# do
# c="$i"
# d="$i"
# e="$i"
# c="$c""_test.csv"
# d="$d""_test.ckpt"
# e="$e""_test"
# parent_dir="with_SL/base_60000_without_repetition_VAE-based-simulator/"
# parent_dir_1="performance/""$parent_dir"
# parent_dir_2="model/""$parent_dir"
# parent_dir_3="debug/""$parent_dir"
# mkdir -p "$parent_dir_1"
# mkdir -p "$parent_dir_2"
# mkdir -p "$parent_dir_3"
# c="$parent_dir_1""$c"
# d="$parent_dir_2""$d"
# e="$parent_dir_3""$e"
# echo "$c"
# echo "$d"
# echo "$e"
# # python run_mydata1.py --num_epoch 30000 --sample no --restore no --save_dir "$c" --model_dir "$d" --with_latent_state no --mask yes --with_w2v no --with_state_one_hot no --with_new_reward_function yes --SL yes --new_response no --with_simulated_prob no --with_rank_reward yes> "$e"
# python run_mydata1.py --num_epoch 60000 --sample yes --restore no --save_dir "$c" --model_dir "$d" --with_latent_state no --mask yes --with_w2v no --with_state_one_hot no --with_new_reward_function no --SL yes --new_response no --with_simulated_prob no --with_rank_reward no --with_additional_reward no --with_kl_reward no --with_repetition_penalty no > "$e"


# done
