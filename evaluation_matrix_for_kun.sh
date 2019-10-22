#!/bin/bash
#
# path="/home/wyshi/simulator/model/save/template/oneHot_newReward_bitMore/0_2019-5-18-21-59-15-5-138-1.pkl" ok
# path="/home/wyshi/simulator/model/save/nlg_sample/oneHot_newReward_bitMore/best/0_2019-5-19-15-13-5-6-139-1.pkl"
# path="/home/wyshi/simulator/model/save/seq2seq/oneHot_newReward_bitMore/0_2019-5-19-22-28-16-6-139-1.pkl"
# path="/home/wyshi/simulator/model/save/sl_simulator/template/oneHot_oldReward_bitMore/0_2019-5-19-23-46-10-6-139-1.pkl"
# path="/home/wyshi/simulator/model/save/sl_simulator/retrieval/oneHot_oldReward_bitMore/best/0_2019-5-19-19-2-18-6-139-1.pkl"
path="/home/wyshi/simulator/model/save/sl_simulator/oneHot_oldReward_bitMore/best/0_2019-5-19-3-27-15-6-139-1.pkl"

sl='True'
gen='True'
temp='False'
samp='False'
python evaluation_matrix_for_kun.py -resume_rl_model_dir ${path} \
                                    -config use_sl_simulator=${sl} \
                                            use_sl_generative=${gen} \
                                            nlg_template=${temp} \
                                            nlg_sample=${samp}

gen='False'
temp='True'
samp='False'
python evaluation_matrix_for_kun.py -resume_rl_model_dir ${path} \
                                    -config use_sl_simulator=${sl} \
                                            use_sl_generative=${gen} \
                                            nlg_template=${temp} \
                                            nlg_sample=${samp}
gen='False'
temp='False'
samp='True'
python evaluation_matrix_for_kun.py -resume_rl_model_dir ${path} \
                                    -config use_sl_simulator=${sl} \
                                            use_sl_generative=${gen} \
                                            nlg_template=${temp} \
                                            nlg_sample=${samp}


sl='False'
gen='True'
temp='False'
samp='False'
python evaluation_matrix_for_kun.py -resume_rl_model_dir ${path} \
                                    -config use_sl_simulator=${sl} \
                                            use_sl_generative=${gen} \
                                            nlg_template=${temp} \
                                            nlg_sample=${samp}

gen='False'
temp='True'
samp='False'
python evaluation_matrix_for_kun.py -resume_rl_model_dir ${path} \
                                    -config use_sl_simulator=${sl} \
                                            use_sl_generative=${gen} \
                                            nlg_template=${temp} \
                                            nlg_sample=${samp}
gen='False'
temp='False'
samp='True'
python evaluation_matrix_for_kun.py -resume_rl_model_dir ${path} \
                                    -config use_sl_simulator=${sl} \
                                            use_sl_generative=${gen} \
                                            nlg_template=${temp} \
                                            nlg_sample=${samp}