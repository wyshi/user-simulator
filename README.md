# user-simulator
Codebase for [How to Build User Simulators to Train RL-based Dialog Systems](https://arxiv.org/pdf/1909.01388.pdf), published as a long paper in EMNLP 2019. The sequicity part is developed based on [Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures](https://github.com/WING-NUS/sequicity).


If you use the datasets or any source codes included in this repository in your
work, please cite the following paper. The bibtex is listed below:

    @article{shi2019build,
      title={How to Build User Simulators to Train RL-based Dialog Systems},
      author={Shi, Weiyan and Qian, Kun and Wang, Xuewei and Yu, Zhou},
      journal={arXiv preprint arXiv:1909.01388},
      year={2019}
    }

# Agenda-based simulator
under simulator/

# Supervised-learning-based simulator
under sequicity_user/

** for the seq2seq model, because the codebase for the seq2seq module exceeds the file limit, please contact us for it. But it's a simple vanilla seq2seq, you can build your own. The code is under seq2seq/, and we use the implementation from https://github.com/IBM/pytorch-seq2seq for the seq2seq generation model. The vectors used in the training can be downloaded from https://nlp.stanford.edu/projects/glove/.


# RL training with agenda-based simulator
python run_mydata_new.py

# RL training with supervised-learning-based simulator
python run_mydata_seq_new.py

# Interacting with trained policies
policies are under simulator/policy/

