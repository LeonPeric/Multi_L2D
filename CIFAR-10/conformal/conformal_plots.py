import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc

# === Latex Options === #
rc('font', family='serif')
rc('text', usetex=True)

# === Matplotlib Options === #
cm = plt.cm.get_cmap('tab10')
plot_args = {"marker": "o",
             "markeredgecolor": "k",
             "markersize": 10,
             "linewidth": 8
             }
sns.set_context("talk", font_scale=1.3)
fig_size = (7,7)


# === Plotting functions === #
def plot_sys_acc():
    pass


def plot_exp_acc():
    pass


def plot_qhat():
    pass


def plot_coverage():
    pass


#
# # ==== GRADUAL OVERLAP
#
# # PLOT ===
#
# p_out = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
#
# # OvA ===
# sys_acc_standard_ova = np.array([method_d["system_accuracy"] for method_d in method_dict_ova["standard"]])
# sys_acc_last_ova = np.array([method_d["system_accuracy"] for method_d in method_dict_ova["last"]])
# sys_acc_random_ova = np.array([method_d["system_accuracy"] for method_d in method_dict_ova["random"]])
# sys_acc_voting_ova = np.array([method_d["system_accuracy"] for method_d in method_dict_ova["voting"]])
#
# # Softmax ===
# sys_acc_standard_softmax = np.array([method_d["system_accuracy"] for method_d in method_dict_softmax["standard"]])
# sys_acc_last_softmax = np.array([method_d["system_accuracy"] for method_d in method_dict_softmax["last"]])
# sys_acc_random_softmax = np.array([method_d["system_accuracy"] for method_d in method_dict_softmax["random"]])
# sys_acc_voting_softmax = np.array([method_d["system_accuracy"] for method_d in method_dict_softmax["voting"]])
#
# f, ax = plt.subplots(1, 1, figsize=fig_size)
# # OvA ===
# ax.plot(p_out, sys_acc_last_ova, "-", label=r"Last", color=cm(0), **plot_args)
# ax.plot(p_out, sys_acc_random_ova, "-", label=r"Random", color=cm(1), **plot_args)
# ax.plot(p_out, sys_acc_voting_ova, "-", label=r"Voting", color=cm(2), **plot_args)
# ax.plot(p_out, sys_acc_standard_ova, "-", label=r"w/o conformal", color=cm(3), **plot_args)
#
# # Softmax ===
# ax.plot(p_out, sys_acc_last_softmax, "--", color=cm(0), **plot_args)
# ax.plot(p_out, sys_acc_random_softmax, "--", color=cm(1), **plot_args)
# ax.plot(p_out, sys_acc_voting_softmax, "--", color=cm(2), **plot_args)
# ax.plot(p_out, sys_acc_standard_softmax, "--", color=cm(3), **plot_args)
#
# plt.xticks(p_out, p_out)
# plt.yticks(list(plt.yticks()[0])[::2])
# plt.ylabel(r'System Acc. ($\%$)')
# plt.xlabel(r'Prob Experts')
# plt.title(r"CIFAR-10")
# plt.legend(loc="best")
# plt.grid()
# f.set_tight_layout(True)
# plt.legend()
#
# plt.savefig("system_acc_gradual_overlap.pdf")
#
# # In[64]:
#
#
# p_out = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
#
# # OvA ===
# exp_acc_standard_ova = np.array([method_d["expert_accuracy"] for method_d in method_dict_ova["standard"]])
# exp_acc_last_ova = np.array([method_d["expert_accuracy"] for method_d in method_dict_ova["last"]])
# exp_acc_random_ova = np.array([method_d["expert_accuracy"] for method_d in method_dict_ova["random"]])
# exp_acc_voting_ova = np.array([method_d["expert_accuracy"] for method_d in method_dict_ova["voting"]])
#
# # Softmax ===
# exp_acc_standard_softmax = np.array([method_d["expert_accuracy"] for method_d in method_dict_softmax["standard"]])
# exp_acc_last_softmax = np.array([method_d["expert_accuracy"] for method_d in method_dict_softmax["last"]])
# exp_acc_random_softmax = np.array([method_d["expert_accuracy"] for method_d in method_dict_softmax["random"]])
# exp_acc_voting_softmax = np.array([method_d["expert_accuracy"] for method_d in method_dict_softmax["voting"]])
#
# f, ax = plt.subplots(1, 1, figsize=fig_size)
# # OvA ===
# ax.plot(p_out, exp_acc_last_ova, "-", label=r"Last", color=cm(0), **plot_args)
# ax.plot(p_out, exp_acc_random_ova, "-", label=r"Random", color=cm(1), **plot_args)
# ax.plot(p_out, exp_acc_voting_ova, "-", label=r"Voting", color=cm(2), **plot_args)
# ax.plot(p_out, exp_acc_standard_ova, "-", label=r"w/o conformal", color=cm(3), **plot_args)
#
# # Softmax ===
# ax.plot(p_out, exp_acc_last_softmax, "--", color=cm(0), **plot_args)
# ax.plot(p_out, exp_acc_random_softmax, "--", color=cm(1), **plot_args)
# ax.plot(p_out, exp_acc_voting_softmax, "--", color=cm(2), **plot_args)
# ax.plot(p_out, exp_acc_standard_softmax, "--", color=cm(3), **plot_args)
#
# plt.xticks(p_out, p_out)
# plt.yticks(list(plt.yticks()[0])[::2])
# plt.ylabel(r'Expert Acc. ($\%$)')
# plt.xlabel(r'Prob Experts')
# plt.title(r"CIFAR-10")
# plt.legend(loc="best")
# plt.grid()
# f.set_tight_layout(True)
# plt.legend()
#
# plt.savefig("expert_acc_gradual_overlap.pdf")
#
# # In[65]:
#
#
# p_out = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
# coverage_ova = np.array([method_d["coverage"] for method_d in method_dict_ova["last"]])
# coverage_softmax = np.array([method_d["coverage"] for method_d in method_dict_softmax["last"]])
#
# f, ax = plt.subplots(1, 1, figsize=fig_size)
# ax.plot(p_out, coverage_ova, "-", label=r"OvA", **plot_args)
# ax.plot(p_out, coverage_softmax, "--", label=r"Softmax", **plot_args)
#
# plt.xticks(p_out, p_out)
# plt.yticks(list(plt.yticks()[0])[::2])
# plt.ylabel(r'Model Coverage. ($\%$)')
# plt.xlabel(r'Prob Experts')
# plt.title(r"CIFAR-10")
# plt.legend(loc="best")
# plt.grid()
# f.set_tight_layout(True)
# plt.legend()
#
# plt.savefig("coverage_gradual_overlap.pdf")