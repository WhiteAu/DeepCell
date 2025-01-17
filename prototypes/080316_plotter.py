import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import seaborn as sns
mpl.rcParams['pdf.fonttype'] = 42

h1 = np.load('/home/nquach/DeepCell2/trained_networks/2016-08-02_ecoli_all_31x31_bn_feature_net_31x31_0.npz')['loss_history'][()]
h2 = np.load('/home/nquach/DeepCell2/trained_networks/2016-08-02_ecoli_all_31x31_bn_feature_net_31x31_1.npz')['loss_history'][()]
h3 = np.load('/home/nquach/DeepCell2/trained_networks/2016-08-02_ecoli_all_31x31_bn_feature_net_31x31_2.npz')['loss_history'][()]
h4 = np.load('/home/nquach/DeepCell2/trained_networks/2016-08-02_ecoli_all_31x31_bn_feature_net_31x31_3.npz')['loss_history'][()]
h5 = np.load('/home/nquach/DeepCell2/trained_networks/2016-08-02_ecoli_all_31x31_bn_feature_net_31x31_4.npz')['loss_history'][()]

e1 = np.subtract(1, h1['acc'])
e2 = np.subtract(1, h2['acc'])
e3 = np.subtract(1, h3['acc'])
e4 = np.subtract(1, h4['acc'])
e5 = np.subtract(1, h5['acc'])
estack = np.stack([e1, e2, e3, e4, e5], axis=0)
emu = np.mean(estack, axis = 0)
es = np.std(estack, axis = 0)

v1 = np.subtract(1, h1['val_acc'])
v2 = np.subtract(1, h2['val_acc'])
v3 = np.subtract(1, h3['val_acc'])
v4 = np.subtract(1, h4['val_acc'])
v5 = np.subtract(1, h5['val_acc'])
vstack = np.stack([v1, v2, v3, v4, v5], axis=0)
vmu = np.mean(vstack, axis=0)
vs = np.std(vstack, axis=0)

epoch = np.arange(1, len(e1)+1, 1)

sns.set_style("white")
solid = mlines.Line2D([], [], color='black', linestyle = '-', label = 'Training')
dashed = mlines.Line2D([],[], color='black', linestyle = '--', label= 'Validation')

plt.figure(0)
plt.plot(epoch, e1, 'k-', epoch, v1, 'k--', epoch, e2, 'k-', epoch, v2, 'k--', epoch, e3, 'k-', epoch, v3, 'k--', epoch, e4, 'k-', epoch, v4, 'k--', epoch, e5, 'k-', epoch, v5, 'k--')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(handles=[solid,dashed])
plt.title('Training and validation error: E. coli all 31x31 feature net')
filename = '/home/nquach/DeepCell2/prototypes/plots/080316_plots/bn_feature_net_31x31_ecoli_all.pdf'
plt.savefig(filename, format = 'pdf')
plt.close()

plt.figure(1)
plt.errorbar(epoch, emu, yerr = es, ls = '-', color=(0.835,0.369,0))
plt.errorbar(epoch, vmu, yerr = vs, ls= '--', color=(0.835,0.369,0))
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Average training and validation error: E. coli 31x31 feature net')
plt.legend(['training','validation'], loc='upper right')
filename = '/home/nquach/DeepCell2/prototypes/plots/080316_plots/bn_feature_net_31x31_ave_ecoli_all.pdf'
plt.savefig(filename, format='pdf')
plt.close()



