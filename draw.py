import re
import matplotlib.pyplot as plt 

D_loss = []
G_loss = []
G_cls_loss = []
with open('log.txt', 'r') as f:
	for st in f:
		res1 = re.findall('D: (.+?),', st)
		res2 = re.findall('G: (.+?),', st)
		res3 = re.findall('G_cls: (.+?)\n', st)
		print(res3)
		D_loss.append(float(res1[0]))
		G_loss.append(float(res2[0]))
		G_cls_loss.append(float(res3[0]))
x = range(300)
plt.plot(x, D_loss, 'b', label = 'D_loss')
plt.plot(x, G_loss, 'r', label = 'G_loss')
plt.plot(x, G_cls_loss, 'g', label = 'G_cls_loss')
plt.legend(bbox_to_anchor=[0.3, 1]) 
plt.show()