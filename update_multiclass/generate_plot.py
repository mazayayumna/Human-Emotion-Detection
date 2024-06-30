import matplotlib.pyplot as plt
import json

history = json.load(open('history.json', 'r'))

print('generate accuracy plot')
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()
plt.savefig('img/accuracy_plot.png')

print('generate loss plot')
plt.plot(history['loss'])
plt.plot(history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()
plt.savefig('img/loss_plot.png')