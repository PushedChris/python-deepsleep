
import pickle    #保存模型
file = open('XXXXXX.pickle', 'wb')
pickle.dump(X1, file)
pickle.dump(X2, file)
pickle.dump(X3, file)
file.close()


with open('XXXXXXX.pickle', 'rb') as file:
    X1 =pickle.load(file)
    X2 = pickle.load(file)
    X3= pickle.load(file)
file.close()