from keras.backend import shape
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

"""udp:0 tcp:1 icmp:2"""
"""private:0 http:1 others:2 telnet:3 ftp_data:4 ecr_i:5 smtp:6 pop_3:7 domain_u:8 netbois_us:9 eco_i:10 discard:11 name:12 ntp_u:13 csnet_ns:14 finger:15 netbois_ss:16 efs:17 ftp:18 urp_i:19 rje:20 echo:21
Z39_50:22 systat:23 remote_job:24 auth:25 shell:26 login:27 imap4:28 gopher:29 nntp:30 time:31 uucp:32 sunrpc:33 ctf:34 vmnet:35
ldap:36 bgp:37 courier:38 ssh:39 IRC:40 iso_tsap:41"""

x={0: "DOS: denial-of-service, e.g. syn flood; (attack)",1: "normal (no attack)",2: "probing: surveillance and other probing, e.g., port scanning (attack)",
3: "R2L: unauthorized access from a remote machine, e.g. guessing password (attack)",
4: "U2R: unauthorized access to local superuser (root) privileges, e.g., various 'buffer overflow' attacks (attack)"}
df = pd.read_csv (r'C:\Users\Akhil\Desktop\testing.csv')

dt=pd.read_csv(r'C:\Users\Akhil\Desktop\training.csv')
x_feat=dt.iloc[:,1:42]
y_feat=dt.iloc[:,42]

x_feat=np.array(x_feat)
y_feat=np.array(y_feat)
print(x_feat[0])


model = Sequential()
model.add(Dense(128, input_shape=(449940, 40), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_feat, y_feat, epochs=200, batch_size=5, verbose=1)

y_pred =model.predict(x_test)

accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))


