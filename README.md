# TestPKUVerifyCode
An automatic script to crack the verification code of PKU course selection website using LibSVM.<br>
You can get those verification codes randomly from http://elective.pku.edu.cn/elective2008/DrawServlet.<br>
### Package Requirments
* PIL
* requests
* LibSVM
You can get the first two from pip and the last one from https://www.lfd.uci.edu/~gohlke/pythonlibs/<br>
Besides, you can install the LibSVM from packages in ./Wheel/ by using wheel.<br>
### File Description
The whole training process is in ./Train.py<br>
Practical functions is in ./Function.py<br>
Initial model is ./Train/model.<br>
There are few comments in the scripts, perhaps they will be made up in the future.<br>