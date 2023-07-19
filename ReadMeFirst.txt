
The model is placed in the 'model' folder, with two.
One is the resnet18 model, which is smaller in size and faster in speed;
The other is the resnet50 model, which has a slightly larger volume but higher accuracy.

How to test:
1. Run code '3.3Test_model.py', you can test a set of images and save the test results to 'pred_result.csv';
2. Run code '3.4Model_power.py', you can read the accuracy and other information on the console and see the Confusion matrix of the test results;
3. Run code '3.5Extracting Error Images.py', and the detected images will be saved to the folder 'WrongResult';

How to replace the tested model:
1. Find the configuration file: information/config.yaml and open it;
2. Change the myNet's value to 'resnet50' or 'resnet18' to test the corresponding model;
(Be careful to leave spaces before and after " : ")

How to replace images for testing:
1. Find the configuration file: information/config.yaml and open it;
2. Change the test_id's value to 0, 1, 2, 3, 4 to test the corresponding image group;

====================Chinese Version=====================
模型放在model文件夹下，有两个。
一个是resnet18的model，体积更小速度更快；
另一个是resnet50的model，体积稍大，但准确度更高。

如何测试：
1，运行代码3.3Test_model.py，你能测试一组图片，测试结果保存至pred_result.csv中；
2，运行代码3.4Model_power.py，你能在控制台读取精确度等信息，并能看到测试结果混淆矩阵；
3，运行代码3.5Extracting Error Images.py，检测错误的图片会保存至文件夹WrongResult中；

如何更换测试的模型：
1，找到配置文件：information/config.yaml 并打开；
2，将其中的myNet对应的值改为 resnet50 或者resnet18 可以测试对应model；
（注意在“:”前后留有空格）

如何更换测试的图片：
1，找到配置文件：information/config.yaml 并打开；
2，将其中的test_id对应的值改为0，1，2，3，4可以测试对应的图片组；