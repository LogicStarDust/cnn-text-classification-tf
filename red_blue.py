# coding=utf-8
import random

# 三卡和每卡的牌面的颜色
ka = [['红', '红'], ['红', '蓝'], ['蓝', '蓝']]

# 第一面红色次数
first_red_num = 0

# 分别是第二面红色、蓝色出现次数和总测试次数
red_num = 0
blue_num = 0
test_num = 100000

for i in range(test_num):
    # 产生一个大于0小于3的随机数，作为抽三卡中哪一张的评判
    choose1 = random.randint(0, 2)
    # 产生一个大于0小于2的随机数，作为看卡片第一面的评判
    choose2 = random.randint(0, 1)

    # 如果本次抽的这张牌看到第一面为红，另一面为则红次数加一，否则蓝色加一
    if ka[choose1][choose2] == "红":
        first_red_num = first_red_num + 1
        if ka[choose1][1 - choose2] == "红":
            red_num = red_num + 1
        else:
            blue_num = blue_num + 1

print("第一面红色次数：%d\n第二面红色次数:%d,蓝色次数:%d\n红蓝比:%s :1" % (first_red_num,red_num, blue_num, red_num / blue_num,))
