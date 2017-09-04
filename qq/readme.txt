update:2017-09-04
author：Wang Guodong（QQ:657496651）
env:python 3.6,qq 8.9.4
declare：
    把在qq消息中心导出的txt格式的聊天记录放到data目录下，
    删除db.sqlite和tmp.xlsx文件，
    直接运行run.py，
    得到tmp.xlsx文件即为统计签到信息表
本次更新：
    1.修复不能匹配昵称后面跟尖括号的问题
    2.修复不能匹配时间的小时部分只有一个数字的问题（凌晨1点到上午9点）
已知问题：
    1.现在版本程序分析的结果excel有两列：（日期 时间 用户名）（日期），可以把第一列用excel的分列功能按长度分成三列
    2.发现有导出数据有没有用户名，但是聊天记录有的情况。累觉不爱，各位针对统计出来的表格中用户名空白手动纠正吧



update: 2017-07-18
env: python 3.6
start from : run.py
