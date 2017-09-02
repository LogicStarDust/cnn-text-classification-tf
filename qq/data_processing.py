#!usr/bin/env python
# _*_ coding: utf-8 _*_

# 读入数据中有日期信息时,用此函数可以直接处理

import re, os,sys,io
import sqlite3
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030') #改变标准输出的默认编码
def get_name(words):
    # try match name from given string。(.*)\(.*\)  (.*)[<(].*[>)]
    matches = re.findall('(.*)[<(].*[>)]', words)
    try:
        namestr = matches[0]
    except:
        namestr = ''
    return namestr#.decode('utf-8') # decode

def get_date(words):
    # try match name from given string
    datepattern = re.compile('\d\d\d\d-\\d\d-\\d\d')
    matches = re.findall(datepattern, words)
    try:
        value = matches[0]
    except:
        value = ''
    return value

def getconn(path):
    conn = sqlite3.connect(path)
    try:
        conn = sqlite3.connect(path)
        return conn
    except:
        pass

def getcursor(conn):
    #return conn.cursor()
    try:
        return conn.cursor()
    except:
        pass

def closeconn(conn):
    conn.commit()
    try:
        conn.close()
    except:
        pass

def createtable(db):
    conn = getconn(db)
    sql = """CREATE TABLE records(
                            name TEXT NOT NULL,
                            date  TEXT NOT NULL,
                            PRIMARY KEY (name, date)
                            )"""
    cursor = getcursor(conn)
    cursor.execute(sql)
    conn.commit()
    conn.close()

def store_data(data):
    # 只存储一种格式 名字， 与日期， 表示一次签到记录
    db = 'db.sqlite'

    conn = getconn(db)
    cursor = getcursor(conn)
    sql = '''INSERT INTO records VALUES (?,?)'''
    for d in data:
        #print(d)
        name, date = d[0], d[1]
        # print(type(name))
        #name = name.encode('utf-8')
        # cursor.execute(sql, (name, date))
        try:
            cursor.execute(sql, (name, date))
        except:
            pass
    conn.commit()
    conn.close()
    # closeconn(conn)

def data_clean(filename):

    timepattern = re.compile('\d{1,2}:\d\d:\d\d')

    keypattern = re.compile('签到')

    recordkey = ''
    namelist = []
    try:
        namepattern = re.compile('\d\d.*\d\d.*\d\d')
        recorddate = re.findall(namepattern, filename)[0]
    except:
        recorddate = ''

    with open(filename,'r',encoding='utf-8') as f:
        line = f.readlines()
        for item in line:
            try:
                print("item:",item)
            except:
                print("exp")
            if re.findall(timepattern, item):
                recordkey = get_name(item)
                if get_date(item):
                    recorddate = get_date(item)
                print("匹配到日期,key：",recordkey,",date:",recorddate)
            # 接着匹配是否有签到
            if re.findall(keypattern, item):
                print("匹配到签到",item)
                if recordkey and recorddate:
                    namelist.append([recordkey, recorddate])
                    print((recordkey, recorddate))
                    recordkey = ''
                    # recorddate = ''
    if namelist:
        store_data(namelist) #数据存入数据库
    #return namelist
## Todo store namelist to db

# get formatted data from db
# neet to import a full name list into store

def travelfolder(rootdir=os.getcwd(),target=''):
    filelist = []
    for (dirpath, dirnames, filenames) in os.walk(rootdir):
        for filename in filenames:
            #filename = filename#.decode('gbk')
            fullname = os.path.join(dirpath,filename)
            data_clean(fullname)
    return filelist

def run():
    # 新建表
    db = 'db.sqlite'
    try:
        createtable(db)
    except:
        pass
    travelfolder('data')

if __name__ == "__main__":
    # 新建表
    db = 'db.sqlite'
    try:
        createtable(db)
    except:
        pass
    travelfolder('data')