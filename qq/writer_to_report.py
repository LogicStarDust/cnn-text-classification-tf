# !/usr/bin/python
# -*- coding: utf-8 -*-

import xlsxwriter
import time
from data_processing import getconn, getcursor

def query_database(db, date=''):
    sql =  '''
            SELECT *
            FROM records

            '''
    db = 'db.sqlite'
    #WHERE date like ?

    conn = getconn(db)
    cursor = getcursor(conn)

    result = cursor.execute(sql)
    return result.fetchall()

def getexcelrightcloumn(number):
    cloumns = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return cloumns[number-1]

def write_to_file(data, filename='tmp', title=''):
    #filename = 'abc'
    # formated data
    # tmp = []
    # # data = list(data)
    # for item in data:
    #     item = list(item)
    #     tmp.append((item[0], item[1]))

    excel_writer(data, ['name','date'],filename)

def excel_writer(data, titlelist=[], filename=''):
    if not filename:
        filename = time.strftime('%Y-%m-%d_%I%M%S',time.localtime()) + '.xlsx'
    #print filename
    filename = filename + '.xlsx'
    lines = len(data)
    columns = len(data[0])
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    titleformat = workbook.add_format()
    titleformat.set_bold()
    titleformat.set_font_size(10)
    titleformat.set_font_name('Microsoft yahei')
    titleformat.set_align('center')
    titleformat.set_align('vcenter')
    #worksheet
    rowformat = workbook.add_format()
    rowformat.set_font_size(10)
    rowformat.set_font_name('Microsoft yahei')
    rowformat.set_align('center')
    rowformat.set_align('vcenter')
    rowformat.set_text_wrap()
    rowformat.set_font_color('#0070C0')
    angle0 = 'A1'
    angle1 = getexcelrightcloumn(columns) + str(lines+1) #to include the title, the number of rows has to plus 1
    rightcolumn = getexcelrightcloumn(columns)
    columnrange = 'A:' + rightcolumn
    worksheet.set_column(columnrange, 12)
    rangetable = angle0 + ':' + angle1
    #worksheet.set_row(0, None, format1)
    if titlelist == []:
        titlelist = [' '] * columns

    datacolumn = []

    for i in titlelist:
        datacolumn.append({'header': i, 'format': rowformat})
    #print(datacolumn)
    worksheet.add_table(rangetable, {'data': data,
                                         'columns': datacolumn,
                                         'style': 'Table Style Light 9',
                                         }
                            )
    #worksheet.set_row(0, None, titleformat)
    worksheet.set_row(0, None, titleformat)
    #worksheet.freeze_panes(1, 0)  # Freeze the first row.

    #print validatesource.split(',')
    # rangevalidate = 'J2:J' + str(len(data) + 1)
    # worksheet.data_validation(rangevalidate, {'validate': 'list',
    #                               'source': validatesource.split(',')})
    # Widen the first column to make the text clear.

    workbook.close()
