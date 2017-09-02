from data_processing import run
from writer_to_report import query_database, write_to_file
run()

db = 'db.sqlite'
data = query_database(db)
if data:
    write_to_file(data)
else:
    print('data is None')