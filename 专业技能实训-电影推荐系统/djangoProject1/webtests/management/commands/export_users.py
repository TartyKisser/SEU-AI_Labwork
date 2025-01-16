import sqlite3
import csv
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = '读取注册用户的名单'

    def handle(self, *args, **options):
        # 连接到 SQLite 数据库
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()

        # 选择表中的所有数据
        cursor.execute('SELECT * FROM webtests_user_info')
        # 获取列名并写入到 CSV 文件
        column_names = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

        # 指定 CSV 文件的路径
        csv_file_path = 'webtests/management/commands/users.csv'

        # 将数据写入 CSV 文件
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            # 先写入列名
            csv_writer.writerow(column_names)
            # 写入数据
            csv_writer.writerows(rows)

        # 关闭数据库连接
        conn.close()
