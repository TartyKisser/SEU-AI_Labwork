import sqlite3
import csv
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = '读取最新的评分'

    def handle(self, *args, **options):
        # 连接到 SQLite 数据库
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()

        # 选择表中的所有数据
        cursor.execute('SELECT username, dataID, rating FROM webtests_rating ORDER BY id DESC LIMIT 1')
        row = cursor.fetchone()

        # 指定 CSV 文件的路径
        csv_file_path = 'webtests/management/commands/douban_users.csv'

        try:
            with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
                last_line = list(csv.reader(csv_file))[-1]
                last_row_number = int(last_line[0])
        except (FileNotFoundError, IndexError, ValueError):
            last_row_number = 0

            # 将数据写入 CSV 文件
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([last_row_number + 1] + list(row))

        # 关闭数据库连接
        conn.close()
