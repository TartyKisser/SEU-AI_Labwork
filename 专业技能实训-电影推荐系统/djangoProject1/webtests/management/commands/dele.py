import sqlite3
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Description of your command'

    def handle(self, *args, **options):
        # 连接到 SQLite 数据库
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()

        # 执行删除命令
        cursor.execute("DELETE FROM webtests_rating")
        # 重置自增 ID
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='webtests_rating'")
        # 提交更改
        conn.commit()

        # 关闭连接
        cursor.close()
        conn.close()
