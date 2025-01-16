import csv
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = '获取用户喜欢的电影类型'

    def handle(self, *args, **options):
        csv_filepath = 'webtests/management/commands/users.csv'
        with open(csv_filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # 调用函数，指定 CSV 文件路径
            csv_file_path = 'webtests/management/commands/users_like.csv'
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                for row in reader:
                    username = row['username']
                    fav_types = []
                    column_names = reader.fieldnames  # 获取所有列的名称
                    for genre in column_names[3:]:
                        if row[genre] == '1':  # 排除名称列，并检查是否为喜欢的类型
                            fav_types.append(genre)
                    # 写入数据
                    csv_writer.writerow([username, ' '.join(fav_types)])
