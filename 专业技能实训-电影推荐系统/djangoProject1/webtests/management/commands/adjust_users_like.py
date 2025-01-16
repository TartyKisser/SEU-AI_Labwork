import re
import csv
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = '拼音与汉字对应'

    def handle(self, *args, **options):
        # 输入文件路径
        input_csv_path = 'webtests/management/commands/users_like.csv'
        # 输出文件路径
        output_csv_path = 'webtests/management/commands/users_like_adjust.csv'

        with (open(input_csv_path, newline='', encoding='utf-8') as csvfile, \
                open(output_csv_path, 'w', newline='', encoding='utf-8') as outputfile):
            reader = csv.reader(csvfile)  # 使用 csv.reader 而非 csv.DictReader
            writer = csv.writer(outputfile)  # 使用 csv.writer 进行写入

            for row in reader:
                # 第二列的索引为 1
                text = row[1]
                # 替换文本内容
                text = (
                    text.replace('qingse', '情色')
                    .replace('xuanyi', '悬疑')
                    .replace('fanzui', '犯罪')
                    .replace('kehuan', '科幻')
                    .replace('tongxing', '同性')
                    .replace('maoxian', '冒险')
                    .replace('kongbu', '恐怖')
                    .replace('juqing', '剧情')
                    .replace('aiqing', '爱情')
                    .replace('zainan', '灾难')
                    .replace('xiju', '喜剧')
                    .replace('donghua', '动画')
                    .replace('jingsong', '惊悚')
                    .replace('jilupian', '纪录片')
                    .replace('dongzuo', '动作')
                    .replace('qihuan', '奇幻')
                    .replace('jiating', '家庭')
                    .replace('yundong', '运动')
                    .replace('zhenrenxiu', '真人秀')
                    .replace('wuxia', '武侠')
                    .replace('guzhuang', '古装')
                    .replace('lishi', '历史')
                    .replace('zhanzheng', '战争')
                    .replace('tuokouxiu', '脱口秀')
                    .replace('yinyue', '音乐')
                    .replace('ertong', '儿童')
                    .replace('zhuanji', '传记')
                    .replace('xibu', '西部')
                    .replace('gewu', '歌舞')
                    .replace('duanpian', '短片')
                    .replace('xiqu', '戏曲')
                    .replace('heise', '黑色电影')
                )

                row[1] = text
                writer.writerow(row)
