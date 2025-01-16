import csv
from django.core.management.base import BaseCommand
from webtests.models import Movies_recommend
from django.db import connection

class Command(BaseCommand):
    help = '将推荐电影导入到数据集中'

    def handle(self, *args, **options):
        Movies_recommend.objects.all().delete()
        with connection.cursor() as cursor:
            if connection.vendor == 'sqlite':
                cursor.execute("DELETE FROM sqlite_sequence WHERE name='webtests_movies_recommend'")
        with open('webtests/management/commands/recommended_movies.csv', 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                _, created = Movies_recommend.objects.get_or_create(
                    name=row[1],
                    english_name=row[2],
                    directors=row[3],
                    writer=row[4],
                    actors=row[5],
                    rate=row[6],
                    style1=row[7],
                    style2=row[8],
                    style3=row[9],
                    country=row[10],
                    language=row[11],
                    date=row[12],
                    duration=row[13],
                    introduction=row[14],
                    dataID=row[15],
                    url=row[16],
                    pic=row[17],
                )
