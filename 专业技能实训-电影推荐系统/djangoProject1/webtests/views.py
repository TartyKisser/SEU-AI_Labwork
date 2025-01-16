from django.shortcuts import render, get_object_or_404, redirect
from django.core.management import call_command
from .models import User_info, Movies_recommend, Movies, Rating
from django.core.cache import cache
# Create your views here.


def login(request):
    movies = Movies.objects.all()
    if request.method == "GET":
        return render(request, 'webtests/login.html', {'movies': movies})
    elif request.method == "POST":
        post_data = request.POST
        username = post_data.get('user')
        pwd = post_data.get('pwd')
        if (User_info.objects.filter(username=username)
                and User_info.objects.filter(username=username).first().PassWord == pwd):
            # 保存用户名到缓存
            cache.set('current_user', username, 3600)
            call_command('recommend_movies')
            call_command('import_recommend_movies')
            return redirect('http://localhost:8000/movie_recommend')
        else:
            return render(request, 'webtests/login.html', {'tips': '用户名或密码错误！', 'movies': movies})


def signup(request):
    if request.method == "GET":
        return render(request, 'webtests/signup.html')
    elif request.method == "POST":
        post_data = request.POST
        username = post_data.get('user')
        pwd = post_data.get('pwd')
        pwd_confirm = post_data.get('pwd_confirm')

        if pwd == pwd_confirm:
            User_info.objects.create(username=username, PassWord=pwd)
            cache.set('current_user', username, 3600)
            call_command('export_users')
            call_command('users_like')
            call_command('adjust_users_like')
            return redirect('http://localhost:8000/preference')
        else:
            return render(request, 'webtests/signup.html', {'tips': '两次密码输入不一致！'})


def preference(request):
    if request.method == "GET":
        return render(request, 'webtests/preference.html', {'username': cache.get('current_user')})
    elif request.method == "POST":
        post_data = request.POST
        xuanyi = post_data.get('xuanyi') == 'True'
        fanzui = post_data.get('fanzui') == 'True'
        kehuan = post_data.get('kehuan') == 'True'
        tongxing = post_data.get('tongxing') == 'True'
        maoxian = post_data.get('maoxian') == 'True'
        kongbu = post_data.get('kongbu') == 'True'
        juqing = post_data.get('juqing') == 'True'
        aiqing = post_data.get('aiqing') == 'True'
        zainan = post_data.get('zainan') == 'True'
        xiju = post_data.get('xiju') == 'True'
        donghua = post_data.get('donghua') == 'True'
        qingse = post_data.get('qingse') == 'True'

        jingsong = post_data.get('jingsong') == 'True'
        jilupian = post_data.get('jilupian') == 'True'
        dongzuo = post_data.get('dongzuo') == 'True'
        qihuan = post_data.get('qihuan') == 'True'
        jiating = post_data.get('jiating') == 'True'
        yundong = post_data.get('yundong') == 'True'
        zhenrenxiu = post_data.get('zhenrenxiu') == 'True'
        wuxia = post_data.get('wuxia') == 'True'
        guzhuang = post_data.get('guzhuang') == 'True'
        lishi = post_data.get('lishi') == 'True'
        zhanzheng = post_data.get('zhanzheng') == 'True'
        tuokouxiu = post_data.get('tuokouxiu') == 'True'
        yinyue = post_data.get('yinyue') == 'True'
        ertong = post_data.get('ertong') == 'True'
        zhuanji = post_data.get('zhuanji') == 'True'
        xibu = post_data.get('xibu') == 'True'
        gewu = post_data.get('gewu') == 'True'
        duanpian = post_data.get('duanpian') == 'True'
        xiqu = post_data.get('xiqu') == 'True'
        heise = post_data.get('heise') == 'True'
        User_info.objects.filter(username=cache.get('current_user')).update(xuanyi=xuanyi, fanzui=fanzui, kehuan=kehuan,
                                                             tongxing=tongxing, maoxian=maoxian, kongbu=kongbu,
                                                             juqing=juqing, aiqing=aiqing, zainan=zainan, xiju=xiju,
                                                             donghua=donghua, qingse=qingse, jingsong=jingsong,
                                                            jilupian=jilupian, dongzuo=dongzuo, qihuan=qihuan,
                                                            jiating=jiating, yundong=yundong, zhenrenxiu=zhenrenxiu,
                                                            wuxia=wuxia, guzhuang=guzhuang, lishi=lishi,
                                                            zhanzheng=zhanzheng, tuokouxiu=tuokouxiu, yinyue=yinyue,
                                                            ertong=ertong, zhuanji=zhuanji, xibu=xibu, gewu=gewu,
                                                                            duanpian=duanpian, xiqu=xiqu, heise=heise)
        return redirect('http://localhost:8000/')


def movie_recommend(request):
    call_command('recommend_movies')
    call_command('import_recommend_movies')
    data_movies = Movies_recommend.objects.all()
    return render(request, 'webtests/movie_recommend.html', {'movie_list': data_movies})


from django.http import JsonResponse
def movie_detail(request, movie_id):
    movie = get_object_or_404(Movies,dataID=movie_id)
    if request.method == "POST" and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # 处理 AJAX POST 请求
        rating = request.POST.get('rating')
        username = cache.get('current_user')
        if username and rating:
            Rating.objects.update_or_create(
                username=username,
                dataID=movie_id,
                defaults={'rating': rating}
            )
            call_command('append_new_rating')
    return render(request,'webtests/movie_detail.html',{'movie':movie})


def movie_all(request):
    data_movies = Movies.objects.all()
    return render(request, 'webtests/movie_all.html', {'movie_list': data_movies})


def search_results(request):
    query = request.GET.get('query', '')
    movies = Movies.objects.filter(name__icontains=query)
    return render(request, 'webtests/search_result.html', {'movies': movies})


def user_homepage(request):
    username = cache.get('current_user')
    query = Rating.objects.filter(username=username)
    data_ids = query.values_list('dataID', flat=True)
    movies = Movies.objects.filter(dataID__in=data_ids)
    # 获取用户的偏好标签
    user_info = User_info.objects.get(username=username)
    tags = {
        'xuanyi': user_info.xuanyi,
        'fanzui': user_info.fanzui,
        'kehuan': user_info.kehuan,
        'tongxing': user_info.tongxing,
        'maoxian': user_info.maoxian,
        'kongbu': user_info.kongbu,
        'juqing': user_info.juqing,
        'aiqing': user_info.aiqing,
        'zainan': user_info.zainan,
        'xiju': user_info.xiju,
        'donghua': user_info.donghua,
        'qingse': user_info.qingse,

        'jingsong': user_info.jingsong,
        'jilupian': user_info.jilupian,
        'dongzuo': user_info.dongzuo,
        'qihuan': user_info.qihuan,
        'jiating': user_info.jiating,
        'yundong': user_info.yundong,
        'zhenrenxiu': user_info.zhenrenxiu,
        'wuxia': user_info.wuxia,
        'guzhuang': user_info.guzhuang,
        'lishi': user_info.lishi,
        'zhanzheng': user_info.zhanzheng,
        'tuokouxiu': user_info.tuokouxiu,
        'yinyue': user_info.yinyue,
        'ertong': user_info.ertong,
        'zhuanji': user_info.zhuanji,
        'xibu': user_info.xibu,
        'gewu': user_info.gewu,
        'duanpian': user_info.duanpian,
        'xiqu': user_info.xiqu,
        'heise': user_info.heise
    }
    # 标签与中文标签的映射
    tag_names = {
        'xuanyi': '悬疑',
        'fanzui': '犯罪',
        'kehuan': '科幻',
        'tongxing': '同性',
        'maoxian': '冒险',
        'kongbu': '恐怖',
        'juqing': '剧情',
        'aiqing': '爱情',
        'zainan': '灾难',
        'xiju': '喜剧',
        'donghua': '动画',
        'qingse': '情色',
        'jingsong': '惊悚',
        'jilupian': '纪录片',
        'dongzuo': '动作',
        'qihuan': '奇幻',
        'jiating': '家庭',
        'yundong': '运动',
        'zhenrenxiu': '真人秀',
        'wuxia': '武侠',
        'guzhuang': '古装',
        'lishi': '历史',
        'zhanzheng': '战争',
        'tuokouxiu': '脱口秀',
        'yinyue': '音乐',
        'ertong': '儿童',
        'zhuanji': '传记',
        'xibu': '西部',
        'gewu': '歌舞',
        'duanpian': '短片',
        'xiqu': '戏曲',
        'heise': '黑色电影'
    }

    # 生成一个包含中文标签的字典
    tags_display = {tag_names[tag]: is_liked for tag, is_liked in tags.items() if is_liked}

    return render(request, 'webtests/user_homepage.html', {'movies': movies, 'tags': tags_display})

